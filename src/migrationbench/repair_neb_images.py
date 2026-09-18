#!/usr/bin/env python3
"""Repair NEB images with conservative physics heuristics before MLFF/QE.

Modes:
- unwrap: make each image continuous with the previous image under PBC.
- smooth-host: unwrap, then linearly interpolate host atoms between endpoints while
  keeping the migrant path from the input images.
- idpp: rebuild all intermediate images from endpoints using ASE IDPP.

The script writes an extxyz and a manifest. It does not create a final barrier;
its purpose is only to make the path initialization physically sane before NEB.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.mep import NEB

from analyze_neb_path_topology import read_qe_image


def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def choose_migrant(images, element):
    symbols = images[0].get_chemical_symbols()
    candidates = [i for i, s in enumerate(symbols) if s == element]
    if not candidates:
        raise ValueError('No atoms with migrant element %r' % element)
    if len(candidates) == 1:
        return candidates[0]
    pos0 = images[0].get_positions(); pos1 = images[-1].get_positions()
    return candidates[int(np.argmax([np.linalg.norm(pos1[i] - pos0[i]) for i in candidates]))]


def unwrap_like_previous(prev, cur):
    fixed = cur.copy()
    if cur.cell.rank != 3 or not any(cur.pbc):
        return fixed
    # Never wrap the already-unwrapped reference image back into the unit cell.
    # Doing so can reintroduce a full lattice-vector jump between adjacent images.
    prev_s = prev.get_scaled_positions(wrap=False)
    cur_s = cur.get_scaled_positions(wrap=False)
    delta = cur_s - prev_s
    delta -= np.rint(delta)
    fixed.set_scaled_positions(prev_s + delta)
    return fixed


def unwrap_images(images):
    out = [images[0].copy()]
    for img in images[1:]:
        out.append(unwrap_like_previous(out[-1], img))
    return out


def attach_cell_from_qe_template(images, template_path):
    parsed = read_qe_image(Path(template_path).resolve())
    if parsed["cell"] is None:
        raise ValueError("Cell template has no CELL_PARAMETERS: %s" % template_path)
    cell = np.asarray(parsed["cell"], dtype=float)
    for image in images:
        if image.cell.rank == 3 and not np.allclose(image.cell.array, cell, atol=1.0e-8):
            raise ValueError("Input trajectory cell disagrees with explicit template")
        image.set_cell(cell, scale_atoms=False)
        image.set_pbc((True, True, True))
    return images


def resample_images_by_arc_length(images, count):
    if count is None or count == len(images):
        return [image.copy() for image in images]
    if count < 2:
        raise ValueError("resample-images must be at least 2")
    positions = [image.get_positions() for image in images]
    segments = [
        float(np.sqrt(np.mean(np.sum((right - left) ** 2, axis=1))))
        for left, right in zip(positions, positions[1:])
    ]
    total = sum(segments)
    if total < 1.0e-14:
        raise ValueError("Cannot resample a zero-length path")
    cumulative = [0.0]
    for segment in segments:
        cumulative.append(cumulative[-1] + segment / total)
    output = []
    cursor = 0
    for target in np.linspace(0.0, 1.0, count):
        while cursor + 1 < len(cumulative) and cumulative[cursor + 1] < target:
            cursor += 1
        if cursor + 1 == len(cumulative):
            output.append(images[-1].copy())
            continue
        low, high = cumulative[cursor], cumulative[cursor + 1]
        fraction = 0.0 if high - low < 1.0e-14 else (target - low) / (high - low)
        image = images[cursor].copy()
        image.set_positions(positions[cursor] + fraction * (positions[cursor + 1] - positions[cursor]))
        output.append(image)
    output[0] = images[0].copy()
    output[-1] = images[-1].copy()
    return output


def smooth_between_endpoints(images, migrant_index, smooth_migrant=False):
    out = [img.copy() for img in images]
    first = out[0]
    last = unwrap_like_previous(first, out[-1])
    if first.cell.rank == 3 and any(first.pbc):
        s0 = first.get_scaled_positions(wrap=False)
        s1 = last.get_scaled_positions(wrap=False)
        delta = s1 - s0
        delta -= np.rint(delta)
        for k, img in enumerate(out):
            t = k / float(len(out) - 1)
            s = s0 + t * delta
            if not smooth_migrant:
                s[migrant_index] = img.get_scaled_positions(wrap=False)[migrant_index]
            img.set_scaled_positions(s)
    else:
        p0 = first.get_positions()
        p1 = last.get_positions()
        for k, img in enumerate(out):
            t = k / float(len(out) - 1)
            pos = p0 + t * (p1 - p0)
            if not smooth_migrant:
                pos[migrant_index] = img.get_positions()[migrant_index]
            img.set_positions(pos)
    return out


def idpp_from_endpoints(images, n_images, mic):
    first = images[0].copy()
    last = images[-1].copy()
    out = [first]
    out.extend(first.copy() for _ in range(n_images - 2))
    out.append(last)
    neb = NEB(out, climb=False, method='improvedtangent')
    neb.interpolate(method='idpp', mic=mic)
    return out


def min_pair_distance(atoms):
    distances = atoms.get_all_distances(mic=True)
    distances[np.tril_indices(len(atoms))] = np.inf
    pair = np.unravel_index(np.argmin(distances), distances.shape)
    return float(distances[pair]), tuple(map(int, pair))


def repel_close_pairs(images, min_distance, steps, step_size, migrant_index, policy):
    repaired = [img.copy() for img in images]
    repair_actions = []
    for image_index, atoms in enumerate(repaired[1:-1], start=1):
        before_positions = atoms.get_positions().copy()
        before_distance, before_pair = min_pair_distance(atoms)
        unresolved_reason = None
        iterations = 0
        if policy == 'none':
            after_distance, after_pair = before_distance, before_pair
            repair_actions.append({
                'image_index_zero_based': image_index,
                'iterations': 0,
                'before_min_pair_distance_A': before_distance,
                'before_pair_indices': list(before_pair) if before_pair else None,
                'after_min_pair_distance_A': after_distance,
                'after_pair_indices': list(after_pair) if after_pair else None,
                'moved_atom_indices': [],
                'maximum_atom_displacement_A': 0.0,
                'migrant_displacement_A': 0.0,
                'unresolved_reason': None,
                'threshold_pass': after_distance >= min_distance - 1.0e-8,
            })
            continue
        for iteration in range(steps):
            d, pair = min_pair_distance(atoms)
            if d >= min_distance or pair is None:
                break
            i, j = pair
            vector = atoms.get_distance(i, j, mic=True, vector=True)
            vector_norm = np.linalg.norm(vector)
            if vector_norm < 1.0e-8:
                vector = np.array([1.0, 0.0, 0.0]); vector_norm = 1.0
            direction = vector / vector_norm
            correction = min(step_size, min_distance - d)
            if policy == 'migrant-only':
                if i == migrant_index:
                    atoms.positions[i] -= correction * direction
                elif j == migrant_index:
                    atoms.positions[j] += correction * direction
                else:
                    unresolved_reason = 'closest_pair_does_not_include_migrant'
                    break
            else:
                atoms.positions[i] -= 0.5 * correction * direction
                atoms.positions[j] += 0.5 * correction * direction
            iterations = iteration + 1
        after_distance, after_pair = min_pair_distance(atoms)
        displacements = np.linalg.norm(atoms.get_positions() - before_positions, axis=1)
        repair_actions.append({
            'image_index_zero_based': image_index,
            'iterations': iterations,
            'before_min_pair_distance_A': before_distance,
            'before_pair_indices': list(before_pair) if before_pair else None,
            'after_min_pair_distance_A': after_distance,
            'after_pair_indices': list(after_pair) if after_pair else None,
            'moved_atom_indices': [int(index) for index in np.where(displacements > 1.0e-12)[0]],
            'maximum_atom_displacement_A': float(displacements.max()),
            'migrant_displacement_A': float(displacements[migrant_index]),
            'unresolved_reason': unresolved_reason,
            'threshold_pass': after_distance >= min_distance - 1.0e-8,
        })
    return repaired, repair_actions


def path_stats(images, migrant_index):
    min_pairs = [min_pair_distance(img)[0] for img in images]
    migrant_steps = []
    for a, b in zip(images[:-1], images[1:]):
        if a.cell.rank == 3 and any(a.pbc):
            ds = b.get_scaled_positions()[migrant_index] - a.get_scaled_positions()[migrant_index]
            ds -= np.rint(ds)
            vec = np.dot(ds, a.cell.array)
        else:
            vec = b.positions[migrant_index] - a.positions[migrant_index]
        migrant_steps.append(float(np.linalg.norm(vec)))
    return {
        'min_pair_distance_A': min(min_pairs),
        'max_migrant_step_A': max(migrant_steps) if migrant_steps else 0.0,
        'total_migrant_path_A': sum(migrant_steps),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--mode', choices=['unwrap', 'smooth-host', 'smooth-all', 'idpp'], default='smooth-host')
    parser.add_argument('--migrant-element', default='Cr')
    parser.add_argument('--n-images', type=int)
    parser.add_argument('--resample-images', type=int, help='Resample the inherited curved path by configuration arc length before repulsion.')
    parser.add_argument('--cell-template', help='QE input providing mandatory CELL_PARAMETERS for plain XYZ paths.')
    parser.add_argument('--mic', action='store_true', default=True)
    parser.add_argument('--min-pair-distance-A', type=float, default=1.7)
    parser.add_argument('--repel-steps', type=int, default=20)
    parser.add_argument('--repel-step-size-A', type=float, default=0.08)
    parser.add_argument('--repel-policy', choices=['none', 'migrant-only', 'both'], default='migrant-only')
    parser.add_argument('--fail-on-invalid', action='store_true', help='Write the audit manifest, then fail if any image misses the distance gate.')
    args = parser.parse_args()

    source = Path(args.images).resolve()
    images = read(source, index=':')
    if len(images) < 2:
        raise ValueError('Need at least two images')
    source_had_cell = all(image.cell.rank == 3 and any(image.pbc) for image in images)
    if args.cell_template:
        images = attach_cell_from_qe_template(images, args.cell_template)
    if not all(image.cell.rank == 3 and any(image.pbc) for image in images):
        raise ValueError('Periodic NEB repair requires cell-bearing images or --cell-template')
    migrant = choose_migrant(images, args.migrant_element)
    before = path_stats(images, migrant)

    if args.mode == 'idpp':
        repaired = idpp_from_endpoints(images, args.n_images or len(images), args.mic)
    else:
        repaired = unwrap_images(images)
        if args.mode == 'smooth-host':
            repaired = smooth_between_endpoints(repaired, migrant, smooth_migrant=False)
        elif args.mode == 'smooth-all':
            repaired = smooth_between_endpoints(repaired, migrant, smooth_migrant=True)
    repaired = resample_images_by_arc_length(repaired, args.resample_images)
    repaired, repair_actions = repel_close_pairs(
        repaired,
        args.min_pair_distance_A,
        args.repel_steps,
        args.repel_step_size_A,
        migrant,
        args.repel_policy,
    )
    after = path_stats(repaired, migrant)

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    for idx, atoms in enumerate(repaired):
        atoms.info['repair_mode'] = args.mode
        atoms.info['repair_image_index'] = idx
        atoms.info['migrant_element'] = args.migrant_element
    write(output, repaired)
    manifest = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_images': str(source),
        'source_sha256': sha256_file(source),
        'output_images': str(output),
        'output_sha256': sha256_file(output),
        'mode': args.mode,
        'migrant_element': args.migrant_element,
        'migrant_index_zero_based': migrant,
        'before': before,
        'after': after,
        'parameters': {
            'n_images': len(repaired),
            'source_n_images': len(images),
            'resample_images': args.resample_images,
            'mic': args.mic,
            'min_pair_distance_A': args.min_pair_distance_A,
            'repel_steps': args.repel_steps,
            'repel_step_size_A': args.repel_step_size_A,
            'repel_policy': args.repel_policy,
        },
        'repair_actions': repair_actions,
        'source_had_periodic_cell': source_had_cell,
        'cell_template': str(Path(args.cell_template).resolve()) if args.cell_template else None,
        'cell_template_sha256': sha256_file(Path(args.cell_template).resolve()) if args.cell_template else None,
        'output_pbc': list(map(bool, repaired[0].pbc)),
        'endpoint_min_pair_distances_A': [
            min_pair_distance(repaired[0])[0], min_pair_distance(repaired[-1])[0]
        ],
        'all_intermediate_images_pass': all(row['threshold_pass'] for row in repair_actions),
        'endpoints_pass': all(
            min_pair_distance(image)[0] >= args.min_pair_distance_A - 1.0e-8
            for image in (repaired[0], repaired[-1])
        ),
        'warning': 'This is a path-initialization repair only. It must be followed by MLFF and/or QE NEB validation.',
    }
    Path(args.manifest).resolve().write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'mode': args.mode, 'before': before, 'after': after}, indent=2))
    if args.fail_on_invalid and not (manifest['all_intermediate_images_pass'] and manifest['endpoints_pass']):
        raise SystemExit('Repaired path failed the configured geometry gate; see manifest')


if __name__ == '__main__':
    main()
