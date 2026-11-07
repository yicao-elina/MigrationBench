#!/usr/bin/env python3
"""Build three Cr NEB representatives from relaxed Rockfish path images.

The source sb2te3.path files contain ten relaxed image coordinate sets.  We
reuse those coordinates, replace the source dopant species by Cr, and emit a
fresh explicit-image neb.in.  The original nine endpoint-only inputs remain
untouched.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path('/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/task3_endpoint_only_neb/remote_snapshot')
OUT = ROOT / 'inputs_relaxed_images' / 'representative_3'
CASES = [
    ('within_QL', '2L-Cu/NEB/neb-QLs', '2L-Cu__neb-QLs', 'Cu'),
    ('gap_to_QL', '2L-Cu/NEB/neb-gap2QL', '2L-Cu__neb-gap2QL', 'Cu'),
    ('within_gap', '2L-Cu/NEB/neb-gap-diffuse', '2L-Cu__neb-gap-diffuse', 'Cu'),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_species(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().upper().startswith('ATOMIC_POSITIONS')) + 1
    species = []
    for line in lines[start:]:
        if not line.strip() or line.strip().upper().startswith(('K_POINTS', 'CELL_PARAMETERS')):
            break
        fields = line.split()
        if len(fields) >= 4:
            species.append(fields[0])
    if len(species) != 81:
        raise ValueError(f'{path}: expected 81 atoms, found {len(species)}')
    return species


def path_images(path: Path) -> list[list[tuple[float, float, float]]]:
    lines = path.read_text().splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith('Image:')]
    images = []
    for pos, start in enumerate(starts):
        stop = starts[pos + 1] if pos + 1 < len(starts) else len(lines)
        body = [line for line in lines[start + 2:stop] if line.strip()]
        if len(body) != 81:
            raise ValueError(f'{path}: image {pos + 1} has {len(body)} atoms')
        images.append([tuple(float(x) for x in line.split()[:3]) for line in body])
    if len(images) != 10:
        raise ValueError(f'{path}: expected 10 images, found {len(images)}')
    return images


def replace_pseudo_dir(text: str) -> str:
    return re.sub(
        r"(pseudo_dir\s*=\s*')[^']+(')",
        r"\1/home/ycao73/ONCVPseudoPack/SG15_v1.2/UPF/PBE/\2",
        text,
        flags=re.I,
    )


def explicit_positions(species: list[str], images: list[list[tuple[float, float, float]]]) -> str:
    blocks = ['BEGIN_POSITIONS']
    for image_index, coords in enumerate(images):
        blocks.append('FIRST_IMAGE' if image_index == 0 else 'LAST_IMAGE' if image_index == len(images) - 1 else 'INTERMEDIATE_IMAGE')
        blocks.append('ATOMIC_POSITIONS angstrom')
        blocks.extend(f'{element:<3s} {x:20.12f} {y:20.12f} {z:20.12f}' for element, (x, y, z) in zip(species, coords))
    blocks.append('END_POSITIONS')
    return '\n'.join(blocks)


def main() -> None:
    manifest = []
    for mechanism, rel, case, old in CASES:
        srcdir = SRC / rel
        base = ROOT / 'inputs' / case
        neb_text = replace_pseudo_dir((base / 'neb.in').read_text())
        neb_text = re.sub(r"restart_mode\s*=\s*'[^']+'", "restart_mode = 'from_scratch'", neb_text)
        begin = neb_text.index('BEGIN_POSITIONS')
        end = neb_text.index('END_POSITIONS') + len('END_POSITIONS')
        species = atomic_species(base / 'pw_1.in')
        species = ['Cr' if x == old else x for x in species]
        images = path_images(srcdir / 'sb2te3.path')
        output = OUT / mechanism / 'neb.in'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(neb_text[:begin] + explicit_positions(species, images) + neb_text[end:])
        for companion in ('pw_1.in', 'pw_2.in'):
            target = output.parent / companion
            target.write_text(replace_pseudo_dir((base / companion).read_text()))
        manifest.append((mechanism, case, str(srcdir / 'sb2te3.path'), str(output), sha(srcdir / 'sb2te3.path'), sha(output)))

    manifest_path = OUT / 'representative_3_manifest.tsv'
    manifest_path.write_text('mechanism\tsource_case\tsource_relaxed_path\toutput_neb\tsource_sha256\toutput_sha256\n' + '\n'.join('\t'.join(row) for row in manifest) + '\n')
    print(f'generated {len(CASES)} explicit-image Cr NEB representatives under {OUT}')


if __name__ == '__main__':
    main()
