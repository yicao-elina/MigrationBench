#!/usr/bin/env python3
"""Read-only SHA/stat recheck for declared personal Scratch16 Revision 1 paths.

Compatible with Python 3.6 on Rockfish. Never follows a final symlink and refuses
paths outside the two declared personal input/run roots.
"""
from __future__ import print_function
import argparse, csv, hashlib, json, os, platform, stat, sys
from datetime import datetime

ROOTS = {
    '/scratch16/pclancy3/yi/revision1_migrationbench_inputs': 'INPUTS',
    '/scratch16/pclancy3/yi/revision1_migrationbench_runs': 'RUNS',
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def verify(row):
    path = row['path']
    out = dict(row)
    out.update({'current_bytes': '', 'current_sha256': '', 'mtime_ns': '',
                'resolved_within_root': '', 'verification_status': 'NOT_CHECKED',
                'observed_type': '', 'symlink_target': ''})
    root = None
    for candidate in ROOTS:
        if path.startswith(candidate + '/'):
            root = candidate
            break
    if root is None or os.path.normpath(path) != path:
        out['verification_status'] = 'OUTSIDE_DECLARED_PERSONAL_ROOT'
        return out
    real = os.path.realpath(path)
    try:
        if os.path.commonpath([root, real]) != root:
            out['verification_status'] = 'RESOLVES_OUTSIDE_DECLARED_ROOT'
            return out
    except Exception:
        out['verification_status'] = 'ROOT_CONTAINMENT_UNCHECKED'
        return out
    out['resolved_within_root'] = 'YES'
    try:
        st = os.lstat(path)
    except OSError as exc:
        out['verification_status'] = 'MISSING' if exc.errno == 2 else 'UNREADABLE'
        out['verification_error'] = str(exc)
        return out
    out['current_bytes'] = str(st.st_size)
    out['mtime_ns'] = str(getattr(st, 'st_mtime_ns', int(st.st_mtime * 1000000000)))
    if stat.S_ISLNK(st.st_mode):
        out['observed_type'] = 'SYMLINK'
        try:
            out['symlink_target'] = os.readlink(path)
        except OSError:
            pass
        out['verification_status'] = 'SYMLINK_NOT_FOLLOWED'
        return out
    if stat.S_ISDIR(st.st_mode):
        out['observed_type'] = 'DIRECTORY'
        out['verification_status'] = 'PRESENT_DIRECTORY'
        return out
    if not stat.S_ISREG(st.st_mode):
        out['observed_type'] = 'OTHER'
        out['verification_status'] = 'NOT_REGULAR_FILE'
        return out
    out['observed_type'] = 'REGULAR_FILE'
    try:
        expected_size = int(row['expected_bytes']) if row['expected_bytes'] else None
        expected_sha = row['expected_sha256']
        metadata = row['target_kind'] == 'METADATA'
        should_hash = metadata or bool(expected_sha) or st.st_size <= 65536
        if should_hash:
            digest = sha256(path)
            out['current_sha256'] = digest
            if expected_size is not None and st.st_size != expected_size:
                out['verification_status'] = 'EXPECTED_SIZE_MISMATCH'
            elif expected_sha and digest != expected_sha:
                out['verification_status'] = 'EXPECTED_SHA_MISMATCH'
            elif metadata:
                out['verification_status'] = 'METADATA_SIZE_SHA_MATCH'
            elif expected_sha:
                out['verification_status'] = 'TARGET_SIZE_SHA_MATCH'
            else:
                out['verification_status'] = 'CURRENT_SHA_ONLY_NO_PRIOR_DIGEST'
        elif expected_size is not None and st.st_size == expected_size:
            out['verification_status'] = 'TARGET_SIZE_MATCH_SHA_NOT_RECHECKED'
        elif expected_size is not None:
            out['verification_status'] = 'EXPECTED_SIZE_MISMATCH'
        else:
            out['verification_status'] = 'PRESENT_SIZE_ONLY_NO_PRIOR_SIZE'
    except OSError as exc:
        out['verification_status'] = 'HASH_READ_ERROR'
        out['verification_error'] = str(exc)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', required=True)
    ap.add_argument('--output-tsv', required=True)
    ap.add_argument('--receipt', required=True)
    args = ap.parse_args()
    with open(args.targets, 'r') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    results = [verify(r) for r in rows]
    fields = sorted(set(k for r in results for k in r.keys()))
    with open(args.output_tsv, 'w') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t', lineterminator='\n')
        w.writeheader(); w.writerows(results)
    counts = {}
    for r in results:
        counts[r['verification_status']] = counts.get(r['verification_status'], 0) + 1
    def digest(path):
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(block)
        return h.hexdigest()
    receipt = {
        'schema_version': '1',
        'captured_utc': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'host': os.uname()[1],
        'python_version': platform.python_version(),
        'declared_roots': sorted(ROOTS.keys()),
        'target_rows': len(rows),
        'verification_counts': counts,
        'targets_sha256': digest(args.targets),
        'results_tsv_sha256': digest(args.output_tsv),
        'read_only': True,
        'symlink_policy': 'lstat only; final symlink targets are not followed',
        'hash_policy': 'all metadata files; candidate files only when prior SHA exists or current size <= 64 KiB'
    }
    with open(args.receipt, 'w') as f:
        json.dump(receipt, f, indent=2, sort_keys=True); f.write('\n')
    print(json.dumps(receipt, indent=2, sort_keys=True))

if __name__ == '__main__':
    main()
