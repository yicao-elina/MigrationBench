#!/usr/bin/env python3
"""Small helper to update node status in state/project_nodes.yaml."""

import argparse
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default="state/project_nodes.yaml")
    parser.add_argument("--node", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--note")
    args = parser.parse_args()
    if yaml is None:
        raise SystemExit("PyYAML is required for update_node_status.py")
    state_path = Path(args.state)
    payload = yaml.safe_load(state_path.read_text())
    node = payload["nodes"][args.node]
    node["status"] = args.status
    if args.note:
        node.setdefault("notes", []).append(args.note)
    state_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    print("%s -> %s" % (args.node, args.status))


if __name__ == "__main__":
    main()
