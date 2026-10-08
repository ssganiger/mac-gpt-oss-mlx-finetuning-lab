"""Print one exact dataset prompt for command substitution in mlx_lm.generate."""

import argparse
import json
import sys
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--version", choices=("routes_v1", "routes_v2", "routes_v3"), default="routes_v3")
parser.add_argument("--split", choices=("train", "test"), default="test")
parser.add_argument("--index", type=int, default=0, help="Zero-based line number")
args = parser.parse_args()

path = Path(__file__).resolve().parents[1] / "data" / args.version / f"{args.split}.jsonl"
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
if not 0 <= args.index < len(rows):
    parser.error(f"--index must be between 0 and {len(rows) - 1}")

sys.stdout.write(rows[args.index]["messages"][0]["content"])
