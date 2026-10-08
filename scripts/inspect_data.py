"""Read the included JSONL files; never downloads or trains a model."""

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAPPING = {"WORK": "R31", "PERSONAL": "R58", "SHOPPING": "R74"}


def records(path: Path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            item = json.loads(line)
            messages = item["messages"]
            if len(messages) != 2 or [m["role"] for m in messages] != ["user", "assistant"]:
                raise ValueError(f"{path}:{number}: expected one user and one assistant message")
            yield messages


for version in ("routes_v1", "routes_v2", "routes_v3"):
    print(f"\n{version}")
    split_notes = {}
    for split in ("train", "test"):
        path = ROOT / "data" / version / f"{split}.jsonl"
        rows = list(records(path))
        notes = [messages[0]["content"].split("\nNote: ", 1)[1] for messages in rows]
        if len(notes) != len(set(notes)):
            raise ValueError(f"duplicate notes in {path}")
        split_notes[split] = set(notes)
        categories = Counter()
        for messages in rows:
            answer = messages[1]["content"]
            category = next((name for name, code in MAPPING.items() if code in answer), None)
            if category is None:
                raise ValueError(f"unexpected label in {path}: {answer!r}")
            categories[category] += 1
        print(f"  {split}: {len(rows)} records; {dict(categories)}")
        print(f"  first question: {rows[0][0]['content']!r}")
        print(f"  first answer: {rows[0][1]['content']!r}")
    overlap = split_notes["train"] & split_notes["test"]
    if overlap:
        raise ValueError(f"train/test overlap in {version}: {sorted(overlap)}")
    print("  train/test note overlap: none")
