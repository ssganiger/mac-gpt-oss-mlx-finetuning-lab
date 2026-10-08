"""Show chat-template and answer token boundaries without loading model weights."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "routes_v3" / "train.jsonl"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--model",
    type=Path,
    default=ROOT / "models" / "gpt-oss-20b-mlx",
    help="Local model/tokenizer directory",
)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model)
records = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines()]

for category in ("WORK", "PERSONAL", "SHOPPING"):
    item = next(
        row for row in records
        if row["messages"][-1]["content"].startswith(category + " ")
    )
    messages = item["messages"]
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    complete = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    complete_ids = tokenizer.encode(complete, add_special_tokens=False)
    aligned = complete_ids[: len(prompt_ids)] == prompt_ids
    if not aligned:
        raise ValueError(f"prompt/answer boundary differs for {category}")
    answer_ids = complete_ids[len(prompt_ids) :]
    print(f"\n{category}: {messages[-1]['content']}")
    print("  prompt matches complete-example prefix:", aligned)
    print("  complete example tokens:", len(complete_ids))
    print("  answer-side tokens:", tokenizer.convert_ids_to_tokens(answer_ids))
