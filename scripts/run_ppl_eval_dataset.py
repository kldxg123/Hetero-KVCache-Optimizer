from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from datasets import load_dataset

import run_ppl_eval as base


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--dataset-text-field", default="text")
    parser.add_argument("--token-offset", type=int, default=0)
    known, remaining = parser.parse_known_args()
    token_offset = max(0, int(known.token_offset))
    output_path = "experiments/ppl_eval.json"
    max_tokens = None
    for idx, item in enumerate(remaining):
        if item == "--output" and idx + 1 < len(remaining):
            output_path = remaining[idx + 1]
        elif item == "--max-tokens" and idx + 1 < len(remaining):
            max_tokens = int(remaining[idx + 1])

    def load_dataset_tokens(tokenizer, max_tokens: int):
        dataset = load_dataset(
            known.dataset_name,
            known.dataset_config,
            split=known.dataset_split,
            download_mode="reuse_dataset_if_exists",
        )
        parts = []
        for row in dataset:
            value = row.get(known.dataset_text_field, "")
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
        text = "\n\n".join(parts)
        ids = tokenizer(text, return_tensors="pt").input_ids
        start = token_offset
        end = token_offset + max_tokens + 1
        if ids.shape[-1] < end:
            raise RuntimeError(
                f"Dataset tokenized length {ids.shape[-1]} < requested offset window {end}"
            )
        return ids[:, start:end]

    base.load_wikitext_tokens = load_dataset_tokens
    sys.argv = [sys.argv[0], *remaining]
    rc = base.main()
    if rc == 0:
        out = Path(output_path)
        if out.exists():
            payload = json.loads(out.read_text(encoding="utf-8"))
            payload["dataset"] = f"{known.dataset_name}/{known.dataset_config}/{known.dataset_split}"
            payload["dataset_name"] = known.dataset_name
            payload["dataset_config"] = known.dataset_config
            payload["dataset_split"] = known.dataset_split
            payload["dataset_text_field"] = known.dataset_text_field
            payload["token_offset"] = token_offset
            if max_tokens is not None:
                payload["token_range"] = [token_offset, token_offset + max_tokens + 1]
            payload["dataset_runner"] = "scripts/run_ppl_eval_dataset.py"
            out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return int(rc) if rc is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
