from __future__ import annotations

import argparse
import sys

import run_ppl_eval as base


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--token-offset", type=int, default=0)
    known, remaining = parser.parse_known_args()
    token_offset = max(0, int(known.token_offset))

    original_loader = base.load_wikitext_tokens

    def load_wikitext_tokens_with_offset(tokenizer, max_tokens: int):
        ids = original_loader(tokenizer, max_tokens + token_offset)
        start = token_offset
        end = token_offset + max_tokens + 1
        if ids.shape[-1] < end:
            raise RuntimeError(
                f"WikiText tokenized length {ids.shape[-1]} < requested offset window {end}"
            )
        return ids[:, start:end]

    base.load_wikitext_tokens = load_wikitext_tokens_with_offset
    sys.argv = [sys.argv[0], *remaining]
    rc = base.main()
    return int(rc) if rc is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
