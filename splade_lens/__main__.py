"""Command line:

    python -m splade_lens "query" "document"            # readable table
    python -m splade_lens "query" "document" --json     # everything, as JSON

Model: --model, or the SPLADE_MODEL environment variable (default: the SPLADE CoCondenser
EnsembleDistil checkpoint on the Hugging Face hub, downloaded on first use).
"""
import argparse
import json
import os

from .lens import DEFAULT_MODEL, SpladeLens
from .report import explanation_dict, explanation_text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query")
    ap.add_argument("document")
    ap.add_argument("--model", default=os.environ.get("SPLADE_MODEL", DEFAULT_MODEL))
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--top", type=int, default=15, help="rows to show (0 = all)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    lens = SpladeLens.from_pretrained(args.model, max_length=args.max_length)
    e = lens.explain(args.query, args.document)
    print(json.dumps(explanation_dict(e, args.top), indent=2) if args.json else explanation_text(e, args.top or 10**9))


if __name__ == "__main__":
    main()
