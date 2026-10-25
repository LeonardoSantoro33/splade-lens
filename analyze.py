"""
analyze.py — Run the lens over many (query, relevant document) pairs and summarise.

Questions it answers, on real data:
  - What share of a relevant document's score comes from terms the model ADDED (expansion),
    rather than from words written in both texts?
  - How much of a query's / document's weight sits on expansion terms?
  - Which expansion terms create matches most often?

Input: a TSV file with 4 columns and no header: query_id, query text, doc_id, document text
(make one with make_pairs.py). Output: results/<name>.pairs.jsonl (one line per pair) and
results/<name>.summary.json, and a short summary printed on screen.

    python analyze.py pairs/scifact.tsv
    python analyze.py pairs/scifact.tsv --limit 200        # quick try
"""
import argparse
import collections
import json
import os
import statistics
import time
from pathlib import Path

from splade_lens import DEFAULT_MODEL, SpladeLens, explain_pair


def read_pairs(path, limit):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                raise SystemExit(f"[ERROR] {path}: expected 4 tab-separated columns, got {len(parts)}")
            out.append(parts)
            if limit and len(out) >= limit:
                break
    return out


def describe(xs):
    xs = sorted(xs)
    return {"mean": statistics.fmean(xs), "median": statistics.median(xs), "min": xs[0], "max": xs[-1], "n": len(xs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs")
    ap.add_argument("--model", default=os.environ.get("SPLADE_MODEL", DEFAULT_MODEL))
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="only the first N pairs (0 = all)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    pairs = read_pairs(args.pairs, args.limit)
    name = Path(args.pairs).stem
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lens = SpladeLens.from_pretrained(args.model, max_length=args.max_length)
    print(f"model {args.model}, {len(pairs)} pairs from {args.pairs}")

    q_cache, rows, exp_terms = {}, [], collections.Counter()
    t0 = time.time()
    with open(out / f"{name}.pairs.jsonl", "w", encoding="utf-8") as f:
        for i, (qid, qtext, did, dtext) in enumerate(pairs, 1):
            if qid not in q_cache:
                q_cache[qid] = lens.encode(qtext)
            e = explain_pair(q_cache[qid], lens.encode(dtext))
            row = {"query_id": qid, "doc_id": did, "score": e.score, "matches": len(e.matches),
                   "expansion_score_share": e.expansion_score_share,
                   "query_nnz": e.query.nnz, "doc_nnz": e.document.nnz,
                   "query_expansion_weight_share": e.query.expansion_weight_share,
                   "doc_expansion_weight_share": e.document.expansion_weight_share,
                   "doc_truncated": e.document.truncated,
                   "top_matches": [[m.term, round(m.share, 4), m.kind] for m in e.matches[:5]]}
            exp_terms.update(m.term for m in e.matches if m.kind == "expansion")
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            if i % 100 == 0:
                print(f"  {i}/{len(pairs)} pairs ({time.time() - t0:.0f} s)")

    keys = ["score", "matches", "expansion_score_share", "query_nnz", "doc_nnz",
            "query_expansion_weight_share", "doc_expansion_weight_share"]
    summary = {"pairs_file": args.pairs, "model": args.model, "max_length": args.max_length,
               "pairs": len(rows), "queries": len(q_cache),
               "docs_truncated": sum(r["doc_truncated"] for r in rows),
               "pairs_with_majority_of_score_from_expansion": sum(r["expansion_score_share"] > 0.5 for r in rows),
               "stats": {k: describe([r[k] for r in rows]) for k in keys},
               "most_frequent_expansion_matches": exp_terms.most_common(30),
               "seconds": time.time() - t0}
    (out / f"{name}.summary.json").write_text(json.dumps(summary, indent=2))

    s = summary["stats"]
    print(f"\n{len(rows)} pairs, {len(q_cache)} queries")
    print(f"share of the score from expansion terms: mean {s['expansion_score_share']['mean']:.3f}, "
          f"median {s['expansion_score_share']['median']:.3f}")
    print(f"pairs where expansion gives more than half of the score: "
          f"{summary['pairs_with_majority_of_score_from_expansion']} / {len(rows)}")
    print(f"query weight on expansion terms: mean {s['query_expansion_weight_share']['mean']:.3f}; "
          f"document: mean {s['doc_expansion_weight_share']['mean']:.3f}")
    print(f"non-zeros: query mean {s['query_nnz']['mean']:.1f}, document mean {s['doc_nnz']['mean']:.1f}")
    print("most frequent expansion matches:", ", ".join(t for t, _ in exp_terms.most_common(15)))
    print(f"\nwritten {out / (name + '.summary.json')} and {out / (name + '.pairs.jsonl')}")


if __name__ == "__main__":
    main()
