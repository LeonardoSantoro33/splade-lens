"""
make_pairs.py — Write (query, relevant document) pairs from a public test collection, for analyze.py.

Uses the ir_datasets package (pip install ir_datasets), which downloads the collection on first
use. Only judged-relevant pairs are written (relevance >= --min-rel).

    python make_pairs.py beir/scifact/test pairs/scifact.tsv
    python make_pairs.py beir/nfcorpus/test pairs/nfcorpus.tsv --min-rel 2
    python make_pairs.py msmarco-passage/trec-dl-2019/judged pairs/dl19.tsv --min-rel 2   # large download (~1 GB)

Documents are written as title + " " + text when the collection has titles.
"""
import argparse
from pathlib import Path


def clean(s):
    return " ".join(str(s).split())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="an ir_datasets id, e.g. beir/scifact/test")
    ap.add_argument("out")
    ap.add_argument("--min-rel", type=int, default=1)
    args = ap.parse_args()
    import ir_datasets
    ds = ir_datasets.load(args.dataset)
    queries = {q.query_id: clean(q.text) for q in ds.queries_iter()}
    store = ds.docs_store()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for qr in ds.qrels_iter():
            if qr.relevance < args.min_rel or qr.query_id not in queries:
                continue
            doc = store.get(qr.doc_id)
            title = clean(getattr(doc, "title", "") or "")
            text = clean(getattr(doc, "text", "") or "")
            f.write(f"{qr.query_id}\t{queries[qr.query_id]}\t{qr.doc_id}\t{(title + ' ' + text).strip()}\n")
            n += 1
    print(f"{n} pairs written to {out}")


if __name__ == "__main__":
    main()
