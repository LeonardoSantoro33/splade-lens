"""
export_site.py — Run the model on the examples and write the data of the website (docs/data.js).

The website (docs/index.html) is static: it does not run the model. This script runs it on your
machine, on the examples in examples/examples.json, and saves everything the page shows. It also
includes the summaries written by analyze.py (results/*.summary.json), if there are any.

    python export_site.py
    python export_site.py --examples examples/examples.json --results results
    python export_site.py --browser-model <user>/splade-cocondenser-ensembledistil-onnx --browser-model-mb 110
    # both 8-bit and full precision (upload both with convert_onnx.py --push-fp32):
    python export_site.py --browser-model <user>/splade-cocondenser-ensembledistil-onnx \
        --browser-model-mb 110 --browser-fp32-mb 440

Then open docs/index.html in a browser, or publish the docs/ folder with GitHub Pages.
"""
import argparse
import datetime
import json
import os
from pathlib import Path

from splade_lens import DEFAULT_MODEL, SpladeLens, explain_pair
from splade_lens.lens import Explanation, Representation

ROOT = Path(__file__).resolve().parent
TOP_TERMS = 60


def rep_data(r: Representation, top: int = TOP_TERMS) -> dict:
    return {
        "text": r.text, "tokens": r.tokens, "truncated": r.truncated,
        "weight_per_token": [round(x, 4) for x in r.weight_per_token()],
        "nnz": r.nnz, "total_weight": round(r.total_weight, 4),
        "expansion_weight_share": round(r.expansion_weight_share, 4),
        "terms": [{"term": t.term, "w": round(t.weight, 4), "pos": t.source_pos, "in_input": t.in_input}
                  for t in (r.terms[:top] if top else r.terms)],
    }


def expl_data(e: Explanation) -> dict:
    qd, dd = e.query.by_id(), e.document.by_id()
    return {
        "score": round(e.score, 4), "expansion_score_share": round(e.expansion_score_share, 4),
        "query": rep_data(e.query), "document": rep_data(e.document, top=0),   # all document terms, for the hover
        "matches": [{"term": m.term, "qw": round(m.query_weight, 4), "dw": round(m.doc_weight, 4),
                     "c": round(m.contribution, 4), "share": round(m.share, 4), "kind": m.kind,
                     "qpos": qd[m.token_id].source_pos, "dpos": dd[m.token_id].source_pos}
                    for m in e.matches],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples", default=str(ROOT / "examples" / "examples.json"))
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "data.js"))
    ap.add_argument("--model", default=os.environ.get("SPLADE_MODEL", DEFAULT_MODEL))
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--browser-model", default="",
                    help="Hugging Face repo with the ONNX model (convert_onnx.py) for the 'Try your own text' box; "
                         "without it the box is not shown")
    ap.add_argument("--browser-dtype", default="q8", help="q8 (8-bit, smaller download) or fp32; used when only one is offered")
    ap.add_argument("--browser-model-mb", type=int, default=0, help="download size of the 8-bit model, shown to visitors")
    ap.add_argument("--browser-fp32-mb", type=int, default=0,
                    help="download size of the full-precision model; if given, the page offers both (8-bit and full precision)")
    args = ap.parse_args()

    spec = json.loads(Path(args.examples).read_text(encoding="utf-8"))
    lens = SpladeLens.from_pretrained(args.model, max_length=args.max_length)

    examples = []
    for ex in spec.get("examples", []):
        e = lens.explain(ex["query"], ex["document"])
        examples.append({"id": ex["id"], "note": ex.get("note", ""), **expl_data(e)})
        print(f"[{ex['id']}] score {e.score:.2f}, {100 * e.expansion_score_share:.0f}% from expansion")

    comparisons = []
    for cp in spec.get("comparisons", []):
        q = lens.encode(cp["query"])
        ea, eb = explain_pair(q, lens.encode(cp["doc_a"])), explain_pair(q, lens.encode(cp["doc_b"]))
        ca = {m.term: m.contribution for m in ea.matches}
        cb = {m.term: m.contribution for m in eb.matches}
        diff = sorted(({"term": t, "a": round(ca.get(t, 0.0), 4), "b": round(cb.get(t, 0.0), 4)}
                       for t in ca.keys() | cb.keys()), key=lambda x: (-abs(x["a"] - x["b"]), x["term"]))
        comparisons.append({"id": cp["id"], "query": cp["query"], "a": expl_data(ea), "b": expl_data(eb),
                            "diff": diff[:20]})
        print(f"[{cp['id']}] A {ea.score:.2f}  B {eb.score:.2f}")

    findings = []
    res = Path(args.results)
    if res.exists():
        for p in sorted(res.glob("*.summary.json")):
            s = json.loads(p.read_text())
            st = s["stats"]
            findings.append({
                "name": p.name.replace(".summary.json", ""), "pairs": s["pairs"], "queries": s["queries"],
                "expansion_share_mean": st["expansion_score_share"]["mean"],
                "expansion_share_median": st["expansion_score_share"]["median"],
                "majority_expansion": s["pairs_with_majority_of_score_from_expansion"],
                "query_expansion_weight_mean": st["query_expansion_weight_share"]["mean"],
                "doc_expansion_weight_mean": st["doc_expansion_weight_share"]["mean"],
                "query_nnz_mean": st["query_nnz"]["mean"], "doc_nnz_mean": st["doc_nnz"]["mean"],
                "top_expansion_terms": [t for t, _ in s["most_frequent_expansion_matches"][:12]],
            })
        print(f"findings: {len(findings)} summary file(s) from {res}")

    options = []
    if args.browser_model and args.browser_fp32_mb:
        options = [{"dtype": "q8", "label": "Fast (8-bit)", "mb": args.browser_model_mb},
                   {"dtype": "fp32", "label": "Exact (full precision)", "mb": args.browser_fp32_mb}]
    data = {"model": args.model, "max_length": args.max_length,
            "browser_model": args.browser_model, "browser_dtype": args.browser_dtype,
            "browser_model_mb": args.browser_model_mb, "browser_options": options,
            "generated": datetime.date.today().isoformat(),
            "examples": examples, "comparisons": comparisons, "findings": findings}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("window.SPLADE_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"written {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
