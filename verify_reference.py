"""
verify_reference.py — Check that SPLADE lens gives the same numbers as a plain, independent
implementation of SPLADE (the formula from the model card), on every text of examples/examples.json.

The reference below does not use the splade_lens package: it tokenises a padded batch, runs the
masked-language model and computes  w_j = max_i log(1 + relu(logit_ij))  directly. For every text
and every query/document pair it compares:
  * the largest absolute difference over all vocabulary weights;
  * whether the same terms are non-zero;
  * how many of the top-20 terms are in common;
  * the score of each pair (dot product) against the score reported by splade_lens.

    HF_HUB_OFFLINE=1 python verify_reference.py
    # writes results/verification.json; report only the numbers that this script prints
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from splade_lens import DEFAULT_MODEL, SpladeLens

ROOT = Path(__file__).resolve().parent


def reference_vectors(lens, texts):
    """Independent SPLADE vectors: padded batch, log1p(relu(logits)) * mask, max over positions."""
    enc = lens.tokenizer(texts, padding=True, truncation=True, max_length=lens.max_length, return_tensors="pt")
    dev = next(lens.model.parameters()).device
    enc = {k: v.to(dev) for k, v in enc.items()}
    with torch.no_grad():
        logits = lens.model(**enc).logits
    act = torch.log1p(torch.relu(logits)) * enc["attention_mask"].unsqueeze(-1)
    return act.max(dim=1).values.double().cpu().numpy()


def lens_vector(rep, size):
    v = np.zeros(size, dtype=np.float64)
    for t in rep.terms:
        v[t.token_id] = t.weight
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples", default=str(ROOT / "examples" / "examples.json"))
    ap.add_argument("--model", default=os.environ.get("SPLADE_MODEL", DEFAULT_MODEL))
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--out", default=str(ROOT / "results" / "verification.json"))
    args = ap.parse_args()

    spec = json.loads(Path(args.examples).read_text(encoding="utf-8"))
    pairs = [(e["id"], e["query"], e["document"]) for e in spec.get("examples", [])]
    for c in spec.get("comparisons", []):
        pairs += [(c["id"] + "/A", c["query"], c["doc_a"]), (c["id"] + "/B", c["query"], c["doc_b"])]
    texts = sorted({t for _, q, d in pairs for t in (q, d)})

    lens = SpladeLens.from_pretrained(args.model, max_length=args.max_length)
    ref = dict(zip(texts, reference_vectors(lens, texts)))   # one padded batch, as in the model card
    size = next(iter(ref.values())).shape[0]

    per_text = []
    for t in texts:
        mine, theirs = lens_vector(lens.encode(t), size), ref[t]
        top_m, top_r = set(np.argsort(-mine)[:20]), set(np.argsort(-theirs)[:20])
        per_text.append({"text": t, "max_abs_weight_diff": float(np.abs(mine - theirs).max()),
                         "same_nonzero_terms": bool(np.array_equal(mine > 0, theirs > 0)),
                         "nonzero_terms": int((theirs > 0).sum()), "top20_in_common": len(top_m & top_r)})

    per_pair = []
    for pid, q, d in pairs:
        ours = lens.explain(q, d)
        ref_score = float(ref[q] @ ref[d])
        per_pair.append({"pair": pid, "score_lens": ours.score, "score_reference": ref_score,
                         "abs_diff": abs(ours.score - ref_score),
                         "sum_of_contributions": float(sum(m.contribution for m in ours.matches))})

    summary = {
        "model": args.model, "max_length": args.max_length, "texts": len(texts), "pairs": len(pairs),
        "max_abs_weight_diff": max(x["max_abs_weight_diff"] for x in per_text),
        "all_same_nonzero_terms": all(x["same_nonzero_terms"] for x in per_text),
        "min_top20_in_common": min(x["top20_in_common"] for x in per_text),
        "max_abs_score_diff": max(x["abs_diff"] for x in per_pair),
        "max_abs_decomposition_gap": max(abs(x["score_lens"] - x["sum_of_contributions"]) for x in per_pair),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "texts": per_text, "pairs": per_pair}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"details in {out}")


if __name__ == "__main__":
    main()
