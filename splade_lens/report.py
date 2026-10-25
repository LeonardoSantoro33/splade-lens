"""report.py — Plain-text and dict views of a Representation / Explanation (CLI and analysis)."""
from __future__ import annotations

from .lens import Explanation, Representation


def representation_dict(r: Representation, top: int = 0) -> dict:
    terms = r.terms[:top] if top else r.terms
    return {
        "text": r.text, "truncated": r.truncated, "nnz": r.nnz, "total_weight": r.total_weight,
        "expansion_weight_share": r.expansion_weight_share,
        "terms": [{"term": t.term, "weight": t.weight, "from": t.source_token, "position": t.source_pos,
                   "in_input": t.in_input} for t in terms],
    }


def explanation_dict(e: Explanation, top: int = 0) -> dict:
    matches = e.matches[:top] if top else e.matches
    return {
        "score": e.score, "expansion_score_share": e.expansion_score_share,
        "matches": [{"term": m.term, "query_weight": m.query_weight, "doc_weight": m.doc_weight,
                     "contribution": m.contribution, "share": m.share, "kind": m.kind,
                     "query_source": m.query_source, "doc_source": m.doc_source} for m in matches],
        "query": representation_dict(e.query, top), "document": representation_dict(e.document, top),
    }


def explanation_text(e: Explanation, top: int = 15) -> str:
    q, d = e.query, e.document
    lines = [f"score = {e.score:.3f}   ({len(e.matches)} matching terms; "
             f"{100 * e.expansion_score_share:.1f}% of the score from expansion terms)",
             f"query:    {q.nnz} terms, {100 * q.expansion_weight_share:.1f}% of its weight on expansion terms",
             f"document: {d.nnz} terms, {100 * d.expansion_weight_share:.1f}% of its weight on expansion terms"
             + ("  [truncated]" if d.truncated else ""),
             "",
             f"{'term':<16}{'q.weight':>9}{'d.weight':>9}{'contrib':>9}{'share':>8}  {'kind':<10} from (query -> doc)"]
    for m in e.matches[:top]:
        lines.append(f"{m.term:<16}{m.query_weight:>9.2f}{m.doc_weight:>9.2f}{m.contribution:>9.2f}"
                     f"{100 * m.share:>7.1f}%  {m.kind:<10} {m.query_source} -> {m.doc_source}")
    return "\n".join(lines)
