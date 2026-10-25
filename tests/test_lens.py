import math

import pytest
import torch

from splade_lens import diff_explanations, explain_pair
from splade_lens.report import explanation_dict, explanation_text


def brute_force(lens, text):
    """Independent recomputation of the SPLADE vector: loop over positions, no vectorised max."""
    enc = lens.tokenizer(text, truncation=True, max_length=lens.max_length, return_tensors="pt")
    with torch.no_grad():
        logits = lens.model(**enc).logits[0]
    L, V = logits.shape
    w = [0.0] * V
    src = [None] * V
    for i in range(L):
        if enc["attention_mask"][0, i] == 0:
            continue
        for j in range(V):
            a = math.log1p(max(0.0, float(logits[i, j])))
            if a > w[j]:
                w[j], src[j] = a, i
    return w, src, enc["input_ids"][0].tolist()


@pytest.mark.parametrize("text", ["symptoms of low iron", "cheap flights to rome", "how do vaccines work ."])
def test_weights_and_sources_match_brute_force(lens, text):
    r = lens.encode(text)
    w, src, ids = brute_force(lens, text)
    got = {t.token_id: t for t in r.terms}
    assert set(got) == {j for j, x in enumerate(w) if x > 0}
    for j, t in got.items():
        assert t.weight == pytest.approx(w[j], rel=1e-5, abs=1e-6)
        assert t.source_pos == src[j]
        assert t.source_token == r.tokens[src[j]]
    special = set(lens.tokenizer.all_special_ids)
    own = {i for i in ids if i not in special}
    assert all(t.in_input == (t.token_id in own) for t in r.terms)


def test_terms_sorted_and_weight_per_token_sums(lens):
    r = lens.encode("symptoms of low iron")
    ws = [t.weight for t in r.terms]
    assert ws == sorted(ws, reverse=True)
    assert sum(r.weight_per_token()) == pytest.approx(r.total_weight)
    assert 0.0 <= r.expansion_weight_share <= 1.0
    assert sum(len(r.terms_from_position(i)) for i in range(len(r.tokens))) == r.nnz


def test_score_is_dot_product_and_decomposition_is_exact(lens):
    q, d = "symptoms of low iron", "anemia is low iron in the blood , pale skin ."
    e = lens.explain(q, d)
    wq, _, _ = brute_force(lens, q)
    wd, _, _ = brute_force(lens, d)
    assert e.score == pytest.approx(sum(a * b for a, b in zip(wq, wd)), rel=1e-5)
    assert sum(m.contribution for m in e.matches) == pytest.approx(e.score, rel=1e-6)
    assert sum(m.share for m in e.matches) == pytest.approx(1.0)
    assert {m.token_id for m in e.matches} == {j for j in range(len(wq)) if wq[j] > 0 and wd[j] > 0}
    for m in e.matches:
        tq, td = e.query.by_id()[m.token_id], e.document.by_id()[m.token_id]
        assert m.kind == ("exact" if tq.in_input and td.in_input else "expansion")
    assert 0.0 <= e.expansion_score_share <= 1.0 + 1e-9


def test_compare_differences(lens):
    ea, eb, diff = lens.compare("symptoms of low iron", "anemia is low iron .", "iron is a metal .")
    assert sum(t.contribution_a for t in diff) == pytest.approx(ea.score, rel=1e-6)
    assert sum(t.contribution_b for t in diff) == pytest.approx(eb.score, rel=1e-6)
    assert abs(diff[0].difference) == pytest.approx(max(abs(t.difference) for t in diff))


def test_truncation_flag(lens):
    long_text = " ".join(["iron"] * 40)
    r = lens.encode(long_text)
    assert r.truncated and len(r.tokens) == lens.max_length
    assert not lens.encode("iron").truncated


def test_empty_text_rejected(lens):
    with pytest.raises(ValueError):
        lens.encode("   ")


def test_reports(lens):
    e = lens.explain("cheap flights to rome", "budget flights to the italy capital .")
    d = explanation_dict(e, top=5)
    assert d["score"] == pytest.approx(e.score) and len(d["matches"]) <= 5
    assert "score =" in explanation_text(e)


def test_explain_pair_no_overlap():
    from splade_lens.lens import Representation, Term
    q = Representation("q", ["[CLS]", "a", "[SEP]"], [Term("a", 10, 1.0, 1, "a", True)])
    d = Representation("d", ["[CLS]", "b", "[SEP]"], [Term("b", 11, 2.0, 1, "b", True)])
    e = explain_pair(q, d)
    assert e.score == 0.0 and e.matches == [] and e.expansion_score_share == 0.0
    assert diff_explanations(e, e) == []
