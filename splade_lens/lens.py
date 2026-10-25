"""
lens.py — Look inside a SPLADE representation.

SPLADE turns a text into a sparse vector over the BERT vocabulary:

    w_j = max_i  log(1 + relu(logit_ij)) * mask_i

where logit_ij is the masked-language-model logit of vocabulary term j at input position i.
Because of the max over positions, every non-zero term j has exactly ONE input position that
produced its weight: argmax_i. This module keeps that position, so for every term we can say
"this weight comes from this word of the input". Nothing is approximated: the source position is
the exact argmax used by the model's pooling.

The relevance score of a document for a query is the dot product of the two vectors. explain()
splits it into one contribution per matching term, and labels each match as:
  exact      the term is a token of both input texts
  expansion  the term is missing from the query text, the document text, or both, and was
             added by the model
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

DEFAULT_MODEL = "naver/splade-cocondenser-ensembledistil"


@dataclass
class Term:
    """One non-zero dimension of a SPLADE vector."""
    term: str            # vocabulary entry (WordPiece; '##' marks a continuation piece)
    token_id: int
    weight: float
    source_pos: int      # input position whose activation is the max (the pooling argmax)
    source_token: str    # the input token at that position ([CLS]/[SEP] included)
    in_input: bool       # True if the term is one of the input's own tokens


@dataclass
class Representation:
    text: str
    tokens: List[str]                    # input tokens, special tokens included
    terms: List[Term]                    # sorted by weight, descending
    truncated: bool = False

    @property
    def nnz(self) -> int:
        return len(self.terms)

    @property
    def total_weight(self) -> float:
        return float(sum(t.weight for t in self.terms))

    @property
    def expansion_weight_share(self) -> float:
        """Fraction of the total weight carried by terms that are not in the input."""
        tot = self.total_weight
        return float(sum(t.weight for t in self.terms if not t.in_input) / tot) if tot > 0 else 0.0

    def by_id(self) -> dict:
        return {t.token_id: t for t in self.terms}

    def weight_per_token(self) -> List[float]:
        """Total weight of the terms each input position produced (the 'owner' of each term)."""
        out = [0.0] * len(self.tokens)
        for t in self.terms:
            out[t.source_pos] += t.weight
        return out

    def terms_from_position(self, pos: int) -> List[Term]:
        return [t for t in self.terms if t.source_pos == pos]


@dataclass
class Match:
    term: str
    token_id: int
    query_weight: float
    doc_weight: float
    contribution: float          # query_weight * doc_weight
    share: float                 # contribution / score
    kind: str                    # "exact" or "expansion"
    query_source: str
    doc_source: str


@dataclass
class Explanation:
    query: Representation
    document: Representation
    score: float
    matches: List[Match] = field(default_factory=list)   # sorted by contribution, descending

    @property
    def expansion_score_share(self) -> float:
        return float(sum(m.share for m in self.matches if m.kind == "expansion"))


@dataclass
class TermDiff:
    term: str
    contribution_a: float
    contribution_b: float

    @property
    def difference(self) -> float:
        return self.contribution_a - self.contribution_b


class SpladeLens:
    """Wraps a SPLADE masked-language model and its tokenizer."""

    def __init__(self, model, tokenizer, max_length: int = 256, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.special_ids = set(tokenizer.all_special_ids)

    @classmethod
    def from_pretrained(cls, name_or_path: str = DEFAULT_MODEL, max_length: int = 256, device: Optional[str] = None):
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(name_or_path)
        model = AutoModelForMaskedLM.from_pretrained(name_or_path)
        return cls(model, tok, max_length=max_length, device=device)

    @torch.no_grad()
    def activations(self, text: str):
        """Per-position activations log(1+relu(logits)) * mask, shape [L, V], plus input ids."""
        if not text or not text.strip():
            raise ValueError("empty text")
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt")
        full_len = len(self.tokenizer(text, truncation=False)["input_ids"])
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits[0]                         # [L, V]
        mask = enc["attention_mask"][0].unsqueeze(-1).to(logits.dtype)
        act = torch.log1p(torch.relu(logits)) * mask
        return act, enc["input_ids"][0], full_len > self.max_length

    def encode(self, text: str) -> Representation:
        act, ids, truncated = self.activations(text)
        weights, pos = act.max(dim=0)                                 # [V], [V]
        nz = torch.nonzero(weights > 0).flatten()
        ids_list = ids.tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(ids_list)
        own = {i for i in ids_list if i not in self.special_ids}
        terms = []
        for j in nz.tolist():
            p = int(pos[j])
            terms.append(Term(term=self.tokenizer.convert_ids_to_tokens(j), token_id=j, weight=float(weights[j]),
                              source_pos=p, source_token=tokens[p], in_input=j in own))
        terms.sort(key=lambda t: (-t.weight, t.token_id))
        return Representation(text=text, tokens=tokens, terms=terms, truncated=truncated)

    def explain(self, query: str, document: str) -> Explanation:
        return explain_pair(self.encode(query), self.encode(document))

    def compare(self, query: str, doc_a: str, doc_b: str):
        q = self.encode(query)
        ea, eb = explain_pair(q, self.encode(doc_a)), explain_pair(q, self.encode(doc_b))
        return ea, eb, diff_explanations(ea, eb)


def explain_pair(q: Representation, d: Representation) -> Explanation:
    dq, dd = q.by_id(), d.by_id()
    common = dq.keys() & dd.keys()
    raw = [(j, dq[j].weight * dd[j].weight) for j in common]
    score = float(sum(c for _, c in raw))
    matches = []
    for j, c in raw:
        tq, td = dq[j], dd[j]
        matches.append(Match(term=tq.term, token_id=j, query_weight=tq.weight, doc_weight=td.weight,
                             contribution=c, share=c / score if score > 0 else 0.0,
                             kind="exact" if (tq.in_input and td.in_input) else "expansion",
                             query_source=tq.source_token, doc_source=td.source_token))
    matches.sort(key=lambda m: (-m.contribution, m.token_id))
    return Explanation(query=q, document=d, score=score, matches=matches)


def diff_explanations(a: Explanation, b: Explanation) -> List[TermDiff]:
    """Per-term contributions of two documents for the same query, sorted by |A - B|."""
    ca = {m.token_id: m for m in a.matches}
    cb = {m.token_id: m for m in b.matches}
    out = []
    for j in ca.keys() | cb.keys():
        term = (ca.get(j) or cb.get(j)).term
        out.append(TermDiff(term=term, contribution_a=ca[j].contribution if j in ca else 0.0,
                            contribution_b=cb[j].contribution if j in cb else 0.0))
    out.sort(key=lambda t: (-abs(t.difference), t.term))
    return out
