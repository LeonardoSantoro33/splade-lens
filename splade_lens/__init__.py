"""splade-lens: look inside SPLADE representations (expansion terms, their source words, and the
per-term decomposition of the query-document score)."""
from .lens import (DEFAULT_MODEL, Explanation, Match, Representation, SpladeLens, Term, TermDiff,
                   diff_explanations, explain_pair)

__all__ = ["DEFAULT_MODEL", "Explanation", "Match", "Representation", "SpladeLens", "Term", "TermDiff",
           "diff_explanations", "explain_pair"]
