"""
app.py — Web interface (Gradio). Runs locally or as a Hugging Face Space.

    python app.py                     # then open http://127.0.0.1:7860

Model: SPLADE_MODEL environment variable (default: SPLADE CoCondenser EnsembleDistil).
"""
import os

import gradio as gr
import pandas as pd

from splade_lens import DEFAULT_MODEL, SpladeLens
from splade_lens.lens import Explanation, Representation

MODEL = os.environ.get("SPLADE_MODEL", DEFAULT_MODEL)
MAX_LENGTH = int(os.environ.get("SPLADE_MAX_LENGTH", "256"))
TOP = 40
lens = SpladeLens.from_pretrained(MODEL, max_length=MAX_LENGTH)

INTRO = f"""
# SPLADE lens

SPLADE represents a text as a **sparse vector of weighted vocabulary terms**: the words of the
text, plus related terms SPLADE **adds** (expansion). The score of a document for a query is the
**dot product** of their two vectors. This page shows, for any query and document you type:

- **which terms** each vector contains, and **which input word produced each term** (exact, not an
  approximation: SPLADE keeps, for every term, the maximum over the input positions, and the page
  shows that position);
- **which terms the score is made of**, and how much of it comes from words written in both texts
  (*exact*) versus terms added by SPLADE (*expansion*).

Model: `{MODEL}` (max {MAX_LENGTH} tokens per text).
"""

ABOUT = """
### How it works

For an input of tokens 1..L, SPLADE's masked-language-model head gives a logit for every
vocabulary term *j* at every position *i*. SPLADE's weight for term *j* is

  w_j = max over i of log(1 + max(0, logit_ij))

The position that reaches the maximum is the **source** of the term: the word the term "comes
from". The highlighted texts colour each input token by the total weight of the terms it is the
source of.

`[CLS]` and `[SEP]` are special tokens that the tokenizer adds at the start and end of every text. They
are not words: through attention they read the whole text, and SPLADE uses them for terms about
the text as a whole, so many expansion terms come from them rather than from a single word.

Score = Σ over the terms in both vectors of w_query × w_document. Each row of the *matching terms*
table is one of these products; *share* is its fraction of the score.

- **exact**: the term is one of the tokens of both texts.
- **expansion**: the term is missing from the query text, the document text, or both; it matches
  only because SPLADE added it.

### Limits

- Terms are WordPiece entries of the BERT vocabulary (`##` marks a word piece).
- The default model is trained on English (MS MARCO); other languages will mostly fail, which is
  itself visible here.
- Texts longer than the maximum length are truncated (shown in the summary).
- Source positions explain where a weight was *pooled from*, not *why* the network produced it.
"""

EXAMPLES = [
    ["symptoms of low iron",
     "Anemia caused by a lack of iron often leads to tiredness, pale skin and shortness of breath. "
     "Doctors usually confirm it with a blood test that measures ferritin."],
    ["cheap flights to rome",
     "Several budget airlines offer low-cost tickets to the Italian capital, especially outside the summer season."],
    ["how do vaccines work",
     "A vaccine exposes the immune system to a harmless piece of a pathogen, so that it learns to "
     "recognise it and produces antibodies before a real infection."],
]
COMPARE_EXAMPLES = [
    ["symptoms of low iron",
     "Anemia caused by a lack of iron often leads to tiredness, pale skin and shortness of breath.",
     "Iron is a metal used to build bridges and ships because it is strong and cheap to produce."],
]


def token_highlights(r: Representation):
    w = r.weight_per_token()
    top = max(w) if w and max(w) > 0 else 1.0
    out = []
    for i, (tok, x) in enumerate(zip(r.tokens, w)):
        out.append((tok + " ", round(x / top, 3) if x > 0 else None))
    return out


def terms_table(r: Representation) -> pd.DataFrame:
    return pd.DataFrame([{"term": t.term, "weight": round(t.weight, 3), "from word": t.source_token,
                          "position": t.source_pos, "in the text": "yes" if t.in_input else "no (expansion)"}
                         for t in r.terms[:TOP]])


def matches_table(e: Explanation) -> pd.DataFrame:
    return pd.DataFrame([{"term": m.term, "query weight": round(m.query_weight, 3), "doc weight": round(m.doc_weight, 3),
                          "contribution": round(m.contribution, 3), "share %": round(100 * m.share, 1), "kind": m.kind,
                          "query word": m.query_source, "doc word": m.doc_source} for m in e.matches[:TOP]])


def summary(e: Explanation) -> str:
    q, d = e.query, e.document
    trunc = " The document was **truncated** to the maximum length." if d.truncated else ""
    trunc += " The query was **truncated**." if q.truncated else ""
    return (f"**Score: {e.score:.2f}** from {len(e.matches)} matching terms; "
            f"**{100 * e.expansion_score_share:.1f}%** of it comes from expansion terms.\n\n"
            f"Query: {q.nnz} terms ({100 * q.expansion_weight_share:.1f}% of its weight on expansion terms). "
            f"Document: {d.nnz} terms ({100 * d.expansion_weight_share:.1f}% on expansion terms).{trunc}")


def run_explain(query, document):
    if not query.strip() or not document.strip():
        raise gr.Error("Write both a query and a document.")
    e = lens.explain(query, document)
    return (summary(e), token_highlights(e.query), token_highlights(e.document), matches_table(e),
            terms_table(e.query), terms_table(e.document))


def run_compare(query, doc_a, doc_b):
    if not all(s.strip() for s in (query, doc_a, doc_b)):
        raise gr.Error("Write a query and two documents.")
    ea, eb, diff = lens.compare(query, doc_a, doc_b)
    winner = "A" if ea.score > eb.score else ("B" if eb.score > ea.score else "neither (tie)")
    text = (f"**Document A: {ea.score:.3f}**  ·  **Document B: {eb.score:.3f}**  "
            f"(difference {ea.score - eb.score:+.3f})  →  ranked first: **{winner}**\n\n"
            f"Expansion share of the score: A {100 * ea.expansion_score_share:.1f}%, "
            f"B {100 * eb.expansion_score_share:.1f}%. The table lists the terms that make the difference.")
    table = pd.DataFrame([{"term": t.term, "contribution A": round(t.contribution_a, 3),
                           "contribution B": round(t.contribution_b, 3), "A − B": round(t.difference, 3)}
                          for t in diff[:TOP]])
    return text, table


with gr.Blocks(title="SPLADE lens") as demo:
    gr.Markdown(INTRO)
    with gr.Tab("Query and document"):
        with gr.Row():
            q_in = gr.Textbox(label="Query", lines=2)
            d_in = gr.Textbox(label="Document", lines=6)
        go = gr.Button("Explain", variant="primary")
        out_summary = gr.Markdown()
        gr.Markdown("**Where the weight comes from** (each input token coloured by the weight of the terms it produced)")
        q_hl = gr.HighlightedText(label="Query tokens", combine_adjacent=False, show_legend=False)
        d_hl = gr.HighlightedText(label="Document tokens", combine_adjacent=False, show_legend=False)
        m_tab = gr.Dataframe(label="Matching terms (the score, term by term)", wrap=True)
        with gr.Accordion("All terms of the query and of the document", open=False):
            q_tab = gr.Dataframe(label=f"Query terms (top {TOP})", wrap=True)
            d_tab = gr.Dataframe(label=f"Document terms (top {TOP})", wrap=True)
        gr.Examples(EXAMPLES, inputs=[q_in, d_in])
        go.click(run_explain, [q_in, d_in], [out_summary, q_hl, d_hl, m_tab, q_tab, d_tab])
    with gr.Tab("Compare two documents"):
        cq = gr.Textbox(label="Query", lines=2)
        with gr.Row():
            ca = gr.Textbox(label="Document A", lines=5)
            cb = gr.Textbox(label="Document B", lines=5)
        go2 = gr.Button("Compare", variant="primary")
        c_sum = gr.Markdown()
        c_tab = gr.Dataframe(label="Terms that make the difference", wrap=True)
        gr.Examples(COMPARE_EXAMPLES, inputs=[cq, ca, cb])
        go2.click(run_compare, [cq, ca, cb], [c_sum, c_tab])
    with gr.Tab("How it works"):
        gr.Markdown(ABOUT)

if __name__ == "__main__":
    demo.launch()
