---
title: SPLADE lens
emoji: 🔍
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: mit
---

# SPLADE lens

**An interactive tool that shows what SPLADE does to a query and a document: which terms it adds,
which input word produced each term, and which terms the relevance score is made of.**

Website: **[https://leonardosantoro33.github.io/splade-lens/](https://leonardosantoro33.github.io/splade-lens/)**

---

## Why

SPLADE is a learned sparse retrieval model. It represents a text as a sparse vector over the
30,522-term BERT vocabulary, and the score of a document for a query is the dot product of the two
vectors. Unlike BM25, the vector contains terms that are **not in the text**: the model expands it.
Expansion is why SPLADE finds documents that use different words from the query. It is also what
makes its behaviour hard to see.

This tool makes that behaviour visible, for any query and document, and measures it on real
relevance judgements:

1. **Term provenance.** For every term in the vector, the input position it came from.
2. **Score decomposition.** The score split into one contribution per matching term. Each match is
   labelled *exact* (the term is written in both texts) or *expansion* (the model added it on at
   least one side).
3. **Pairwise comparison.** For one query and two documents, the terms that decide which document
   ranks first.
4. **Corpus-level analysis** (`analyze.py`). The share of the score that comes from expansion terms
   over many (query, relevant document) pairs, and the expansion terms that create matches most
   often.

## How it works

For an input of tokens 1..L, the masked-language-model head gives a logit for every vocabulary term
*j* at every position *i*. SPLADE's weight for term *j* is

```
w_j = max_i  log(1 + relu(logit_ij))
```

Because the pooling is a **max**, every non-zero term has exactly one input position that produced
its weight: the argmax. The tool keeps that position. Term provenance is therefore exact, not an
attribution heuristic. It says where a weight was pooled from, not *why* the network produced it.

The score is `Σ_j w_q(j) · w_d(j)` over the terms present in both vectors. Each product is one row
of the decomposition, and the rows sum exactly to the score; the tests check this.

## Use

```bash
pip install -r requirements.txt

python app.py                                              # web interface on http://127.0.0.1:7860
python -m splade_lens "symptoms of low iron" "Anemia caused by a lack of iron ..."   # terminal
python -m splade_lens "query" "document" --json            # everything, as JSON
```

Corpus-level analysis on a public test collection (needs `pip install ir_datasets`):

```bash
python make_pairs.py beir/scifact/test pairs/scifact.tsv   # (query, relevant document) pairs
python analyze.py pairs/scifact.tsv                        # -> results/scifact.summary.json
```

The website in `docs/` is static: it shows examples computed in advance. To rebuild its data
after editing `examples/examples.json` or running `analyze.py`:

```bash
python export_site.py          # writes docs/data.js, then open docs/index.html
```

The website can also run the model in the visitor's browser ("Try your own text"), with
[transformers.js](https://huggingface.co/docs/transformers.js). This needs an ONNX copy of the model
on the Hugging Face hub:

```bash
pip install onnx onnxruntime
python convert_onnx.py --out splade_onnx --push <user>/splade-cocondenser-ensembledistil-onnx
python export_site.py --browser-model <user>/splade-cocondenser-ensembledistil-onnx --browser-model-mb 110
cd docs && python -m http.server 8000      # the in-browser box needs http://, not file://
```

The browser uses the 8-bit model by default; `convert_onnx.py` prints how much it differs from
PyTorch.

The model is chosen with the `SPLADE_MODEL` environment variable. The default is
`naver/splade-cocondenser-ensembledistil`, downloaded from the Hugging Face hub on first use. Texts
are truncated to 256 tokens (`SPLADE_MAX_LENGTH`).

## Findings

**[To be written from `results/*.summary.json` after running `analyze.py`. Report only numbers
from those files, with the dataset and the number of pairs.]**

| dataset | pairs | score share from expansion terms (mean / median) | pairs where expansion > 50% of the score | query weight on expansion terms (mean) |
|---|---|---|---|---|
| | | | | |

## Files

| file | purpose |
|---|---|
| `splade_lens/lens.py` | the core: SPLADE encoding with term provenance, score decomposition, comparison of two documents |
| `splade_lens/report.py` | text and JSON views of an explanation |
| `splade_lens/__main__.py` | command-line interface |
| `app.py` | interactive web interface (Gradio) to run on your own texts; also usable as a Hugging Face Space |
| `docs/index.html` | the static website (GitHub Pages): the examples, the score flow, the comparison and the findings |
| `docs/splade-core.js` | the same SPLADE computation in JavaScript, for the in-browser box |
| `convert_onnx.py` | exports the model to ONNX (full precision and 8-bit), checks both against PyTorch, uploads to the hub |
| `export_site.py` | runs the model on `examples/examples.json`, collects `results/*.summary.json`, writes `docs/data.js` for the website |
| `examples/examples.json` | the query–document pairs and comparisons shown on the website |
| `make_pairs.py` | writes (query, relevant document) pairs from an `ir_datasets` collection |
| `analyze.py` | runs the lens over many pairs and writes per-pair results and a summary |
| `tests/` | tests on a tiny randomly initialised BERT built offline. They check the weights and source positions against an independent brute-force computation, and that the score decomposition is exact |

## Related work

[SPLADE Visualizer](https://www.nathanleroy.dev/spladeviz/) shows, in the browser, how a SPLADE
vector is built for a single query. SPLADE lens focuses on the query–document score: it decomposes
it term by term, separates matches on words written in both texts from matches created by
expansion, and measures this on test collections with relevance judgements.

## Limitations

The in-browser model is an 8-bit quantised ONNX export. On 3 test texts its top-20 terms matched PyTorch on 14/20, 20/20 and 17/20 positions, with a largest SPLADE weight difference of 0.7048. The full-precision ONNX export matched on 20/20 for all three, with a largest difference of 1.218e-05. The examples on the site are computed in PyTorch.

- Terms are WordPiece entries of the BERT vocabulary (`##` marks a word piece), so one word can be
  several terms.
- The default model is trained on English (MS MARCO).
- Provenance is exact for the pooling step, not a causal explanation of the network.
- Texts longer than the maximum length are truncated; the interface and the analysis report it.

## Credits and licences

Code: MIT (see `LICENSE`). The default model `naver/splade-cocondenser-ensembledistil` is
distributed by its authors under **CC BY-NC-SA 4.0** (non-commercial). It is described in:

T. Formal, C. Lassance, B. Piwowarski, S. Clinchant. *From Distillation to Hard Negative Sampling:
Making Sparse Neural IR Models More Effective.* SIGIR 2022 (arXiv:2205.04733).
