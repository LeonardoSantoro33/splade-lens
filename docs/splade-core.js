// splade-core.js — SPLADE encoding and score decomposition in JavaScript.
// The same computation as splade_lens/lens.py, for the "Try your own text" box of the website,
// which runs the model in the browser with transformers.js. The output objects have the same
// shape as the ones written by export_site.py, so the page draws them with the same code.

export const TOP_TERMS = 60;

// w_j = max_i log(1 + relu(logit_ij)) over unmasked positions i; src_j = the maximising position.
export function spladeFromLogits(logits, L, V, mask) {
  const w = new Float32Array(V);
  const src = new Int32Array(V).fill(-1);
  for (let i = 0; i < L; i++) {
    if (!mask[i]) continue;
    const base = i * V;
    for (let j = 0; j < V; j++) {
      const x = logits[base + j];
      if (x > 0) {
        const a = Math.log1p(x);
        if (a > w[j]) { w[j] = a; src[j] = i; }
      }
    }
  }
  return { w, src };
}

// A representation in the format of export_site.py (plus `all`, the full term list, used for matching).
export function buildRep(text, ids, tokens, specialIds, w, src, idToToken, truncated) {
  const own = new Set(ids.filter((id) => !specialIds.has(id)));
  const all = [];
  for (let j = 0; j < w.length; j++) {
    if (w[j] > 0) all.push({ id: j, term: idToToken(j), w: w[j], pos: src[j], in_input: own.has(j) });
  }
  all.sort((a, b) => b.w - a.w || a.id - b.id);
  const wpt = new Array(tokens.length).fill(0);
  let total = 0, expw = 0;
  for (const t of all) { wpt[t.pos] += t.w; total += t.w; if (!t.in_input) expw += t.w; }
  return {
    text, tokens, truncated, all,
    weight_per_token: wpt, nnz: all.length, total_weight: total,
    expansion_weight_share: total > 0 ? expw / total : 0,
    terms: all.slice(0, TOP_TERMS).map((t) => ({ term: t.term, w: t.w, pos: t.pos, in_input: t.in_input })),
  };
}

// Score = sum over shared terms of w_q * w_d, split per term; "exact" if the term is in both texts.
export function explain(q, d) {
  const dmap = new Map(d.all.map((t) => [t.id, t]));
  const raw = [];
  for (const tq of q.all) {
    const td = dmap.get(tq.id);
    if (td) raw.push([tq, td, tq.w * td.w]);
  }
  const score = raw.reduce((s, r) => s + r[2], 0);
  const matches = raw.map(([tq, td, c]) => ({
    term: tq.term, id: tq.id, qw: tq.w, dw: td.w, c, share: score > 0 ? c / score : 0,
    kind: tq.in_input && td.in_input ? "exact" : "expansion", qpos: tq.pos, dpos: td.pos,
  }));
  matches.sort((a, b) => b.c - a.c || a.id - b.id);
  const expansion = matches.filter((m) => m.kind === "expansion").reduce((s, m) => s + m.share, 0);
  return { score, expansion_score_share: expansion, query: q, document: d, matches };
}

// Encode one text with a transformers.js tokenizer and masked-LM model.
export async function encode(tokenizer, model, text, maxLength) {
  if (!text || !text.trim()) throw new Error("empty text");
  const full = tokenizer(text, { truncation: false });
  const enc = tokenizer(text, { truncation: true, max_length: maxLength });
  const out = await model(enc);
  const [, L, V] = out.logits.dims;
  const ids = Array.from(enc.input_ids.data, Number);
  const mask = Array.from(enc.attention_mask.data, Number);
  const { w, src } = spladeFromLogits(out.logits.data, L, V, mask);
  const tokens = tokenizer.model.convert_ids_to_tokens(ids);
  const special = new Set(tokenizer.all_special_ids);
  const idToToken = (j) => tokenizer.model.convert_ids_to_tokens([j])[0];
  return buildRep(text, ids, tokens, special, w, src, idToToken, full.input_ids.dims.at(-1) > maxLength);
}
