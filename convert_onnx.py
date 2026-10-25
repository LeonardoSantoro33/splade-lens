"""
convert_onnx.py — Convert the SPLADE model to ONNX so that it can run inside a web browser.

The website's "Try your own text" box runs the model in the visitor's browser with transformers.js,
which reads ONNX models. This script:
  1. exports the masked-language model to ONNX (onnx/model.onnx, full precision);
  2. writes an 8-bit quantised copy (onnx/model_quantized.onnx, about four times smaller), which is
     the one the website downloads;
  3. checks both against PyTorch on a few texts and prints how much they differ;
  4. optionally uploads the folder to a Hugging Face model repository (--push).

    pip install onnx onnxruntime
    python convert_onnx.py --out splade_onnx
    python convert_onnx.py --out splade_onnx --push <your_hf_username>/splade-cocondenser-ensembledistil-onnx

--push needs a Hugging Face token with write access (export HF_TOKEN=...). The model's licence
(CC BY-NC-SA 4.0) allows redistribution with attribution, non-commercially and under the same
licence; the model card written here says so.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

DEFAULT_MODEL = "naver/splade-cocondenser-ensembledistil"
CHECK_TEXTS = ["symptoms of low iron",
               "Iron deficiency anemia often causes fatigue, pale skin and shortness of breath.",
               "cheap flights to rome"]

CARD = """---
license: cc-by-nc-sa-4.0
library_name: transformers.js
base_model: {base}
tags: [splade, learned-sparse-retrieval, onnx]
---

# {name}

ONNX export of [{base}](https://huggingface.co/{base}) for use in the browser with
[transformers.js](https://huggingface.co/docs/transformers.js): `onnx/model_quantized.onnx` (8-bit
dynamic quantisation) and, if present, `onnx/model.onnx` (full precision). Made for
[SPLADE lens](https://github.com/leonardosantoro33/splade-lens).

The weights are those of the original model, converted without retraining. The original model is
described in T. Formal, C. Lassance, B. Piwowarski, S. Clinchant, *From Distillation to Hard
Negative Sampling: Making Sparse Neural IR Models More Effective*, SIGIR 2022, and is distributed
under CC BY-NC-SA 4.0; this conversion is distributed under the same licence.

Measured differences against PyTorch on a few test texts (largest absolute difference of a SPLADE
weight): full precision {d32:.2e}, 8-bit {d8:.3f}.
"""


def splade(logits, mask):
    act = np.log1p(np.maximum(logits, 0.0)) * mask[..., None]
    return act.max(axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.environ.get("SPLADE_MODEL", DEFAULT_MODEL))
    ap.add_argument("--out", default="splade_onnx")
    ap.add_argument("--push", default="", help="Hugging Face repo id to upload to, e.g. user/name")
    ap.add_argument("--push-fp32", action="store_true",
                    help="also upload the full-precision model (about 440 MB; the website only needs the 8-bit one)")
    args = ap.parse_args()

    from transformers import AutoModelForMaskedLM, AutoTokenizer
    out = Path(args.out)
    (out / "onnx").mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model, attn_implementation="eager").eval()
    model.config.save_pretrained(out)
    tok.save_pretrained(out)
    if not (out / "tokenizer.json").exists():
        raise SystemExit("[ERROR] no tokenizer.json written: transformers.js needs a fast tokenizer")

    # 1. export ------------------------------------------------------------------------------------
    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.m(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits

    enc = tok(CHECK_TEXTS[:2], padding=True, return_tensors="pt")
    names = ["input_ids", "attention_mask", "token_type_ids"]
    dyn = {n: {0: "batch_size", 1: "sequence_length"} for n in names}
    dyn["logits"] = {0: "batch_size", 1: "sequence_length"}
    f32 = out / "onnx" / "model.onnx"
    torch.onnx.export(Wrapper(model), tuple(enc[n] for n in names), str(f32), input_names=names,
                      output_names=["logits"], dynamic_axes=dyn, opset_version=17, dynamo=False)
    print(f"exported {f32} ({f32.stat().st_size / 1e6:.0f} MB)")

    # 2. quantise ----------------------------------------------------------------------------------
    from onnxruntime.quantization import QuantType, quantize_dynamic
    q8 = out / "onnx" / "model_quantized.onnx"
    quantize_dynamic(str(f32), str(q8), weight_type=QuantType.QInt8)
    print(f"quantised {q8} ({q8.stat().st_size / 1e6:.0f} MB)")

    # 3. check -------------------------------------------------------------------------------------
    import onnxruntime as ort
    model = AutoModelForMaskedLM.from_pretrained(args.model).eval()   # fresh reference copy
    diffs = {}
    for path in (f32, q8):
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        worst = 0.0
        for text in CHECK_TEXTS:
            e = tok(text, return_tensors="np")
            feeds = {n: e[n].astype(np.int64) for n in names}
            got = splade(sess.run(["logits"], feeds)[0], feeds["attention_mask"])
            with torch.no_grad():
                ref = model(**{n: torch.from_numpy(feeds[n]) for n in names}).logits.numpy()
            want = splade(ref, feeds["attention_mask"])
            worst = max(worst, float(np.abs(got - want).max()))
            top_g, top_w = set(np.argsort(-got[0])[:20]), set(np.argsort(-want[0])[:20])
            print(f"  {path.name}: '{text[:30]}' top-20 terms in common {len(top_g & top_w)}/20")
        diffs[path.name] = worst
        print(f"  {path.name}: largest SPLADE weight difference vs PyTorch {worst:.4g}")

    name = args.push.split("/")[-1] if args.push else out.name
    (out / "README.md").write_text(CARD.format(base=args.model, name=name, d32=diffs["model.onnx"],
                                               d8=diffs["model_quantized.onnx"]))
    (out / "conversion.json").write_text(json.dumps({"base_model": args.model, "max_abs_weight_diff": diffs}, indent=2))

    # 4. upload ------------------------------------------------------------------------------------
    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=args.push, repo_type="model",
                          commit_message="ONNX export of " + args.model,
                          ignore_patterns=[] if args.push_fp32 else ["onnx/model.onnx"])
        print(f"uploaded to https://huggingface.co/{args.push}")
    print("Done.")


if __name__ == "__main__":
    main()
