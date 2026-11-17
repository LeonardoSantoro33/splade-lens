import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cli(tiny_model_dir):
    env = dict(os.environ, SPLADE_MODEL=tiny_model_dir)
    r = subprocess.run([sys.executable, "-m", "splade_lens", "low iron", "anemia is low iron .", "--json",
                        "--max-length", "16"], env=env, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "score" in json.loads(r.stdout)


def test_app_functions(tiny_model_dir, monkeypatch):
    monkeypatch.setenv("SPLADE_MODEL", tiny_model_dir)
    monkeypatch.setenv("SPLADE_MAX_LENGTH", "16")
    sys.path.insert(0, str(ROOT))
    import app
    out = app.run_explain("symptoms of low iron", "anemia is low iron .")
    assert out[0].startswith("**Score:")
    assert len(out[1]) == len(app.lens.encode("symptoms of low iron").tokens)
    text, table = app.run_compare("low iron", "anemia is low iron .", "iron is a metal .")
    assert "Document A" in text and list(table.columns)[0] == "term"


def test_export_site(tiny_model_dir, tmp_path):
    ex = tmp_path / "ex.json"
    ex.write_text(json.dumps({
        "examples": [{"id": "a", "query": "symptoms of low iron", "document": "anemia is low iron in the blood ."}],
        "comparisons": [{"id": "c", "query": "low iron", "doc_a": "anemia is low iron .", "doc_b": "iron is a metal ."}]}))
    out = tmp_path / "data.js"
    env = dict(os.environ, SPLADE_MODEL=tiny_model_dir)
    r = subprocess.run([sys.executable, str(ROOT / "export_site.py"), "--examples", str(ex), "--out", str(out),
                        "--results", str(tmp_path / "none"), "--max-length", "16"],
                       env=env, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    text = out.read_text()
    assert text.startswith("window.SPLADE_DATA = ") and text.rstrip().endswith(";")
    data = json.loads(text[len("window.SPLADE_DATA = "):].rstrip().rstrip(";"))
    e = data["examples"][0]
    assert abs(sum(m["c"] for m in e["matches"]) - e["score"]) < 1e-2
    assert all(0 <= m["qpos"] < len(e["query"]["tokens"]) and 0 <= m["dpos"] < len(e["document"]["tokens"])
               for m in e["matches"])
    assert len(e["query"]["weight_per_token"]) == len(e["query"]["tokens"])
    assert data["comparisons"][0]["diff"] and data["findings"] == []


def test_convert_onnx(tiny_model_dir, tmp_path):
    import pytest
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    out = tmp_path / "onnx_model"
    r = subprocess.run([sys.executable, str(ROOT / "convert_onnx.py"), "--model", tiny_model_dir, "--out", str(out)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    for f in ["config.json", "tokenizer.json", "onnx/model.onnx", "onnx/model_quantized.onnx", "README.md"]:
        assert (out / f).exists(), f
    diffs = json.loads((out / "conversion.json").read_text())["max_abs_weight_diff"]
    assert diffs["model.onnx"] < 1e-4          # full precision must match PyTorch
