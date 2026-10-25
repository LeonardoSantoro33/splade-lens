"""A tiny, randomly initialised BERT masked-LM with a small WordPiece vocabulary, built offline.
Its outputs are meaningless, but they exercise exactly the same code paths as the real model."""
import pytest
import torch

WORDS = ["iron", "low", "symptoms", "of", "anemia", "blood", "test", "the", "a", "is", "metal", "pale", "skin",
         "cheap", "flights", "to", "rome", "italy", "capital", "budget", "##s", "##ing", "vaccine", "immune",
         "system", "work", "how", "do", ".", ",", "and", "doctor"]


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    from transformers import BertConfig, BertForMaskedLM, BertTokenizer
    d = tmp_path_factory.mktemp("tiny_splade")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS
    (d / "vocab.txt").write_text("\n".join(vocab) + "\n")
    tok = BertTokenizer(str(d / "vocab.txt"), do_lower_case=True)
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=64)
    model = BertForMaskedLM(cfg)
    with torch.no_grad():                       # shift the MLM bias up so that many logits are > 0
        model.cls.predictions.bias.add_(0.5)
    model.save_pretrained(d)
    tok.save_pretrained(d)
    return str(d)


@pytest.fixture(scope="session")
def lens(tiny_model_dir):
    from splade_lens import SpladeLens
    return SpladeLens.from_pretrained(tiny_model_dir, max_length=16, device="cpu")
