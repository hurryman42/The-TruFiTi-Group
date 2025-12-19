import pytest

from src.tokenizer.bpe_tokenizer import BPETokenizer


@pytest.fixture
def tokenizer():
    return BPETokenizer()


def test_special_tokens_initialized(tokenizer):
    assert tokenizer.pad_id == 0
    assert tokenizer.bos_id == 1
    assert tokenizer.eos_id == 2


def test_byte_vocabulary_initialized(tokenizer):
    # 3 special tokens + 256 bytes
    assert tokenizer.get_vocab_size == 259


def test_encode_with_special_tokens(tokenizer):
    tokens = tokenizer.encode_with_special_tokens("A")
    assert tokens == [1, 68, 2]  # BOS, A, EOS


def test_decode_roundtrip(tokenizer):
    original = "Hello World!"
    tokens = tokenizer.encode(original)
    decoded = tokenizer.decode(tokens)
    assert decoded == original


def test_decode_skips_special_tokens_by_default(tokenizer):
    tokens = [1, 68, 69, 2]  # BOS, A, B, EOS
    decoded = tokenizer.decode(tokens)
    assert decoded == "AB"


def test_train_creates_merge_rules():
    texts = ["aaaa aaaa aaaa"] * 100
    tokenizer = BPETokenizer.train(texts, target_size=264)

    assert len(tokenizer._merge_rules) > 0
    assert len(tokenizer._merge_rules) == 5  # 265 - 259 base vocab
