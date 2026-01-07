import pytest

from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.enums.types import SpecialTokensEnum


NUM_SPECIAL = len(SpecialTokensEnum)
BASE_VOCAB_SIZE = NUM_SPECIAL + 256


@pytest.fixture
def tokenizer():
    return BPETokenizer()


def test_special_tokens_initialized(tokenizer):
    assert tokenizer.pad_id == 0
    assert tokenizer.bos_id == 1
    assert tokenizer.eos_id == 2


def test_byte_vocabulary_initialized(tokenizer):
    # length special tokens + 256 bytes
    assert tokenizer.get_vocab_size() == BASE_VOCAB_SIZE


def test_encode_with_special_tokens(tokenizer):
    tokens = tokenizer.encode_with_special_tokens("A")
    assert tokens == [1, 65 + NUM_SPECIAL, 2]  # BOS, A, EOS


def test_decode_roundtrip(tokenizer):
    original = "Hello World!"
    tokens = tokenizer.encode(original)
    decoded = tokenizer.decode(tokens)
    assert decoded == original


def test_decode_skips_special_tokens_by_default(tokenizer):
    tokens = [1, 65 + NUM_SPECIAL, 66 + NUM_SPECIAL, 2]  # BOS, A, B, EOS
    decoded = tokenizer.decode(tokens)
    assert decoded == "AB"


def test_train_creates_merge_rules():
    texts = ["aaaa aaaa aaaa"] * 100
    tokenizer = BPETokenizer.train(texts, target_size=BASE_VOCAB_SIZE + 5)

    expected_merges = (BASE_VOCAB_SIZE + 5) - BASE_VOCAB_SIZE
    assert len(tokenizer._merge_rules) > 0
    assert len(tokenizer._merge_rules) == expected_merges
