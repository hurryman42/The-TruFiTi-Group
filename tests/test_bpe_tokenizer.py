import pytest

from src.tokenizer.bpe_tokenizer import BPETokenizer

@pytest.fixture
def tokenizer():
    return BPETokenizer.train(["a cat with a hat eats the hat cat"], target_size=260)

def test_encode_decode(tokenizer):
    test_text = "brat the mat"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    assert decoded == test_text

def test_merge_rules(tokenizer):
    result = tokenizer._apply_merge_rule([2,3,5,67,45],[5,67,90])
    assert result == [2,3,90,45]