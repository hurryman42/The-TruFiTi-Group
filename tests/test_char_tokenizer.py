import pytest

from src.tokenizer.CharTokenizer import CharTokenizer

@pytest.fixture
def tokenizer():
    return CharTokenizer.train(["hello world"])


def test_train_creates_vocabulary(tokenizer):
    assert len(tokenizer._chars) > 0
    assert 'h' in tokenizer._chars
    assert 'e' in tokenizer._chars
    assert 'l' in tokenizer._chars
    assert 'o' in tokenizer._chars

def test_encode_decode(tokenizer):
    test_text = "hello"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    assert decoded == test_text