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

def test_save_and_load(tokenizer, tmp_path):
    # Use tmp_path to create a temporary directory
    temp_dir = tmp_path / "my_temp_dir"
    temp_dir.mkdir()

    temp_file = temp_dir / "test_file.txt"

    tokenizer.save(temp_file)

    tokenizer2 = BPETokenizer.load(temp_file)

    print("TEST:", tokenizer2._vocabulary)

    test_text = "brat the mat"
    encoded = tokenizer2.encode(test_text)
    decoded = tokenizer2.decode(encoded)

    assert decoded == test_text