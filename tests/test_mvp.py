from src.mvp import sentence_to_list, list_to_sentence


def test_list_to_sentence():
    assert list_to_sentence(["This", "is", "a", "test"]) == "This is a test", "Converting lists to sentences failed!"

def test_sentence_to_list():
    assert sentence_to_list("This is a test") == ["This", "is", "a", "test"], "Converting sentences to lists failed!"
