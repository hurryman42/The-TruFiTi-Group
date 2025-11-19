from src.mvp import list_to_sentence, sentence_to_list


def test_list_to_sentence():
    assert list_to_sentence(["This", "is", "a", "test"]) == "This is a test", "Converting lists to sentences failed!"


def test_sentence_to_list():
    assert sentence_to_list("This is a test") == ["This", "is", "a", "test"], "Converting sentences to lists failed!"
