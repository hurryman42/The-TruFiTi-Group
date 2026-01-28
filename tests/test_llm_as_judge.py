import pytest

from src.evaluation.llm_as_judge import LLMAsJudge


@pytest.mark.skip(reason="Requires external setup (LM Studio), skipped by default")
def test_mauve_metric():
    judge = LLMAsJudge("llama-3.1-8b")

    result1 = judge.compute("It’s literally so easy to make friends when you’re the dude with the good kush")

    result2 = judge.compute(
        "Shocked at how humanist and hopeful this movie is given how cynical the original "
        "film is about the nature of man."
    )

    assert result1.score <= result2.score
