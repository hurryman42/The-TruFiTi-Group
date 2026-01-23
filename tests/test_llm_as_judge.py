from src.evaluation.llm_as_judge import LLMAsJudge


def test_mauve_metric():
    judge = LLMAsJudge("gpt-oss")

    result1 = judge.compute("It’s literally so easy to make friends when you’re the dude with the good kush")

    result2 = judge.compute("Shocked at how humanist and hopeful this movie is given how cynical the original "
            "film is about the nature of man.")

    assert result1.score <= result2.score