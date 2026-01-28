import os
import pytest

if os.getenv("GITHUB_ACTIONS") == "true":
    pytest.skip("Skipped in CI (requires HF downloads)", allow_module_level=True)

from src.evaluation.toxicity import Toxicity


def test_meteor_metric():
    predictions = ["This was a terrible fucking movie, I hated it.", "Such a good film, it was awesome."]
    toxicity_original = Toxicity("original")
    toxicity_unbiased = Toxicity("unbiased")

    result_original = list()
    result_unbiased = list()
    for prediction in predictions:
        result_original.append(toxicity_original.compute(prediction, None))
        result_unbiased.append(toxicity_unbiased.compute(prediction, None))

    assert result_original[0].score > result_original[1].score
    assert result_unbiased[0].score > result_unbiased[1].score
