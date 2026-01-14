from src.evaluation.meteor_metric import MeteorMetric


def test_meteor_metric():
    predictions = ["It is a guide to action which ensures that the military always obeys the commands of the party"]
    references = [
        [
            "It is a guide to action that ensures that the military will forever heed Party commands",
            "It is the guiding principle which guarantees the military forces always being under the commands of the party",
        ]
    ]
    meteor = MeteorMetric()

    result = meteor.compute(predictions, references)

    assert result.score > 0.6
