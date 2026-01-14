from src.evaluation.mauve_metric import MauveMetric

def test_mauve_metric():
    predictions = ["It is a guide to action which ensures that the military always obeys the commands of the party"]
    references = [['It is a guide to action that ensures that the military will forever heed Party commands',
                   'It is the guiding principle which guarantees the military forces always being under the commands of the party']]
    mauve = MauveMetric()

    result = mauve.compute(predictions, references)

    assert result.score > 0