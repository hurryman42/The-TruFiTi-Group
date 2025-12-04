from src.evaluation.rouge_n_metric import RougeNMetric

def test_rouge_n_metric():
    examples = ["The cat sat on the mat.", "the cat and the dog around the hat"]
    references = [["The cat is on the mat."], ["cat and the dog around the hat"]]

    references_2 = [["fat mat on that mat."], ["hat and the dog around the cat"]]

    metric = RougeNMetric("rouge1")
    result_1 = metric.compute(examples, references)
    result_2 = metric.compute(examples, references_2)
    assert result_1.name == "rouge1-score"
    assert result_2.score <= result_1.score
