from src.evaluation.rouge_n_metric import RougeNMetric

def test_rouge_n_metric():
    examples = ["The cat sat on the mat.", "the cat and the dog around the hat"]
    references = [["The cat is on the mat."], ["cat and the dog around the hat"]]
    references_2 = [["fat mat on that mat."], ["hat and the dog around the cat"]]

    metric_rouge1 = RougeNMetric("rouge1")
    result_1 = metric_rouge1.compute(examples, references)
    result_2 = metric_rouge1.compute(examples, references_2)
    assert result_1.name == "rouge1-score"
    assert result_2.score <= result_1.score

    metric_rouge2 = RougeNMetric("rouge2")
    result_1 = metric_rouge2.compute(examples, references)
    result_2 = metric_rouge2.compute(examples, references_2)
    assert result_1.name == "rouge2-score"
    assert result_2.score <= result_1.score

    metric_rougeL = RougeNMetric("rougeL")
    result_1 = metric_rougeL.compute(examples, references)
    result_2 = metric_rougeL.compute(examples, references_2)
    assert result_1.name == "rougeL-score"
    assert result_2.score <= result_1.score