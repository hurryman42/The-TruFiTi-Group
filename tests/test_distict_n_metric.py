from src.evaluation.distinct_n_metric import DistinctNMetric

def test_distinct_n_metric():
    generated = ["<BOS> A B A B <EOS>", "<BOS> B A B A <EOS>"]

    metric1 = DistinctNMetric(n=1)
    result1 = metric1.compute(generated)
    assert result1.details["unique_ngrams"] == 2  # A, B
    assert result1.details["total_ngrams"] == 8  # 4 tokens in each sentence
    assert abs(result1.score - (2 / 8)) < 1e-6  # = 0.25

    metric2 = DistinctNMetric(n=2)
    result2 = metric2.compute(generated)
    assert result2.name == "distinct-2"
    assert result2.details["unique_ngrams"] == 2  # (A,B) and (B,A)
    assert result2.details["total_ngrams"] == 6  # 3 each sentence
    assert abs(result2.score - (2 / 6)) < 1e-6  # = 0.3333