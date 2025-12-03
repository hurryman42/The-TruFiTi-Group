from src.evaluation.distinct_n_metric import DistinctNMetric

def test_distinct_n_metric():
    generated = ["A B A B", "B A B A"]
    metric = DistinctNMetric(n=1)
    result = metric.compute(generated)
    assert result.name == "distinct-1"
    # there are 2 unique unigrams: 'A', 'B'
    assert abs(result.score - 1.0) < 1e-5

    metric2 = DistinctNMetric(n=2)
    result2 = metric2.compute(generated)
    # for bigrams: 'A B', 'B A'
    assert abs(result2.score - 1.0) < 1e-5