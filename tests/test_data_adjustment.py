from src.data.data_adjustment import ReviewAdjuster


def test_review_adjuster():
    review_texts = ["this is a correct sentence", "this is not a correct senence"]
    review_adjuster = ReviewAdjuster()

    result = list()
    for review_text in review_texts:
        result.append(review_adjuster.adjust_review(review_text))

    assert result[0] == "this is a correct sentence"
    assert result[1] == "this is not a correct sentence"
