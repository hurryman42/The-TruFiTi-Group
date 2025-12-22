import json

from src.enums.types import DataEnum


def read_file_only_reviews(file_path) -> list[str]:
    reviews = []
    with open(file_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON in line {i}: {e}")
                continue

            if data.get(DataEnum.REVIEW_TEXTS):
                reviews.extend(data[DataEnum.REVIEW_TEXTS])

    print(f"Number of reviews: {len(reviews):,}".replace(",", "."))

    return reviews


def read_file_synopsis_review_pairs(file_path) -> list[str]:
    pairs = []
    with open(file_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON in line {i}: {e}")
                continue

            synopsis = data.get(DataEnum.SYNOPSIS)
            if not synopsis or not data.get(DataEnum.REVIEW_TEXTS):
                continue

            for review in data[DataEnum.REVIEW_TEXTS]:
                pairs.append(f"{synopsis} {review}")

    print(f"Number of synopsis-review-pairs: {len(pairs):,}".replace(",", "."))

    return pairs
