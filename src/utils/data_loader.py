import json


def read_file_only_reviews(file_path) -> list[str]:
    reviews = []
    with open(file_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON in line {i}: {e}")
                continue

            if data.get("review_texts"):
                reviews.extend(data["review_texts"])

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

            synopsis = data.get("synopsis")
            if not synopsis or not data.get("review_texts"):
                continue

            for review in data["review_texts"]:
                pairs.append(f"{synopsis} {review}")

    print(f"Number of synopsis-review-pairs: {len(pairs):,}".replace(",", "."))

    return pairs
