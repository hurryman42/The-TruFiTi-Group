import json
import random
from tqdm import tqdm
from openai import OpenAI

LM_CLIENT = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="none",
)


def parse_numbered_results(text, n):
    results = {}
    lines = text.strip().split("\n")

    for line in lines:
        if ":" not in line:
            continue
        num, content = line.split(":", 1)
        num = num.strip()
        content = content.strip()

        if num.isdigit():
            results[int(num)] = content

    return [results.get(i, "DISCARD") for i in range(1, n + 1)]


def improve_reviews_batch(title, year, reviews):
    prompt = (
        f"Edit the following user reviews for the film {title} ({year})."
        "Correct spelling, spacing, grammar, and punctuation.\n"
        "Capitalize the first letter of every sentence.\n"
        "You may add missing sentence subjects such as “This film”, “The movie”, “It”, or “I” only to turn fragments "
        "into complete sentences."
        "Do NOT add any other new ideas or information.\n"
        "Preserve meaning, tone, humor, and sentiment.\n"
        "Remove emojis and non‑Latin symbols.\n"
        "If a review is meaningless or unusable → respond DISCARD.\n"
        "If already clean → respond SKIP.\n\n"
        "Return answers only as:\n"
        "1: text\n"
        "2: text\n"
        "3: text\n\n"
        "Reviews:\n"
    )
    for i, r in enumerate(reviews, 1):
        prompt += f"{i}: {r}\n"

    response = LM_CLIENT.chat.completions.create(
        model="local-model",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=2000,
    )

    raw = response.choices[0].message.content.strip()
    parsed = parse_numbered_results(raw, len(reviews))

    cleaned = []
    for original, result in zip(reviews, parsed, strict=True):
        match result:
            case "SKIP":
                cleaned.append(original)
            case "DISCARD":
                cleaned.append(None)
            case _:
                cleaned.append(result.strip())
    return cleaned


def improve_reviews_per_film(data):
    title = data.get("title")
    year = data.get("year")
    reviews = data.get("review_texts", [])

    if not reviews:
        return data

    improved_reviews = improve_reviews_batch(title, year, reviews)
    improved_reviews = [r for r in improved_reviews if r is not None]

    return {
        "title": data.get("title"),
        "year": data.get("year"),
        "synopsis": data.get("synopsis"),
        "review_texts": improved_reviews,
    }


def main():
    film_counter = 0

    input_path = "data/letterboxd_filtered_0.99.jsonl"
    output_path = "data/letterboxd_filtered_llm.jsonl"

    with open(input_path, encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in tqdm(infile, desc="Processing films"):
            if film_counter >= 100:
                break
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            film_counter += 1

            result = improve_reviews_per_film(data)
            outfile.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("Finished processing dataset. Saved to:", output_path)


def test_random_films(n=3, input_file="data/letterboxd_filtered.jsonl"):
    lines = []
    with open(input_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < n:
                lines.append(line)
            else:
                j = random.randint(0, i)
                if j < n:
                    lines[j] = line

    print("Testing on", n, "random films:\n")

    for raw in lines:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON when testing, skipping. {e}")
            continue

        title = data.get("title")
        year = data.get("year")
        reviews = data.get("review_texts")

        print(f"{title} ({year})")
        print("--- Original Reviews ---")
        for r in reviews:
            print("-", r)

        print("\n--- Improved Reviews ---")
        improved_reviews = improve_reviews_batch(title, year, reviews)
        improved_reviews = [r for r in improved_reviews if r is not None]
        for r in improved_reviews:
            print("-", r)

        print("\n" + "=" * 40 + "\n")


if __name__ == "__main__":
    main()
    # test_random_films(2)
