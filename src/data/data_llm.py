import json
import random
import openai
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
    prompt = f"""
        Edit the following user reviews for the film {title} ({year}).
        - Correct spelling, spacing, grammar, and punctuation.
        - Capitalize the first letter of every sentence.
        - You may add missing sentence subjects such as “This film”, “The movie”, “It”, or “I” to turn fragments or
        incomplete sentences into complete sentences.
        - Your perspective is that of a user rating a film.
        - Preserve meaning, tone, humor, and sentiment.
        - Remove emojis and non‑Latin symbols.
        - If a review is meaningless or unusable → respond DISCARD
        - If already clean → respond SKIP
        
        Return answers only as:
        1: text
        2: text
        3: text
        
        Reviews:
        """
    for i, r in enumerate(reviews, 1):
        prompt += f"{i}: {r}\n"

    try:
        response = LM_CLIENT.chat.completions.create(
            model="local-model",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=2000,
        )
    except openai.APIConnectionError as e:
        print(e)
        return []

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


def main(input_path="data/letterboxd_filtered.jsonl"):
    film_counter = 0

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


def test_random_films(n=3, input_path="data/letterboxd_filtered.jsonl"):
    lines = []
    with open(input_path, encoding="utf-8") as f:
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
            print("\n-", r)

        print("\n--- Improved Reviews ---")
        improved_reviews = improve_reviews_batch(title, year, reviews)
        improved_reviews = [r for r in improved_reviews if r is not None]
        for r in improved_reviews:
            print("\n-", r)

        print("\n" + "=" * 40 + "\n")


if __name__ == "__main__":
    # main()
    test_random_films(1)
