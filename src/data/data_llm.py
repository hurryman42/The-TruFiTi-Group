import os
import json
import random
from tqdm import tqdm
from openai import OpenAI

from src.data.data_utils import make_key, load_processed_keys, load_unprocessed_keys

LM_CLIENT = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="none",
)

CACHE_FILE = "data/review_cache.json"
REVIEW_CACHE = {}

SYSTEM_PROMPT = """
You are an assistant that edits user-written movie reviews.
Your task:
- Fix grammar, spelling, punctuation, and capitalization.
- Improve clarity while preserving meaning, tone, and intent.
- Rewrite sentences only when needed to make them natural English.
- Treat each review independently; do not copy formatting or style from one review to another.
- If a review is already clean, output SKIP.
- If a review contains no understandable English, output DISCARD.
- Output exactly one line per review in the format:
  1: text
  2: text
  3: text
Do not output anything else.
"""


def load_cache():
    global REVIEW_CACHE
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, encoding="utf-8") as f:
                REVIEW_CACHE = json.load(f)
        except (OSError, json.JSONDecodeError):
            REVIEW_CACHE = {}


def save_cache():
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(REVIEW_CACHE, f, ensure_ascii=False)


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


def improve_reviews_per_film(data):
    title = data.get("title")
    year = data.get("year")
    reviews = data.get("review_texts", [])
    film_key = make_key(title, year)

    if film_key in REVIEW_CACHE:
        cached = REVIEW_CACHE[film_key]
        return {
            "title": title,
            "year": year,
            "synopsis": data.get("synopsis"),
            "review_texts": [r for r in cached if r],
        }

    if not reviews:
        REVIEW_CACHE[film_key] = []
        return {
            "title": title,
            "year": year,
            "synopsis": data.get("synopsis"),
            "review_texts": [],
        }

    user_prompt = f'Reviews for "{title}":\n'
    for i, r in enumerate(reviews, 1):
        user_prompt += f"{i}: {r}\n"

    try:
        response = LM_CLIENT.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=400,
        )
    except Exception as e:
        print(f"Error: {e}")
        REVIEW_CACHE[film_key] = []
        return {
            "title": title,
            "year": year,
            "synopsis": data.get("synopsis"),
            "review_texts": [],
        }

    raw = response.choices[0].message.content
    parsed = parse_numbered_results(raw, len(reviews))

    improved = []
    for original, result in zip(reviews, parsed, strict=False):
        match result:
            case "SKIP":
                improved.append(original)
            case "DISCARD":
                improved.append(None)
            case _:
                improved.append(result.strip())

    REVIEW_CACHE[film_key] = improved
    return {
        "title": title,
        "year": year,
        "synopsis": data.get("synopsis"),
        "review_texts": [r for r in improved if r],
    }


def main(input_file, output_file):
    load_cache()
    processed_keys = load_processed_keys(output_file)
    cache_keys = set(REVIEW_CACHE.keys())
    keys_to_skip = processed_keys | cache_keys
    print(f"Already processed: {len(keys_to_skip)} films")

    unprocessed = load_unprocessed_keys(input_file, keys_to_skip)
    print(f"Films remaining to process: {len(unprocessed)}")

    processed_count = 0

    with open(output_file, "a", encoding="utf-8") as outfile:
        for line in tqdm(unprocessed, desc="Processing films"):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            result = improve_reviews_per_film(data)

            outfile.write(json.dumps(result, ensure_ascii=False) + "\n")

            processed_count += 1
            if processed_count % 50 == 0:
                save_cache()
        save_cache()

    print(f"Processed films: {processed_count}")
    print(f"Saved to: {output_file}")
    print(f"Cache size:      {len(REVIEW_CACHE)}")


def requery_all_discard_films(input_file, output_file, updated_output_file):
    load_cache()

    original_map = {}
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = make_key(data.get("title"), data.get("year"))
            original_map[key] = data

    with open(output_file, encoding="utf-8") as infile, open(updated_output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            try:
                film = json.loads(line)
            except json.JSONDecodeError:
                continue

            title = film.get("title")
            year = film.get("year")
            key = make_key(title, year)

            reviews = film.get("review_texts", [])
            all_discard = (not reviews) or all(r is None or r == "DISCARD" for r in reviews)

            if not all_discard:
                outfile.write(json.dumps(film, ensure_ascii=False) + "\n")
                continue

            if key not in original_map:
                outfile.write(json.dumps(film, ensure_ascii=False) + "\n")
                continue

            original = original_map[key]
            print(f"Re-querying {title} ({year}) – all reviews discarded.")

            new_result = improve_reviews_per_film(original)

            REVIEW_CACHE[key] = [r if r is not None else None for r in new_result.get("review_texts", [])]

            outfile.write(json.dumps(new_result, ensure_ascii=False) + "\n")

        save_cache()

    print(f"Completed re-query. Updated file saved to: {updated_output_file}")
    print(f"Cache size is now {len(REVIEW_CACHE)}.")


def test_random_films(input_file="data/letterboxd_filtered_post.jsonl", n=3):
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
        except json.JSONDecodeError:
            continue

        title = data.get("title")
        year = data.get("year")
        reviews = data.get("review_texts", [])

        print(f"{title} ({year})")
        print("--- Original Reviews ---")
        for r in reviews:
            print("-", r)
        print("")

        result = improve_reviews_per_film(data)
        improved_reviews = result["review_texts"]

        print("--- Improved Reviews ---")
        padded_improved = improved_reviews + [None] * (len(reviews) - len(improved_reviews))
        for orig, new in zip(reviews, padded_improved, strict=False):
            if new is None:
                print("- [DISCARD]")
            elif new == orig:
                print("- [SKIP]", orig)
            else:
                print("-", new)

        print("\n" + "=" * 40 + "\n")


if __name__ == "__main__":
    main(input_file="data/splits/part_3.jsonl", output_file="data/tomerge/letterboxd_filtered_llm_part_3.jsonl")
    # test_random_films()
