import os
import json
import random
from tqdm import tqdm
from openai import OpenAI

from src.data.data_helper import make_key, load_processed_keys, load_unprocessed_keys

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
- If a review is already clean, output SKIP.
- If a review contains no understandable English, output DISCARD.
- Output exactly one line per review in the format:
  1: text
  2: text
  3: text
Do not output anything else.
"""

INSTRUCTION_SECTION = """
<BEGIN_INSTRUCTION>
You edit user reviews while preserving meaning, tone, and intent.

Rules:
- Improve clarity, grammar, and readability.
- Fix spelling, punctuation, and capitalization.
- Rewrite sentences when needed to make them natural and fluent, but keep the original meaning.
- Maintain the reviewer’s emotional tone and style.
- If a review is already clean and natural, output SKIP.
- If the review has no understandable English or no coherent meaning, output DISCARD.

Output EXACTLY this structure:

<RESULTS>
<FILM title="TITLE">
1: corrected text
2: corrected text
...
</FILM>
(repeat for each film)
</RESULTS>

Follow this format strictly.
<END_INSTRUCTION>
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


def build_batch_prompt(films):
    txt = "<BEGIN_INPUT>\n"
    for film in films:
        txt += f'<FILM title="{film["title"]}">\n'
        for i, r in enumerate(film["reviews"], 1):
            txt += f"{i}: {r}\n"
        txt += "</FILM>\n\n"
    txt += "<END_INPUT>"
    return INSTRUCTION_SECTION + txt


def parse_film_block(text, expected_count):
    lines = text.strip().split("\n")
    results = {}
    for line in lines:
        if ":" not in line:
            continue
        num, val = line.split(":", 1)
        num = num.strip()
        if num.isdigit():
            results[int(num)] = val.strip()
    out = []
    for i in range(1, expected_count + 1):
        out.append(results.get(i, "DISCARD"))
    return out


def parse_batch_output(output, films):
    if "<RESULTS>" not in output or "</RESULTS>" not in output:
        return [["DISCARD"] * len(f["reviews"]) for f in films]

    core = output.split("<RESULTS>", 1)[1].split("</RESULTS>", 1)[0]

    blocks = core.split("</FILM>")
    film_blocks = []
    for block in blocks:
        if '<FILM title="' in block:
            film_blocks.append(block)

    results = []
    for block, film in zip(film_blocks, films, strict=False):
        lines = block.split("\n", 1)[1] if "\n" in block else ""
        parsed = parse_film_block(lines, len(film["reviews"]))
        results.append(parsed)

    return results


def improve_reviews_batch(films, max_tokens=200):
    full_prompt = build_batch_prompt(films)

    try:
        resp = LM_CLIENT.chat.completions.create(
            model="local-model",
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0,
            max_tokens=max_tokens * len(films),
        )
    except Exception as e:
        print("Batch error:", e)
        return [["DISCARD"] * len(f["reviews"]) for f in films]

    output = resp.choices[0].message.content
    return parse_batch_output(output, films)


def improve_reviews_batched(input_file, output_file, batch_size=16):
    load_cache()
    processed = load_processed_keys(output_file)
    cache_keys = set(REVIEW_CACHE.keys())
    skip = processed | cache_keys

    unprocessed = load_unprocessed_keys(input_file, skip)
    print("Already processed:", len(skip))
    print("Remaining:", len(unprocessed))

    with open(output_file, "a", encoding="utf-8") as outfile:
        batch = []

        for line in tqdm(unprocessed, desc="Batching films"):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            title = data.get("title")
            year = data.get("year")
            reviews = data.get("review_texts", [])
            key = make_key(title, year)

            if key in REVIEW_CACHE:
                cleaned = [r for r in REVIEW_CACHE[key] if r]
                outfile.write(
                    json.dumps(
                        {"title": title, "year": year, "synopsis": data.get("synopsis"), "review_texts": cleaned},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            batch.append(
                {"title": title, "year": year, "reviews": reviews, "key": key, "synopsis": data.get("synopsis")}
            )

            if len(batch) >= batch_size:
                improved_list = improve_reviews_batch(batch)
                for film, improved in zip(batch, improved_list, strict=False):
                    cleaned = []
                    for orig, r in zip(film["reviews"], improved, strict=False):
                        if r == "SKIP":
                            cleaned.append(orig)
                        elif r == "DISCARD":
                            cleaned.append(None)
                        else:
                            cleaned.append(r)
                    REVIEW_CACHE[film["key"]] = cleaned
                    outfile.write(
                        json.dumps(
                            {
                                "title": film["title"],
                                "year": film["year"],
                                "synopsis": film["synopsis"],
                                "review_texts": [x for x in cleaned if x],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                batch = []
                save_cache()

        if batch:
            improved_list = improve_reviews_batch(batch)
            for film, improved in zip(batch, improved_list, strict=False):
                cleaned = []
                for orig, r in zip(film["reviews"], improved, strict=False):
                    if r == "SKIP":
                        cleaned.append(orig)
                    elif r == "DISCARD":
                        cleaned.append(None)
                    else:
                        cleaned.append(r)
                REVIEW_CACHE[film["key"]] = cleaned
                outfile.write(
                    json.dumps(
                        {
                            "title": film["title"],
                            "year": film["year"],
                            "synopsis": film["synopsis"],
                            "review_texts": [x for x in cleaned if x],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        save_cache()


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


def test_single_batch(input_file="data/letterboxd_filtered_post.jsonl", batch_size=5):
    lines = []
    with open(input_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < batch_size:
                lines.append(line)
            else:
                j = random.randint(0, i)
                if j < batch_size:
                    lines[j] = line

    batch = []
    for raw in lines:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue

        batch.append(
            {
                "title": data.get("title"),
                "year": data.get("year"),
                "reviews": data.get("review_texts", []),
                "key": make_key(data.get("title"), data.get("year")),
                "synopsis": data.get("synopsis"),
            }
        )

    print("=== ORIGINAL REVIEWS ===\n")
    for film in batch:
        print(f"{film['title']} ({film['year']})")
        for r in film["reviews"]:
            print("-", r)
        print("")

    print("=== IMPROVED REVIEWS ===\n")
    results = improve_reviews_batch(batch)
    # print(results)
    for film, improved in zip(batch, results, strict=False):
        print(f"{film['title']} ({film['year']})")
        for orig, new in zip(film["reviews"], improved, strict=False):
            match new:
                case "SKIP":
                    print("-[SKIP]", orig)
                case "DISCARD":
                    print("-[DISCARD]")
                case _:
                    print("-", new)
        print("")


if __name__ == "__main__":
    main(input_file="data/splits/part_1.jsonl", output_file="data/letterboxd_filtered_llm_part_1.jsonl")
    # test_random_films()
    # improve_reviews_batched("data/splits/part_1.jsonl", "data/letterboxd_filtered_llm_part_1.jsonl", 20)
    # test_single_batch()
