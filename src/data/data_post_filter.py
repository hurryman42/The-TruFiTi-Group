import re
import json
from tqdm import tqdm
from datetime import datetime
from lingua import Language, LanguageDetectorBuilder
from spellchecker import SpellChecker

from src.data.data_pre_filter import (
    get_hash,
    count_emojis,
    count_weird_chars,
    has_excessive_repetition,
    count_non_latin_script_chars,
    is_english,
    misspelled_ratio,
)

MIN_REVIEW_WORDS = 5
MAX_REVIEW_WORDS = 170
MAX_EMOJIS = 0
MAX_WEIRD_CHARS = 10
MAX_REPETITION = 15
MAX_NON_LATIN_CHARS = 0
MISSPELLED_RATIO = 0.05

BAD_PATTERNS = re.compile(
    r"(this review may contain spoilers"  # "This review may contain spoilers. I can handle the truth."
    r"|english version below"  # "Deutsche Kritik oben. English Version below ..."
    r"|^starring:"  # "Starring: Jackie Chan, Chris Tucker, Tom Wilkinson"
    r"|^seen (?:at|via|on)"  # "Seen via Panic Fest 2023" or "Seen at the cinema"
    r"|^part of (?:my|the)"  # "Part of my Japanese New Wave Top 200"
    r"|challenge$"  # "All Disney Features and Shorts Challenge"
    r"|^review from"  # "Review from my VOD column 'This Week on Demand'"
    r"|watchlist$"  # "French Film Noir Watchlist"
    r"|^action! -)",  # "ACTION! - KILLER MIKE"
    flags=re.IGNORECASE | re.MULTILINE,
)

spell = SpellChecker()
CACHE_FILE = "data/review_cache.json"


def is_valid_review(text, detector):
    if not text or not text.strip():
        return False

    text_clean = text.strip()
    words = text.lower().strip().split()
    word_count = len(words)

    if word_count < MIN_REVIEW_WORDS:
        return False
    if word_count > MAX_REVIEW_WORDS:
        return False

    if BAD_PATTERNS.search(text_clean):
        return False

    if count_emojis(text_clean) > MAX_EMOJIS:
        return False

    if count_weird_chars(text_clean) > MAX_WEIRD_CHARS:
        return False

    if has_excessive_repetition(text_clean, MAX_REPETITION):
        return False

    if count_non_latin_script_chars(text_clean) > MAX_NON_LATIN_CHARS:
        return False

    if not is_english(text_clean, detector):
        return False

    if misspelled_ratio(text_clean) > MISSPELLED_RATIO:
        return False

    return True


def filter_per_film(data, seen_hashes, detector):
    reviews = data.get("review_texts", [])
    filtered_reviews = []

    for review_text in reviews:
        if not is_valid_review(review_text, detector):
            continue

        text_hash = get_hash(review_text)
        if text_hash in seen_hashes:
            continue
        seen_hashes.add(text_hash)

        filtered_reviews.append(review_text)

    if not filtered_reviews:
        return None

    return {
        "title": data.get("title"),
        "year": data.get("year"),
        "synopsis": data.get("synopsis"),
        "review_texts": filtered_reviews,
    }


def clean_cache(detector):
    try:
        with open(CACHE_FILE, encoding="utf-8") as f:
            cache = json.load(f)
    except (OSError, json.JSONDecodeError):
        print("No cache or invalid cache file. Skipping cache cleaning.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_path = f"{CACHE_FILE}.backup_{timestamp}.json"
    try:
        with open(backup_path, "w", encoding="utf-8") as backup:
            json.dump(cache, backup, ensure_ascii=False)
        print(f"Created backup: {backup_path}")
    except Exception as e:
        print(f"WARNING: Could not create backup! {e}")
        print("Aborting to avoid data loss.")
        return

    cleaned_cache = {}
    removed_count = 0
    seen_hashes = set()

    for film_key, reviews in tqdm(cache.items(), desc="Processing films..."):
        if not isinstance(reviews, list):
            removed_count += 1
            continue

        cleaned_reviews = []

        for r in reviews:
            if not r or not isinstance(r, str):
                removed_count += 1
                continue
            r_clean = r.strip()
            if not r_clean:
                removed_count += 1
                continue

            if not is_valid_review(r_clean, detector):
                removed_count += 1
                continue

            h = get_hash(r_clean)
            if h in seen_hashes:
                removed_count += 1
                continue
            seen_hashes.add(h)

            cleaned_reviews.append(r_clean)

        if cleaned_reviews:
            cleaned_cache[film_key] = cleaned_reviews
        else:
            removed_count += 1

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned_cache, f, ensure_ascii=False)

    print("--------------------------------------------------")
    print("Cache cleaned!")
    print(f"Removed {removed_count} invalid/duplicate reviews.")
    print(f"Remaining films in cache: {len(cleaned_cache)}")
    print("--------------------------------------------------")


def main(
    input_file="data/letterboxd_filtered_llm.jsonl",
    output_file="data/letterboxd_filtered_llm_filtered.jsonl",
    cache_cleaning=False,
):
    detector = (
        LanguageDetectorBuilder.from_languages(
            Language.ENGLISH,
            Language.SPANISH,
            Language.FRENCH,
            Language.GERMAN,
            Language.PORTUGUESE,
            Language.ITALIAN,
            Language.DUTCH,
            Language.SWEDISH,
            Language.POLISH,
            Language.RUSSIAN,
            Language.JAPANESE,
            Language.KOREAN,
            Language.CHINESE,
        )
        .with_preloaded_language_models()
        .build()
    )

    skipped_count = 0
    processed_count = 0
    total_films = 0
    total_reviews = 0
    filtered_films = 0
    filtered_reviews = 0

    with open(input_file, encoding="utf-8") as infile:
        total_lines = sum(1 for _ in infile)

    with open(input_file, encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        seen_hashes = set()
        for index, line in tqdm(enumerate(infile, 1), total=total_lines):
            try:
                data = json.loads(line)
                total_films += 1
                total_reviews += len(data.get("review_texts", []))
                filtered = filter_per_film(data, seen_hashes, detector)
                if filtered:
                    outfile.write(json.dumps(filtered, ensure_ascii=False) + "\n")
                    processed_count += 1
                    filtered_films += 1
                    filtered_reviews += len(filtered["review_texts"])
                else:
                    skipped_count += 1
            except json.JSONDecodeError as e:
                print(f"Skipped invalid JSON at line {index}: {e}")
                continue

    print("\n---------- Post-pre-filter complete! --- Summary: ----------")
    print(f"Processed films:          {processed_count}")
    print(f"Skipped films:            {skipped_count}")
    print("------------------------------------------")
    print(f"Films before filtering:   {total_films}")
    print(f"Reviews before filtering: {total_reviews}")
    print("------------------------------------------")
    print(f"Films after filtering:    {filtered_films}")
    print(f"Reviews after filtering:  {filtered_reviews}")
    print("------------------------------------------")
    print(f"Saved to {output_file}")
    print("------------------------------------------")

    if cache_cleaning:
        clean_cache(detector)


if __name__ == "__main__":
    main(
        input_file="data/letterboxd_filtered_llm.jsonl",
        output_file="data/letterboxd_filtered_llm_filtered.jsonl",
        cache_cleaning=False,
    )
