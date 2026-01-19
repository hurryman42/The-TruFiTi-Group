import unicodedata
import argparse
import hashlib
import emoji
import json
import os
import re
from tqdm import tqdm
from lingua import Language, LanguageDetectorBuilder
from spellchecker import SpellChecker

MIN_REVIEW_WORDS = 10
MAX_REVIEW_WORDS = 111
MAX_EMOJIS = 0
MAX_WEIRD_CHARS = 10
MAX_REPETITION = 15
MAX_NON_LATIN_CHARS = 0
MISSPELLED_RATIO = 0.03

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

# review_adjuster = ReviewAdjuster()
spell = SpellChecker()


def count_non_latin_script_chars(text):
    count = 0
    for c in text:
        if c.isalpha():
            try:
                name = unicodedata.name(c)
                if "LATIN" not in name:
                    count += 1
            except ValueError:
                count += 1
    return count


def count_emojis(text):
    return sum(1 for c in text if c in emoji.EMOJI_DATA)


def misspelled_ratio(text):
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return 1.0
    misspelled = spell.unknown(words)
    return len(misspelled) / len(words)


def get_hash(text):
    return hashlib.md5(text.lower().strip().encode("utf-8")).hexdigest()


# lingua-py returns language detection in the form of a language object, like 'Language.ENGLISH'
def is_english(text: str, detector) -> bool:
    try:
        lang = detector.detect_language_of(text)
        return lang == Language.ENGLISH
    except Exception as e:
        print("[ERROR in is_english]", e)
        return False


def count_weird_chars(text):
    weird = 0
    for c in text:
        cat = unicodedata.category(c)
        # So = Symbol other (Braille)
        # Mn = Mark non spacing (Zalgo combining chars)
        # Cf = Format
        # Co = Private use
        if cat in ("So", "Mn", "Cf", "Co"):
            weird += 1
    return weird


def has_excessive_repetition(text, max_repeat):
    pattern = rf"(.)\1{{{max_repeat},}}"
    return bool(re.search(pattern, text))


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
    reviews = data.get("reviews", [])
    filtered_reviews = []

    for r in reviews:
        review_text = r.get("review_text", "")
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


def main():
    parser = argparse.ArgumentParser(
        description="Filter Letterboxd data per film", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input_file", help="Input JSONL file path")
    args = parser.parse_args()

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

    output_filename = "letterboxd_filtered_pre.jsonl"
    output_file = os.path.join(os.path.dirname(args.input_file), output_filename)

    skipped_count = 0
    processed_count = 0
    total_films = 0
    total_reviews = 0
    filtered_films = 0
    filtered_reviews = 0

    with open(args.input_file, encoding="utf-8") as infile:
        total_lines = sum(1 for _ in infile)

    with open(args.input_file, encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        seen_hashes = set()
        for index, line in tqdm(enumerate(infile, 1), total=total_lines):
            try:
                data = json.loads(line)
                total_films += 1
                total_reviews += len(data.get("reviews", []))
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

    print("\n---------- Pre-filter complete! --- Summary: ----------")
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


if __name__ == "__main__":
    main()
