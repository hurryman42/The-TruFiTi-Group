import argparse
import fasttext
import hashlib
import json
import os
import re

DEFAULT_MIN_REVIEW_WORDS=15
DEFAULT_MIN_SYNOPSIS_WORDS=0
DEFAULT_MAX_EMOJIS=5
DEFAULT_MAX_NON_LATIN_CHARS=20

BAD_PATTERNS = re.compile(
    r"(this review may contain spoilers"  # "This review may contain spoilers. I can handle the truth."
    r"|english version below"  # "Deutsche Kritik oben. English Version below ..."
    r"|^starring:"  # "Starring: Jackie Chan, Chris Tucker, Tom Wilkinson"
    r"|^seen (?:at|via|on)"  # "Seen via Panic Fest 2023" or "Seen at the cinema"
    r"|^watched (?:at|via|with|on)"  # "Watched with the Golden Reel Gin Joint"
    r"|^part of (?:my|the)"  # "Part of my Japanese New Wave Top 200"
    r"|challenge$"  # "All Disney Features and Shorts Challenge"
    r"|^review from"  # "Review from my VOD column 'This Week on Demand'"
    r"|watchlist$"  # "French Film Noir Watchlist"
    r"|^action! -)"  # "ACTION! - KILLER MIKE"
)


def count_non_latin_chars(text):
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"()-\n")
    return sum(c not in allowed for c in text)


def count_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        "\U0001f300-\U0001f5ff"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f1e0-\U0001f1ff"  # flags
        "]+",
        flags=re.UNICODE,
    )
    return len(emoji_pattern.findall(text))


def has_sufficient_synopsis(synopsis, min_words):
    if not synopsis or not isinstance(synopsis, str):
        return False
    word_count = len(synopsis.split())
    return word_count >= min_words


def get_hash(text):
    return hashlib.md5(text.lower().strip().encode("utf-8")).hexdigest()


def is_english(text: str, model) -> bool:
    try:
        labels, probs = model.predict(text.replace("\n", " ").strip(), k=1)
        if not labels:
            return False
        return labels[0] == "__label__en" and float(probs[0]) >= 0.60
    except (ValueError, TypeError, IndexError):
        return False


def is_valid_review(
    text,
    max_non_latin_chars,
    model,
):
    if not text or not text.strip():
        return False

    text_lower = text.lower().strip()

    word_count = len(text_lower.split())
    if word_count < DEFAULT_MIN_REVIEW_WORDS:
        return False

    if BAD_PATTERNS.search(text_lower):
        return False

    if count_emojis(text) > DEFAULT_MAX_EMOJIS:
        return False

    if count_non_latin_chars(text) > max_non_latin_chars:
        return False

    if not is_english(text, model):
        return False

    return True


def filter_per_film(
    data,
    min_synopsis_words,
    max_non_latin_chars,
    seen_hashes,
    model,
):
    reviews = data.get("reviews", [])
    filtered_reviews = []

    for r in reviews:
        review_text = r.get("review_text", "")
        if not is_valid_review(review_text, max_non_latin_chars, model):
            continue

        text_hash = get_hash(review_text)
        if text_hash in seen_hashes:
            continue
        seen_hashes.add(text_hash)

        filtered_reviews.append(review_text)

    if not filtered_reviews:
        return None

    if min_synopsis_words > 0 and not has_sufficient_synopsis(data.get("synopsis"), min_synopsis_words):
        return None

    return {
        "title": data.get("title"),
        "year": data.get("year"),
        "synopsis": data.get("synopsis"),
        "review_texts": filtered_reviews,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Filter Letterboxd data by film or review", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input_file", help="Input JSONL file path")
    parser.add_argument(
        "--min-synopsis-words",
        type=int,
        default=DEFAULT_MIN_SYNOPSIS_WORDS,
        help=f"Minimum number of words required in synopsis (default: {DEFAULT_MIN_SYNOPSIS_WORDS}, no filtering)",
    )
    parser.add_argument(
        "--max-non-latin-chars",
        type=int,
        default=DEFAULT_MAX_NON_LATIN_CHARS,
        help=f"Maximum number of non-Latin characters allowed per review (default: {DEFAULT_MAX_NON_LATIN_CHARS})",
    )
    args = parser.parse_args()

    model = fasttext.load_model("data/lid.176.ftz")

    output_filename = f"letterboxd_filtered.jsonl"
    if args.min_synopsis_words > 0:
        output_filename = f"letterboxd_filtered_short_synopsis.jsonl"
    output_file = os.path.join(os.path.dirname(args.input_file), output_filename)

    skipped_count = 0
    processed_count = 0
    total_films = 0
    total_reviews = 0
    filtered_films = 0
    filtered_reviews = 0

    with open(args.input_file, encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        seen_hashes = set()
        for index, line in enumerate(infile, 1):
            try:
                data = json.loads(line)
                total_films += 1
                total_reviews += len(data.get("reviews", []))
                filtered = filter_per_film(
                    data, args.min_synopsis_words, args.max_non_latin_chars, seen_hashes, model
                )
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

    print("\n---------- Processing complete! --- Summary: ----------")
    print(f"Processed: {processed_count} entries")
    print(f"Skipped entries: {skipped_count}")
    print(f"Films before filtering: {total_films}")
    print(f"Reviews before filtering: {total_reviews}")
    print(f"Films after filtering: {filtered_films}")
    print(f"Reviews after filtering: {filtered_reviews}")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
