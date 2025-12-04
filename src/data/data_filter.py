import argparse
import hashlib
import json
import os
import re
import sys

DEFAULT_MIN_SYNOPSIS_WORDS = 0
DEFAULT_MAX_NON_LATIN_CHARS = 20
DEFAULT_MAX_EMOJIS = 5
DEFAULT_MIN_REVIEW_WORDS = 4

BAD_PATTERNS = [
    re.compile(r"this review may contain spoilers"),  # "This review may contain spoilers. I can handle the truth."
    re.compile(r"english version below"),  # "Deutsche Kritik oben. English Version below ..."
    re.compile(r"^starring:"),  # "Starring: Jackie Chan, Chris Tucker, Tom Wilkinson"
    re.compile(r"^seen (at|via|on)"),  # "Seen via Panic Fest 2023" or "Seen at the cinema"
    re.compile(r"^watched (at|via|with|on)"),  # "Watched with the Golden Reel Gin Joint"
    re.compile(r"^part of (my|the)"),  # "Part of my Japanese New Wave Top 200"
    re.compile(r"challenge$"),  # "All Disney Features and Shorts Challenge"
    re.compile(r"^review from"),  # "Review from my VOD column 'This Week on Demand'"
    re.compile(r"watchlist$"),  # "French Film Noir Watchlist"
    re.compile(r"^action! -"),  # "ACTION! - KILLER MIIKE"
]


def count_non_latin_chars(text):
    if not text:
        return 0
    pattern = re.compile(
        r"[^A-Za-z0-9\s.,!?;:'\"()\-\n]"
    )  # any character not in Latin letters, digits, or basic punctuation
    return len(pattern.findall(text))


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


# def is_english(text: str) -> bool:


def is_valid_review(
    text,
    max_non_latin_chars,
    max_emojis,
    min_words=DEFAULT_MIN_REVIEW_WORDS,
):
    if not text or not text.strip():
        return False

    text_lower = text.lower().strip()

    word_count = len(text_lower.split())
    if word_count < min_words:
        return False

    for pattern in BAD_PATTERNS:
        if pattern.search(text_lower):
            return False

    if count_emojis(text) > max_emojis:
        return False

    if count_non_latin_chars(text) > max_non_latin_chars:
        return False

    # if not is_english(text):
    #     return False

    return True


def filter_per_film(
    data,
    min_synopsis_words,
    max_non_latin_chars,
    max_emojis,
    seen_hashes,
):
    reviews = data.get("reviews", [])
    filtered_reviews = []

    for r in reviews:
        review_text = r.get("review_text", "")
        if not is_valid_review(review_text, max_non_latin_chars, max_emojis):
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


def filter_per_review(
    data,
    min_synopsis_words,
    max_non_latin_chars,
    max_emojis,
):  # each review has its own entry with its film metadata
    if min_synopsis_words > 0 and not has_sufficient_synopsis(data.get("synopsis"), min_synopsis_words):
        return None

    title = data.get("title")
    year = data.get("year")
    synopsis = data.get("synopsis")
    reviews = data.get("reviews", [])
    filtered_reviews = []
    for r in reviews:
        review_text = r.get("review_text", "")
        if not is_valid_review(review_text):
            continue
        if count_emojis(review_text) <= max_emojis and count_non_latin_chars(review_text) <= max_non_latin_chars:
            filtered_reviews.append({"title": title, "year": year, "synopsis": synopsis, "review_text": review_text})
    return filtered_reviews if filtered_reviews else None


def main():
    parser = argparse.ArgumentParser(
        description="Filter Letterboxd data by film or review", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "mode", choices=["film", "review"], help="'film' for one entry per film, 'review' for one entry per review"
    )
    parser.add_argument("input_file", help="Input JSONL file path")
    parser.add_argument(
        "--min-synopsis-words",
        type=int,
        default=DEFAULT_MIN_SYNOPSIS_WORDS,
        help="Minimum number of words required in synopsis (default: 0, no filtering)",
    )
    parser.add_argument(
        "--max-emojis",
        type=int,
        default=DEFAULT_MAX_EMOJIS,
        help=f"Maximum number of emojis allowed per review (default: {DEFAULT_MAX_EMOJIS})",
    )
    parser.add_argument(
        "--max-non-latin-chars",
        type=int,
        default=DEFAULT_MAX_NON_LATIN_CHARS,
        help=f"Maximum number of non-Latin characters allowed per review (default: {DEFAULT_MAX_NON_LATIN_CHARS})",
    )

    args = parser.parse_args()
    if len(sys.argv) < 3:
        print("usage: python data_filter.py <mode> <input_file>")
        print("  mode = 'film' for one entry per film")
        print("  mode = 'review' for one entry per review")
        sys.exit(1)

    output_filename = f"letterboxd_filtered_{args.mode}.jsonl"
    if args.min_synopsis_words > 0:
        output_filename = f"letterboxd_filtered_short_synopsis_{args.mode}.jsonl"
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

                if args.mode == "film":
                    filtered = filter_per_film(
                        data, args.min_synopsis_words, args.max_non_latin_chars, args.max_emojis, seen_hashes
                    )
                    if filtered:
                        outfile.write(json.dumps(filtered, ensure_ascii=False) + "\n")
                        processed_count += 1
                        filtered_films += 1
                        filtered_reviews += len(filtered["review_texts"])
                    else:
                        skipped_count += 1
                else:  # args.mode == "review"
                    review_entries = filter_per_review(
                        data, args.min_synopsis_words, args.max_non_latin_chars, args.max_emojis
                    )
                    if review_entries:
                        for review_entry in review_entries:
                            outfile.write(json.dumps(review_entry, ensure_ascii=False) + "\n")
                        processed_count += len(review_entries)
                        filtered_films += 1
                        filtered_reviews += len(review_entries)
                    else:
                        skipped_count += 1
            except Exception as e:
                print(f"Skipped line {index} (Error: {e})")

    print("\nProcessing complete!")
    print(f"Processed: {processed_count} entries")
    print(f"Skipped entries: {skipped_count}")

    print("\n---------- Summary ----------")
    print(f"Films before filtering: {total_films}")
    print(f"Reviews before filtering: {total_reviews}")
    print(f"Films after filtering: {filtered_films}")
    print(f"Reviews after filtering: {filtered_reviews}")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
