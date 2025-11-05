import argparse
import json
import os
import sys

DEFAULT_MIN_SYNOPSIS_WORDS = 0

def has_sufficient_synopsis(synopsis, min_words):
    if not synopsis or not isinstance(synopsis, str):
        return False
    word_count = len(synopsis.split())
    return word_count >= min_words

def filter_per_film(data, min_synopsis_words):
    if min_synopsis_words > 0 and not has_sufficient_synopsis(data.get("synopsis"), min_synopsis_words):
        return None

    return {
        "title": data.get("title"),
        "year": data.get("year"),
        "synopsis": data.get("synopsis"),
        "review_texts": [r.get("review_text", "") for r in data.get("reviews", [])],
    }

def filter_per_review(data, min_synopsis_words): # each review has its own entry with its film metadata
    if min_synopsis_words > 0 and not has_sufficient_synopsis(data.get("synopsis"), min_synopsis_words):
        return None

    title = data.get("title")
    year = data.get("year")
    synopsis = data.get("synopsis")
    reviews = data.get("reviews", [])
    return [
        {
            "title": title,
            "year": year,
            "synopsis": synopsis,
            "review_text": r.get("review_text", "")
        }
        for r in reviews if r.get("review_text")
    ]

def main():
    parser = argparse.ArgumentParser(
        description="Filter Letterboxd data by film or review",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=["film", "review"],
            help="'film' for one entry per film, 'review' for one entry per review")
    parser.add_argument("input_file", help="Input JSONL file path")
    parser.add_argument("--min-synopsis-words", type=int, default=DEFAULT_MIN_SYNOPSIS_WORDS,
            help="Minimum number of words required in synopsis (default: 0, no filtering)")

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

    with (open(args.input_file, "r", encoding="utf-8") as infile,
          open(output_file, "w", encoding="utf-8") as outfile):
        for index, line in enumerate(infile, 1):
            try:
                data = json.loads(line)
                if args.mode == "film":
                    filtered = filter_per_film(data, args.min_synopsis_words)
                    if filtered:
                        outfile.write(json.dumps(filtered, ensure_ascii=False) + "\n")
                        processed_count += 1
                    else:
                        skipped_count += 1
                else:  # args.mode == "review"
                    review_entries = filter_per_review(data, args.min_synopsis_words)
                    if review_entries:
                        for review_entry in review_entries:
                            outfile.write(json.dumps(review_entry, ensure_ascii=False) + "\n")
                        processed_count += len(review_entries)
                    else:
                        skipped_count += 1
            except Exception as e:
                print(f"Skipped line {index} (Error: {e})")

    print(f"\nProcessing complete!")
    print(f"Processed: {processed_count} entries")
    print(f"Skipped due to insufficient synopsis: {skipped_count}")

if __name__ == "__main__":
    main()