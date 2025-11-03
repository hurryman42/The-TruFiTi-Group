import sys
import os
import json

def filter_per_film(data): # each film has multiple reviews
    return {
        "title": data.get("title"),
        "year": data.get("year"),
        "synopsis": data.get("synopsis"),
        "review_texts": [r.get("review_text", "") for r in data.get("reviews", [])],
    }

def filter_per_review(data): # each review has its own entry with its film metadata
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
    if len(sys.argv) < 3:
        print("usage: python data_filter.py <mode> <input_file>")
        print("  mode = 'film' for one entry per film")
        print("  mode = 'review' for one entry per review")
        sys.exit(1)

    mode = sys.argv[1].lower()
    if mode not in {"film", "review"}:
        print("Error: mode must be 'film' or 'review'")
        sys.exit(1)
    input_file = sys.argv[2]

    output_filename = f"letterboxd_filtered_{mode}.jsonl"
    output_file = os.path.join(os.path.dirname(input_file), output_filename)

    with (open(input_file, "r", encoding="utf-8") as infile,
          open(output_file, "w", encoding="utf-8") as outfile):
        for index, line in enumerate(infile, 1):
            try:
                data = json.loads(line)
                if mode == "film":
                    filtered = filter_per_film(data)
                    outfile.write(json.dumps(filtered, ensure_ascii=False) + "\n")
                else:  # mode == "review"
                    for review_entry in filter_per_review(data):
                        outfile.write(json.dumps(review_entry, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Skipped line {index} (Error: {e})")

if __name__ == "__main__":
    main()