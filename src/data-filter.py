import sys
import os
import json

if len(sys.argv) < 2:
    print("usage: python data-filter.py <input_file>")
    sys.exit(1)

#input_file = "../data/letterboxd_full.jsonl"   # hardcoded
#output_file = "../data/letterboxd_filtered.jsonl"  # hardcoded
input_file = sys.argv[1]
output_file = os.path.join(os.path.dirname(input_file), "letterboxd_filtered.jsonl")

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        data = json.loads(line)
        filtered = {
            "title": data.get("title"),
            "year": data.get("year"),
            "synopsis": data.get("synopsis"),
            "review_texts": [r.get("review_text", "") for r in data.get("reviews", [])]
        }
        outfile.write(json.dumps(filtered, ensure_ascii=False) + "\n")