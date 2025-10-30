import json

input_file = "../data/letterboxd_full.jsonl"   # your original file
output_file = "../data/letterboxd_filtered.jsonl"  # the filtered file

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        data = json.loads(line)
        filtered = {
            "title": data.get("title"),
            "year": data.get("year"),
            "synopsis": data.get("synopsis"),
            #"reviews": data.get("reviews")
            "review_texts": [r.get("review_text", "") for r in data.get("reviews", [])]
        }
        outfile.write(json.dumps(filtered, ensure_ascii=False) + "\n")