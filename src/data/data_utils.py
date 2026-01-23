import re
import os
import json
import argparse

from src.data.data_pre_filter import get_hash


def normalize_title(title):
    title = title.lower().strip()
    title = re.sub(r"[^a-z0-9]+", " ", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def make_key(title, year):
    year_str = str(year) if year else "NOYEAR"
    return normalize_title(title) + "||" + year_str


def load_keys(file):
    keys = set()
    if os.path.exists(file):
        with open(file, encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                title = data.get("title")
                year = data.get("year")
                key = make_key(title, year)
                keys.add(key)
    return keys


def load_unprocessed_keys(input_file, processed_keys):
    unprocessed = []
    with open(input_file, encoding="utf-8") as infile:
        for line in infile:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            title = data.get("title")
            year = data.get("year")
            key = make_key(title, year)

            if key not in processed_keys:
                unprocessed.append(line)
    return unprocessed


def split_jsonl(input_path, n, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, encoding="utf-8") as infile:
        lines = infile.readlines()

    total = len(lines)
    part_size = (total + n - 1) // n

    paths = []
    for i in range(n):
        start = i * part_size
        end = min(start + part_size, total)
        part_lines = lines[start:end]

        out_path = os.path.join(output_dir, f"part_{i + 1}.jsonl")
        with open(out_path, "w", encoding="utf-8") as outfile:
            outfile.writelines(part_lines)

        paths.append(out_path)

    print(f"Split complete: {n} parts created in {output_dir}")
    return paths


def merge_jsonl(input_paths, output_path):
    with open(output_path, "w", encoding="utf-8") as outfile:
        for p in input_paths:
            with open(p, encoding="utf-8") as infile:
                for line in infile:
                    outfile.write(line)

    print(f"Merged {len(input_paths)} files into {output_path}")


def remove_none_reviews(input_file, output_file):
    with open(input_file, encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            reviews = data.get("review_texts", [])
            cleaned = []
            for r in reviews:
                if not r:
                    continue
                r = r.strip()
                cleaned.append(r)
            data["review_texts"] = cleaned

            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

    print("Cleaning complete.")


def merge_review_caches(cache_a_path, cache_b_path, output_path):
    def load_cache(path):
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"{path} is missing or empty. Using empty cache.")
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"{path} is corrupted. Using empty cache.")
            return {}

    print("Loading caches...")
    cache_a = load_cache(cache_a_path)
    cache_b = load_cache(cache_b_path)

    print("Merging...")
    merged = {}
    global_seen_hashes = set()

    def add_from_cache(source_cache):
        for film_key, reviews in source_cache.items():
            if not isinstance(reviews, list):
                continue

            if film_key not in merged:
                merged[film_key] = []

            for r in reviews:
                if not isinstance(r, str):
                    continue
                r_clean = r.strip()
                if not r_clean:
                    continue

                h = get_hash(r_clean)
                if h in global_seen_hashes:
                    continue

                global_seen_hashes.add(h)
                merged[film_key].append(r)

    add_from_cache(cache_a)
    add_from_cache(cache_b)

    print("Writing merged cache...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False)

    print("--------------------------------------------------")
    print("Review cache merge complete!")
    print(f"Films in output cache: {len(merged)}")
    total_reviews = sum(len(v) for v in merged.values())
    print(f"Total reviews in output: {total_reviews}")
    print("--------------------------------------------------")

    return merged


def split_json_dict(input_path, n, output_dir):
    import json
    import os

    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    keys = list(data.keys())
    total = len(keys)
    part_size = (total + n - 1) // n

    paths = []
    for i in range(n):
        start = i * part_size
        end = min(start + part_size, total)

        part_keys = keys[start:end]
        subset = {k: data[k] for k in part_keys}

        out_path = os.path.join(output_dir, f"part_{i + 1}.json")
        with open(out_path, "w", encoding="utf-8") as out:
            json.dump(subset, out, ensure_ascii=False)

        paths.append(out_path)

    return paths


def merge_json_dicts(input_paths, output_path):
    import json

    merged = {}
    for p in input_paths:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
            merged.update(d)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False)


def count_duplicate_films_in_cache(cache_path="data/review_cache.json"):
    import json

    try:
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    except Exception as e:
        print("Error loading cache:", e)
        return None

    seen = {}
    for key in cache.keys():
        seen[key] = seen.get(key, 0) + 1

    duplicates = {k: v for k, v in seen.items() if v > 1}

    print("Films appearing multiple times:", len(duplicates))
    for film, count in duplicates.items():
        print(f"{film}: {count} times")

    return len(duplicates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="store_true", help="Run split_jsonl")
    parser.add_argument("--merge", action="store_true", help="Run merge_jsonl")
    args = parser.parse_args()

    if args.split:
        split_jsonl("data/letterboxd_filtered_omdb.jsonl", 2, "data/splits")
    if args.merge:
        merge_jsonl(
            ["data/tomerge/part_1.jsonl", "data/tomerge/part_2.jsonl"],
            "data/merged.jsonl",
        )
    else:
        merge_review_caches("data/review_cache_old.json", "data/review_cache_part4.json", "data/review_cache.json")
        count_duplicate_films_in_cache()
