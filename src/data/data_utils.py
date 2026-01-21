import re
import os
import json
import argparse


def normalize_title(title):
    title = title.lower().strip()
    title = re.sub(r"[^a-z0-9]+", " ", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def make_key(title, year):
    year_str = str(year) if year else "NOYEAR"
    return normalize_title(title) + "||" + year_str


def load_processed_keys(output_file):
    processed = set()
    if os.path.exists(output_file):
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                title = data.get("title")
                year = data.get("year")
                key = make_key(title, year)
                processed.add(key)
    return processed


def load_unprocessed_keys(input_file, keys_to_skip):
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

            if key not in keys_to_skip:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="store_true", help="Run split_jsonl")
    parser.add_argument("--merge", action="store_true", help="Run merge_jsonl")
    args = parser.parse_args()

    if args.split:
        split_jsonl("data/letterboxd_filtered_post.jsonl", 4, "data/splits")

    if args.merge:
        merge_jsonl(
            [
                "data/tomerge/letterboxd_filtered_llm_part_1.jsonl",
                "data/tomerge/letterboxd_filtered_llm_part_2.jsonl",
                "data/tomerge/letterboxd_filtered_llm_part_3.jsonl",
                "data/tomerge/letterboxd_filtered_llm_part_4.jsonl",
            ],
            "data/letterboxd_filtered_llm.jsonl",
        )
