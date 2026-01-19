import re
import os
import json


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
