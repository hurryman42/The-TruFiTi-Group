import os
import re
import json
import time
import random
import unicodedata

import requests
from tqdm import tqdm

OMDB_KEY = os.environ.get("OMDB_API_KEY")  # export OMDB_API_KEY=xxx

CACHE_FILE = "data/plot_cache.json"
plot_cache = {}  # in‑memory cache { (title_lower, year): plot }

REQUEST_LIMIT = 90  # TODO: change 90000 when paid key, ALSO: delete cache from free key (contains only short plots)
omdb_requests_made = 0


def load_cache():
    global plot_cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, encoding="utf-8") as f:
                plot_cache = json.load(f)
        except (OSError, json.JSONDecodeError):
            plot_cache = {}


def save_cache():
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(plot_cache, f, ensure_ascii=False)


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


def prepare_omdb_title(title):
    # remove accents, weird Unicode, normalize whitespace
    title = unicodedata.normalize("NFKD", title)
    title = "".join(c for c in title if not unicodedata.combining(c))
    title = re.sub(r"\s+", " ", title).strip()
    return title


def fetch_url(url, params=None, retries=3, delay=0.5):
    for _ in range(retries):
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                return r
        except requests.exceptions.RequestException:
            pass
        time.sleep(delay)
    return None


def fetch_omdb_info(title, year):
    global omdb_requests_made
    if omdb_requests_made >= REQUEST_LIMIT:
        return None

    title = prepare_omdb_title(title)

    params = {
        "t": title,
        "plot": "full",
        "apikey": OMDB_KEY,
    }
    if year:
        params["y"] = year

    r = fetch_url("http://www.omdbapi.com/", params)
    omdb_requests_made += 1
    if not r:
        return None

    data = r.json()
    if data.get("Response") == "True":
        return data

    return None


def extract_omdb_fields(omdb_data):
    if not omdb_data:
        return {}

    fields = ["imdbID", "Runtime", "Metascore", "imdbRating", "imdbVotes", "Type"]
    out = {}

    for f in fields:
        val = omdb_data.get(f)
        if val and val != "N/A":
            out[f.lower()] = val

    return out


def get_plot_per_film(data, debug):
    title = data.get("title")
    year = data.get("year")
    synopsis = data.get("synopsis")

    if not title:
        return None, {}

    key = make_key(title, year)

    if key in plot_cache:
        cached = plot_cache[key]
        if isinstance(cached, tuple) and len(cached) == 2:
            return cached
        else:
            return cached, {}

    if year:
        omdb = fetch_omdb_info(title, year)
    else:
        omdb = fetch_omdb_info(title, None)

    if not omdb:
        plot_cache[key] = (None, {})
        return None, {}

    omdb_plot = omdb.get("Plot")
    if omdb_plot == "N/A":
        omdb_plot = None
    omdb_meta = extract_omdb_fields(omdb)

    if omdb_plot:
        if debug:
            print(f"--- Plot from OMDb:\n{omdb_plot}\n")
        if synopsis:
            if omdb_plot.endswith("...") or synopsis.lower().startswith(omdb_plot.lower()):
                if debug:
                    print("--- OMDb has the same as dataset or is truncated! ---\n")
                plot_cache[key] = (synopsis, omdb_meta)
                return synopsis, omdb_meta

        plot_cache[key] = (omdb_plot, omdb_meta)
        return omdb_plot, omdb_meta

    plot_cache[key] = (None, omdb_meta)
    return None, omdb_meta


def main():
    load_cache()

    input_file = "data/letterboxd_full.jsonl"
    output_file = "data/letterboxd_very_full.jsonl"

    processed_keys = load_processed_keys(output_file)
    print(f"Already processed: {len(processed_keys)} films")

    skipped = 0
    total = 0

    with open(input_file, encoding="utf-8") as infile, open(output_file, "a", encoding="utf-8") as outfile:
        for _index, line in tqdm(enumerate(infile, 1), desc="Processing films"):
            if omdb_requests_made >= REQUEST_LIMIT:
                print(f"Daily OMDb limit reached ({REQUEST_LIMIT}). Stopping early.")
                break
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1

            title = data.get("title")
            year = data.get("year")
            key = make_key(title, year)
            if key in processed_keys:
                continue

            plot, meta = get_plot_per_film(data, False)

            if plot:
                data["plot"] = plot
            else:
                data["plot"] = None
                skipped += 1

            for k, v in meta.items():
                if k not in data:
                    data[k] = v

            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
            processed_keys.add(key)

        save_cache()

    print(f"Processed {total} films.")
    print(f"Could not find plot for {skipped} films.")
    print("Cache size:", len(plot_cache))


def test_random_films(n=5, input_file="data/letterboxd_full.jsonl"):
    load_cache()

    lines = []
    with open(input_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < n:
                lines.append(line)
            else:
                j = random.randint(0, i)
                if j < n:
                    lines[j] = line

    sample = random.sample(lines, min(n, len(lines)))

    for raw in sample:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON when testing, skipping. {e}")
            continue

        print("\n===================================================")
        print(f"{data.get('title')} ({data.get('year')})\n")

        synopsis = data.get("synopsis")
        print(synopsis if synopsis else "(None)")
        print("\n")

        plot, meta = get_plot_per_film(data, True)

        if plot:
            print(f"--- Final plot:\n{plot}\n")
        else:
            print("--- No plot found. ---\n")

        print(f"Metadata from OMDb: {meta}\n")

    save_cache()


if __name__ == "__main__":
    # main()
    test_random_films(5)
