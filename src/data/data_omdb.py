import os
import re
import json
import time
import random
import requests
from tqdm import tqdm

from src.data.data_utils import normalize_title, make_key, load_keys, load_unprocessed_keys

OMDB_KEY = os.environ.get("OMDB_API_KEY")  # export OMDB_API_KEY=xxx

CACHE_FILE = "data/plot_cache.json"
PLOT_CACHE = {}  # in‑memory cache { (title_lower, year): plot }

REQUEST_LIMIT = 100000
omdb_requests_made = 0


def load_cache():
    global PLOT_CACHE
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, encoding="utf-8") as f:
                PLOT_CACHE = json.load(f)
        except (OSError, json.JSONDecodeError):
            PLOT_CACHE = {}


def save_cache():
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(PLOT_CACHE, f, ensure_ascii=False)


def count_unprocessed(input_file, processed_keys):
    count = 0
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
                count += 1
    return count


def fix_invalid_json_escapes(text):
    # replace backslashes not followed by a valid JSON escape char like " \ / b f n r t u
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)


def extract_omdb_fields(omdb_data):
    if not omdb_data:
        return {}

    fields = ["imdbID", "runtime", "metascore", "imdbRating", "imdbVotes", "type"]
    out = {}

    for f in fields:
        val = omdb_data.get(f)
        if val and val != "N/A":
            out[f.lower()] = val

    return out


def fetch_url(url, params=None, retries=3, delay=0.5):
    session = getattr(fetch_url, "_session", None)
    if session is None:
        session = requests.Session()
        fetch_url._session = session

    for _ in range(retries):
        try:
            r = session.get(url, params=params, timeout=10)
            return r
        except requests.exceptions.RequestException as e:
            print(f"[REQUEST ERROR]: {e}")
            time.sleep(delay)
    return None


def fetch_omdb_info(title, year):
    global omdb_requests_made, REQUEST_LIMIT
    if omdb_requests_made >= REQUEST_LIMIT:
        return None

    title = normalize_title(title)

    params = {
        "t": title,
        "plot": "full",
        "apikey": OMDB_KEY,
    }
    if year:
        params["y"] = year

    r = fetch_url("http://www.omdbapi.com/", params)
    omdb_requests_made += 1

    if r is None:
        print(f"No HTTP response for '{title}' ({year})")
        return None

    # TODO: add "s" requests for failed attempts

    try:
        data = r.json()
    except json.JSONDecodeError:
        raw = r.text
        raw_fixed = fix_invalid_json_escapes(raw)
        try:
            data = json.loads(raw_fixed)
        except json.JSONDecodeError as e:
            print(f"JSON decoding error for title '{title}': {e}")
            print(f"Status: {r.status_code}")
            print("Raw:", r.text[:500])
            if "limit" in r.text.lower():
                REQUEST_LIMIT = omdb_requests_made
                print("\nOMDb REQUEST LIMIT REACHED (HTML)\n")
            return None

    if data.get("Response") == "False":
        err = data.get("Error", "")
        if "Movie not found!" in err:
            return None
        print(f"[OMDb ERROR] {title} ({year}): {err}")
        if "limit" in err.lower():
            REQUEST_LIMIT = omdb_requests_made
            print("\nOMDb REQUEST LIMIT REACHED (JSON)\n")
        return None

    return data


def get_plot_per_film(data, check_cache=True):
    title = data.get("title")
    year = data.get("year")
    if not title:
        return None, {}
    key = make_key(title, year)

    if check_cache and key in PLOT_CACHE:
        cached = PLOT_CACHE[key]
        if isinstance(cached, (tuple, list)) and len(cached) == 2:
            return cached[0], cached[1]
        else:
            return cached, {}

    omdb = fetch_omdb_info(title, year) if year else fetch_omdb_info(title, None)
    if not omdb:
        PLOT_CACHE[key] = (None, {})
        return None, {}

    omdb_plot = omdb.get("Plot")
    if omdb_plot == "N/A":
        omdb_plot = None
    omdb_meta = extract_omdb_fields(omdb)

    if omdb_plot:
        PLOT_CACHE[key] = (omdb_plot, omdb_meta)
        return omdb_plot, omdb_meta

    PLOT_CACHE[key] = (None, omdb_meta)
    return None, omdb_meta


def main(input_file, output_file):
    load_cache()
    processed_keys = load_keys(output_file)
    print(f"Films already in output: {len(processed_keys)}")
    cache_keys = set(PLOT_CACHE.keys())
    print(f"Total cached films: {len(cache_keys)}")
    input_keys = load_keys(input_file)
    cached_overlap = input_keys & cache_keys
    print(f"Films in input AND cache: {len(cached_overlap)}")
    print(f"Films remaining to truly OMDb-process: {len(input_keys - cache_keys)}")

    unprocessed = load_unprocessed_keys(input_file, processed_keys)
    print(f"Films remaining to process: {len(unprocessed)}")
    print("--------------------------------------------------")

    skipped_count = 0
    processed_count = 0
    total_processed_this_run = 0

    with open(output_file, "a", encoding="utf-8") as outfile:
        for line in tqdm(unprocessed, desc="Processing films"):
            if omdb_requests_made >= REQUEST_LIMIT:
                print(f"Daily OMDb limit reached ({REQUEST_LIMIT}). Stopping early.")
                break

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            plot, meta = get_plot_per_film(data, check_cache=True)

            if plot:
                data["plot"] = plot
            else:
                data["plot"] = None
                skipped_count += 1

            for k, v in meta.items():
                if k not in data:
                    data[k] = v

            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

            processed_count += 1
            total_processed_this_run += 1

            if total_processed_this_run % 1000 == 0:
                save_cache()

        save_cache()

    print(f"Processed films:         {processed_count}")
    print(f"Films without OMDb plot: {skipped_count}")
    print(f"Cache size:              {len(PLOT_CACHE)}")


def retry_missing_plots(output_file):
    print("Retrying entries with plot=None...")
    load_cache()

    total_retries = 0
    with open(output_file, encoding="utf-8") as infile:
        for line in infile:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if data.get("plot") is None:
                total_retries += 1

    print(f"Found {total_retries} entries to retry.")

    if total_retries == 0:
        print("Nothing to retry.")
        return

    updated_lines = []
    success_count = 0
    skipped_count = 0

    with open(output_file, encoding="utf-8") as infile:
        for line in tqdm(infile, total=total_retries, desc="Retrying plots"):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                updated_lines.append(line.strip())
                continue

            if data.get("plot") is None:
                plot, meta = get_plot_per_film(data, check_cache=False)
                if plot:
                    data["plot"] = plot
                    success_count += 1
                else:
                    data["plot"] = None
                    skipped_count += 1

                for k, v in meta.items():
                    if k not in data:
                        data[k] = v

            updated_lines.append(json.dumps(data, ensure_ascii=False))

    with open(output_file, "w", encoding="utf-8") as outfile:
        for line in updated_lines:
            outfile.write(line + "\n")

    save_cache()

    print(f"Retried entries:  {total_retries}")
    print(f"Found a plot for  {success_count} film.")
    print(f"Still no plot for {skipped_count} films.")
    print(f"Cache size:       {len(PLOT_CACHE)}")


def test_random_films(input_file, n=5):
    print(f"Used API Key: {OMDB_KEY}")
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

        plot, meta = get_plot_per_film(data, check_cache=False)

        if plot:
            print(f"--- Plot from OMDb:\n{plot}\n")
            if synopsis and (plot.endswith("...") or synopsis.lower().startswith(plot.lower())):
                print("--- OMDb has the same as dataset or is truncated! ---\n")
        else:
            print("--- No plot found. ---\n")

        print(f"Metadata from OMDb: {meta}\n")

    save_cache()


if __name__ == "__main__":
    # delete data/letterboxd_filtered_omdb.jsonl beforehand if you want it fresh
    main(input_file="data/letterboxd_filtered_pre.jsonl", output_file="data/letterboxd_filtered_omdb.jsonl")

    # main(input_file="data/letterboxd_full.jsonl", output_file="data/letterboxd_very_full.jsonl")
    # retry_missing_plots("data/letterboxd_filtered_omdb.jsonl")
    # test_random_films("data/letterboxd_full.jsonl", 2)
