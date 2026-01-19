import os
import json
from tqdm import tqdm

MIN_TEXT_LENGTH = 20  # minimum characters for plot or synopsis
IMDB_VOTE_THRESHOLD = 100  # if imdb_votes exists and < threshold → skip film


def normalize_text_field(value):
    if not value:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        string_elements = [str(v).strip() for v in value if isinstance(v, (str, int, float))]
        return " ".join(string_elements).strip()

    return str(value).strip()


def choose_best_description(data):
    synopsis = normalize_text_field(data.get("synopsis"))
    plot = normalize_text_field(data.get("plot"))

    if plot and len(plot) >= MIN_TEXT_LENGTH:
        if synopsis and len(synopsis) > len(plot):  # and plot.lower() in synopsis.lower()
            return synopsis
        return plot

    if synopsis and len(synopsis) >= MIN_TEXT_LENGTH:
        return synopsis

    return None


def passes_imdb_vote_filter(data):
    votes = data.get("imdb_votes")

    if votes is None:
        return True

    if isinstance(votes, int) and votes < IMDB_VOTE_THRESHOLD:
        return False

    return True


def main(input_path="data/letterboxd_filtered_omdb.jsonl", output_path="data/letterboxd_filtered_post.jsonl"):
    output_file = os.path.basename(output_path)

    total_films = 0
    total_reviews = 0
    filtered_films = 0
    filtered_reviews = 0
    skipped_count = 0

    with open(input_path, encoding="utf-8") as infile:
        total_lines = sum(1 for _ in infile)

    with open(input_path, encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for index, line in tqdm(enumerate(infile, 1), total=total_lines):
            try:
                data = json.loads(line)
                total_films += 1
                total_reviews += len(data.get("review_texts"))

                if not passes_imdb_vote_filter(data):
                    skipped_count += 1
                    continue

                best_desc = choose_best_description(data)
                if not best_desc:
                    skipped_count += 1
                    continue

                data["synopsis"] = best_desc

                outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

                filtered_films += 1
                filtered_reviews += len(data["review_texts"])
            except json.JSONDecodeError as e:
                print(f"Skipped invalid JSON at line {index}: {e}")
                continue

    print("\n---------- Post-filter complete! --- Summary: ----------")
    print(f"Films before filtering:   {total_films}")
    print(f"Reviews before filtering: {total_reviews}")
    print("------------------------------------------")
    print(f"Skipped films:            {skipped_count}")
    print("------------------------------------------")
    print(f"Films after filtering:    {filtered_films}")
    print(f"Reviews after filtering:  {filtered_reviews}")
    print("------------------------------------------")
    print(f"Saved to {output_file}")


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
    main()
    # split_jsonl("data/letterboxd_filtered_post.jsonl", 4, "data/splits")
    # merge_jsonl(["data/splits/part_1.jsonl", "data/splits/part_2.jsonl"], "data/merged.jsonl")
