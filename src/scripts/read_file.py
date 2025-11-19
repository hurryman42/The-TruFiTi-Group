import json


def read_file(file_path) -> list[str]:
    texts = []
    with open(file_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON in line {i}: {e}")
                continue

            if data.get("review_texts"):
                texts.extend(data["review_texts"])
    return texts
