from pathlib import Path

from src.tokenizer.char_tokenizer import CharTokenizer
from src.utils.data_loader import read_file_only_reviews

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR.parent / "data" / "letterboxd_filtered_short_synopsis_film.jsonl"
SAVE_PATH = BASE_DIR.parent / "tokenizer" / "char_tokenizer.json"


def train_and_save():
    texts = read_file_only_reviews(INPUT_FILE)
    tokenizer = CharTokenizer().train(texts)
    tokenizer.save(str(SAVE_PATH))
    print(f"Tokenizer saved to {SAVE_PATH}")
    return tokenizer


def verify(tokenizer: CharTokenizer):
    test_text = "Hello World!"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    loaded = CharTokenizer.load(str(SAVE_PATH))
    encoded_loaded = loaded.encode(test_text)

    assert encoded == encoded_loaded, "Encoding mismatch!"
    assert decoded == test_text, "Decoding mismatch!"
    print("Verification passed!")


if __name__ == "__main__":
    tokenizer = train_and_save()
    verify(tokenizer)
