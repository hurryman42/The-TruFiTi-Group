from pathlib import Path

from src.tokenizer.bpe_tokenizer import BPETokenizer

from src.enums.types import SpecialTokensEnum
from src.utils.data_loader import read_file_only_reviews

# TODO choose tokenizer size in config file
VOCAB_SIZE = 4000
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR.parent / "data" / "letterboxd_filtered.jsonl"
SAVE_PATH = BASE_DIR.parent / "tokenizer" / f"bpe_tokenizer_{VOCAB_SIZE}.json"


def train_bpe_tokenizer(texts: list[str], vocab_size: int) -> BPETokenizer:
    tokenizer = BPETokenizer.train(texts, vocab_size=vocab_size)

    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

    return tokenizer


def train_and_save() -> BPETokenizer:
    texts = read_file_only_reviews(INPUT_FILE)
    print(f"Number of texts: {len(texts):,}".replace(",", "."))

    tokenizer = train_bpe_tokenizer(texts, vocab_size=VOCAB_SIZE)

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(SAVE_PATH))
    print(f"Tokenizer saved to {SAVE_PATH}")

    return tokenizer


def verify(tokenizer: BPETokenizer):
    test_text = "Hello World!"
    encoded = tokenizer.encode(test_text)

    loaded = BPETokenizer.load(str(SAVE_PATH))
    encoded_loaded = loaded.encode(test_text)

    assert encoded == encoded_loaded, "Encoding mismatch!"
    print("Verification passed!")


if __name__ == "__main__":
    tokenizer = train_and_save()
    verify(tokenizer)
