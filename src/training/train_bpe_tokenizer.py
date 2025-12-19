"""Training script for the custom BPE tokenizer."""

from pathlib import Path

from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.utils.data_loader import read_file_only_reviews

VOCAB_SIZE = 2000
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR.parent / "data" / "letterboxd_filtered.jsonl"
SAVE_PATH = BASE_DIR.parent / "tokenizer" / "bpe_custom_tokenizer.json"


def train_bpe_tokenizer(texts: list[str], vocab_size: int) -> BPETokenizer:
    tokenizer = BPETokenizer.train(
        texts,
        target_size=vocab_size,
        verbose=True,
    )
    print(f"Vocabulary size: {tokenizer.get_vocab_size}")
    print(f"Special tokens: BOS={tokenizer.bos_id}, EOS={tokenizer.eos_id}, PAD={tokenizer.pad_id}")
    return tokenizer


def train_and_save() -> BPETokenizer:
    texts = read_file_only_reviews(INPUT_FILE)
    print(f"Number of texts: {len(texts):,}".replace(",", "."))

    part_texts = texts[:10000]

    print(f"Training on texts: {len(part_texts):,}".replace(",", "."))

    tokenizer = train_bpe_tokenizer(part_texts, vocab_size=VOCAB_SIZE)

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(SAVE_PATH)
    print(f"Tokenizer saved to {SAVE_PATH}")

    return tokenizer


def verify(tokenizer: BPETokenizer) -> None:
    test_texts = [
        "Hello World!",
        "This is a great movie! 🎬",
        "Größe, naïve, café",
    ]

    loaded = BPETokenizer.load(SAVE_PATH)

    for text in test_texts:
        original_encoded = tokenizer.encode(text)
        loaded_encoded = loaded.encode(text)
        assert original_encoded == loaded_encoded, f"Encoding mismatch for: {text}"

        original_with_special = tokenizer.encode_with_special_tokens(text)
        loaded_with_special = loaded.encode_with_special_tokens(text)
        assert original_with_special == loaded_with_special, f"Special token mismatch for: {text}"

        decoded = loaded.decode(loaded_encoded)
        assert decoded == text, f"Decode roundtrip failed for: {text}"

    assert tokenizer.bos_id == loaded.bos_id
    assert tokenizer.eos_id == loaded.eos_id
    assert tokenizer.pad_id == loaded.pad_id

    print("Verification passed!")


if __name__ == "__main__":
    tokenizer = train_and_save()
    verify(tokenizer)
