import argparse
from pathlib import Path

from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.utils.data_loader import read_file_only_reviews, read_file_synopsis_review_pairs

# TODO choose tokenizer size in config file
VOCAB_SIZE = 2000
BASE_DIR = Path(__file__).resolve().parent.parent.parent


def train_bpe_tokenizer(texts: list[str], vocab_size: int) -> BPETokenizer:
    tokenizer = BPETokenizer.train(
        texts,
        target_size=vocab_size,
        verbose=True,
    )
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(
        f"Special tokens:"
        f"BOS={tokenizer.bos_id},"
        f"EOS={tokenizer.eos_id},"
        f"PAD={tokenizer.pad_id},"
        f"SYN={tokenizer.syn_id},"
        f"REV={tokenizer.rev_id}"
    )
    return tokenizer


# TODO: add progress bar
def train_and_save(input_path: Path, output_path: Path, level: int) -> BPETokenizer:
    match level:
        case 1:
            texts = read_file_only_reviews(input_path)
        case 2:
            texts = read_file_synopsis_review_pairs(input_path)
        case _:
            raise ValueError(f"Invalid level input: {level}")
    print(f"Number of texts: {len(texts):,}".replace(",", "."))

    tokenizer = train_bpe_tokenizer(texts, vocab_size=VOCAB_SIZE)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_path)
    print(f"Tokenizer saved to {output_path}")

    return tokenizer


def verify(tokenizer: BPETokenizer, output_path: Path) -> None:
    test_texts = [
        "Hello World!",
        "This is a great movie! 🎬",
        "Größe, naïve, café",
    ]

    loaded = BPETokenizer.load(output_path)

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
    assert tokenizer.syn_id == loaded.syn_id
    assert tokenizer.rev_id == loaded.rev_id

    print("Verification passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--l", type=str, required=True, help="Level")
    args, _ = parser.parse_known_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = BASE_DIR / "data" / dataset_path

    dataset_name = dataset_path.stem
    save_path = BASE_DIR / "tokenizer" / f"bpe_custom_L{args.l}_{dataset_name}.json"

    if dataset_path.suffix != ".jsonl":
        raise ValueError("Dataset must be a .jsonl file")

    tokenizer = train_and_save(dataset_path, save_path, int(args.l))
    verify(tokenizer, save_path)
