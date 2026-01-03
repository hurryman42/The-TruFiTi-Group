"""Training script for the BPE tokenizer from Hugging Face."""

import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from src.enums.types import SpecialTokensEnum
from src.utils.data_loader import read_file_only_reviews, read_file_synopsis_review_pairs

# TODO choose tokenizer size in config file
VOCAB_SIZE = 4000
BASE_DIR = Path(__file__).resolve().parent.parent.parent


def train_bpe_tokenizer(texts: list[str], vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[token.value for token in SpecialTokensEnum],
        show_progress=True,
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

    return tokenizer


def train_and_save(input_path: Path, output_path: Path, level: int) -> Tokenizer:
    print(f"Level: {level}")
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
    tokenizer.save(str(output_path))
    print(f"Tokenizer saved to {output_path}")

    return tokenizer


def verify(tokenizer: Tokenizer, output_path: Path) -> None:
    test_text = "Hello World!"
    encoded = tokenizer.encode(test_text)

    loaded = Tokenizer.from_file(str(output_path))
    encoded_loaded = loaded.encode(test_text)

    assert encoded.ids == encoded_loaded.ids, "Encoding mismatch!"
    print("Verification passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--l", "--level", type=str, required=True, help="Level")
    args, _ = parser.parse_known_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = BASE_DIR / "data" / dataset_path

    dataset_name = dataset_path.stem
    save_path = BASE_DIR / "tokenizer" / f"bpe_hf_L{args.l}_{dataset_name}.json"

    if dataset_path.suffix != ".jsonl":
        raise ValueError("Dataset must be a .jsonl file")

    tokenizer = train_and_save(dataset_path, save_path, int(args.l))
    verify(tokenizer, save_path)
