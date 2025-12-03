from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from src.enums.types import SpecialTokensEnum
from src.utils.data_loader import read_file_only_reviews

# TODO choose tokenizer size in config file
VOCAB_SIZE = 4000
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR.parent / "data" / "letterboxd_filtered_short_synopsis_film.jsonl"
SAVE_PATH = BASE_DIR.parent / "tokenizer" / f"bpe_hugging_face_tokenizer_{VOCAB_SIZE}.json"


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


def train_and_save() -> Tokenizer:
    texts = read_file_only_reviews(INPUT_FILE)
    print(f"Number of texts: {len(texts):,}".replace(",", "."))

    tokenizer = train_bpe_tokenizer(texts, vocab_size=VOCAB_SIZE)

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(SAVE_PATH))
    print(f"Tokenizer saved to {SAVE_PATH}")

    return tokenizer


def verify(tokenizer: Tokenizer):
    test_text = "Hello World!"
    encoded = tokenizer.encode(test_text)

    loaded = Tokenizer.from_file(str(SAVE_PATH))
    encoded_loaded = loaded.encode(test_text)

    assert encoded.ids == encoded_loaded.ids, "Encoding mismatch!"
    print("Verification passed!")


if __name__ == "__main__":
    tokenizer = train_and_save()
    verify(tokenizer)
