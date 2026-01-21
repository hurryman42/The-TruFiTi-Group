from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer

from src.config import get_tokenizer_path
from src.config.config import TokenizerConfig
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.tokenizer.char_tokenizer import CharTokenizer

from src.enums import TokenizerTypeEnum

type AnyTokenizer = CharTokenizer | HFTokenizer | BPETokenizer


def load_tokenizer(config: TokenizerConfig) -> AnyTokenizer:
    path = get_tokenizer_path(config.name)
    tokenizer: AnyTokenizer
    match config.type:
        case TokenizerTypeEnum.CHAR:
            tokenizer = load_char_tokenizer(path)
        case TokenizerTypeEnum.BPE_HUGGING_FACE:
            tokenizer = load_bpe_hugging_face_tokenizer(path)
        case TokenizerTypeEnum.BPE:
            tokenizer = load_bpe_custom_tokenizer(path)
        case _:
            raise ValueError(f"Unknown tokenizer type: {config.type}")
    return tokenizer


def load_char_tokenizer(path: Path) -> CharTokenizer:
    tokenizer = CharTokenizer.load(str(path))
    print(f"Loaded char tokenizer - vocab size: {tokenizer.get_vocab_size()}")
    return tokenizer


def load_bpe_hugging_face_tokenizer(path: Path) -> HFTokenizer:
    tokenizer = HFTokenizer.from_file(str(path))
    print(f"Loaded BPE hugging face tokenizer - vocab size: {tokenizer.get_vocab_size()}")
    return tokenizer


def load_bpe_custom_tokenizer(path: Path) -> BPETokenizer:
    tokenizer = BPETokenizer.load(str(path))
    print(f"Loaded custom BPE tokenizer - vocab size: {tokenizer.get_vocab_size()}")
    return tokenizer
