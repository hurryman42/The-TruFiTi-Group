from src.enums import TokenizerTypeEnum
from src.enums.types import SpecialTokensEnum
from src.tokenizer.bpe_tokenizer import BPETokenizer


def encode_char(texts: list[str], tokenizer) -> list[int]:
    text_joined = "\n".join(texts)
    return tokenizer.encode(text_joined)


def encode_hugging_face_bpe(texts: list[str], tokenizer, batch_size: int = 1000) -> list[int]:
    all_ids = []

    bos_id = tokenizer.token_to_id(SpecialTokensEnum.BOS)
    eos_id = tokenizer.token_to_id(SpecialTokensEnum.EOS)
    if eos_id is None or bos_id is None:
        raise ValueError("Special tokens were not trained!")

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded_batch = tokenizer.encode_batch(batch)
        for encoded in encoded_batch:
            all_ids.append(bos_id)
            all_ids.extend(encoded.ids)
            all_ids.append(eos_id)
    return all_ids


def encode_bpe(
    texts: list[str],
    tokenizer: BPETokenizer,
    batch_size: int = 1000,
) -> list[int]:
    all_ids: list[int] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded_batch = tokenizer.encode_batch(batch, add_special_tokens=True)

        for encoded in encoded_batch:
            all_ids.extend(encoded)

    return all_ids


def encode_texts(texts: list[str], tokenizer, tokenizer_type: TokenizerTypeEnum) -> list[int]:
    match tokenizer_type:
        case TokenizerTypeEnum.CHAR:
            return encode_char(texts, tokenizer)
        case TokenizerTypeEnum.BPE_HUGGING_FACE:
            return encode_hugging_face_bpe(texts, tokenizer)
        case TokenizerTypeEnum.BPE:
            return encode_bpe(texts, tokenizer)
        case _:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
