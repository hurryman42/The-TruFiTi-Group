from src.enums import TokenizerTypeEnum


def encode_char(texts: list[str], tokenizer) -> list[int]:
    text_joined = "\n".join(texts)
    return tokenizer.encode(text_joined)


def encode_bpe(texts: list[str], tokenizer, batch_size: int = 1000) -> list[int]:
    all_ids = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded_batch = tokenizer.encode_batch(batch)
        for encoded in encoded_batch:
            all_ids.extend(encoded.ids)
    return all_ids


def encode_texts(texts: list[str], tokenizer, tokenizer_type: TokenizerTypeEnum) -> list[int]:
    if tokenizer_type == TokenizerTypeEnum.CHAR:
        return encode_char(texts, tokenizer)
    return encode_bpe(texts, tokenizer)
