from enum import StrEnum


class ModelTypeEnum(StrEnum):
    BIGRAM = "bigram"
    TRANSFORMER = "transformer"


class TokenizerTypeEnum(StrEnum):
    CHAR = "char"
    BPE_HUGGING_FACE = "bpe_hugging_face"


class SpecialTokensEnum(StrEnum):
    PAD = "<PAD>"
    EOS = "<EOS>"
    BOS = "<BOS>"


class DataEnum(StrEnum):
    SYNOPSIS = "synopsis"
    REVIEW_TEXT = "review_text"
