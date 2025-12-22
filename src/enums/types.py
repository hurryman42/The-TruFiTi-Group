from enum import StrEnum


class ModelTypeEnum(StrEnum):
    BIGRAM = "bigram"
    TRANSFORMER = "transformer"


class TokenizerTypeEnum(StrEnum):
    CHAR = "char"
    BPE_HUGGING_FACE = "bpe_hugging_face"
    BPE = "bpe"


class SpecialTokensEnum(StrEnum):
    PAD = "<PAD>"
    BOS = "<BOS>"
    EOS = "<EOS>"


class DataEnum(StrEnum):
    SYNOPSIS = "synopsis"
    REVIEW_TEXTS = "review_texts"
