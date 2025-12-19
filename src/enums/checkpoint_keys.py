from enum import StrEnum


class CheckpointEnum(StrEnum):
    MODEL = "model"
    TOKEN_EMBEDDING = "token_embedding"
    POS_ENCODING = "pos_encoding"
    VOCAB_SIZE = "vocab_size"
    D_MODEL = "d_model"
    SEQ_LEN = "seq_len"
    NUM_HEADS = "num_heads"
    NUM_BLOCKS = "num_blocks"
    NUM_PARAMS = "num_params"
    FF_HIDDEN_DIM = "ff_hidden_dim"
    DROPOUT = "dropout"
    TOKENIZER_TYPE = "tokenizer_type"
    TOKENIZER_NAME = "tokenizer_name"
    DATA_SEED = "data_seed"
    USE_ROPE = "use_rope"
    DATA_FILE = "data_file"
