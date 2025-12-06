from enum import StrEnum


class SectionEnum(StrEnum):
    MODEL = "model"
    TOKENIZER = "tokenizer"
    TRAINING = "training"
    DATA = "data"


class BigramModelEnum(StrEnum):
    TYPE = "type"
    D_MODEL = "d_model"
    SEQ_LEN = "seq_len"


class TransformerModelEnum(StrEnum):
    TYPE = "type"
    D_MODEL = "d_model"
    NUM_HEADS = "num_heads"
    NUM_BLOCKS = "num_blocks"
    SEQ_LEN = "seq_len"
    DROPOUT = "dropout"
    FF_HIDDEN_DIM = "ff_hidden_dim"
    HEAD_DIM = "head_dim"
    USE_ROPE = "use_rope"


class TrainingEnum(StrEnum):
    BATCH_SIZE = "batch_size"
    LEARNING_RATE = "learning_rate"
    WEIGHT_DECAY = "weight_decay"
    MAX_ITERS = "max_iters"
    EVAL_INTERVAL = "eval_interval"
    EVAL_ITERS = "eval_iters"
    WARMUP_ITERS = "warmup_iters"


class DataConfigEnum(StrEnum):
    TRAIN_SIZE = "train_size"
    VAL_SIZE = "val_size"
    TEST_SIZE = "test_size"
    SEED = "seed"
    FILE = "file"
