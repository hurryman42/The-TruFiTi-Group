import wandb
from src.enums import SectionEnum, TrainingEnum, TransformerModelEnum

WANDB_TO_CONFIG_MAPPING = {
    "training.learning_rate": (SectionEnum.TRAINING, TrainingEnum.LEARNING_RATE),
    "training.batch_size": (SectionEnum.TRAINING, TrainingEnum.BATCH_SIZE),
    "training.weight_decay": (SectionEnum.TRAINING, TrainingEnum.WEIGHT_DECAY),
    "training.warmup_iters": (SectionEnum.TRAINING, TrainingEnum.WARMUP_ITERS),
    "model.dropout": (SectionEnum.MODEL, TransformerModelEnum.DROPOUT),
    "model.d_model": (SectionEnum.MODEL, TransformerModelEnum.D_MODEL),
    "model.num_blocks": (SectionEnum.MODEL, TransformerModelEnum.NUM_BLOCKS),
    "model.num_heads": (SectionEnum.MODEL, TransformerModelEnum.NUM_HEADS),
    "model.seq_len": (SectionEnum.MODEL, TransformerModelEnum.SEQ_LEN),
}


def apply_wandb_overrides(config: dict) -> dict:
    for wandb_key, (section, key) in WANDB_TO_CONFIG_MAPPING.items():
        if wandb_key in wandb.config:
            config[section][key] = wandb.config[wandb_key]
    return config
