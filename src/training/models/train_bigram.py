import argparse
import torch

from src.config import load_config
from src.config.config import Config, BigramModelConfig, BigramTrainingConfig
from src.enums import BigramCheckpointEnum, ModelTypeEnum, DataSplitEnum
from src.models.bigram.bigram_language_model import BigramLanguageModel
from src.models.embeddings.token_embedding import TokenEmbedding
from src.utils.device import get_device
from src.training.trainer import train_loop
from src.training.train_utils import (
    load_and_prepare_data,
    print_training_statistics,
    save_model_checkpoint,
    save_metrics,
)
from dataclasses import asdict


def create_forward_pass(token_embedding):
    def forward_pass(model, x, y):
        embedding = token_embedding(x)
        _, loss = model(embedding, y)
        return loss

    return forward_pass


def main(config: Config):
    assert isinstance(config.model, BigramModelConfig)
    assert isinstance(config.training, BigramTrainingConfig)
    model_config: BigramModelConfig = config.model
    train_config: BigramTrainingConfig = config.training

    device = get_device()

    print(f"Using device: {device}")
    print(f"Tokenizer: {config.tokenizer.type}\n")

    tokenizer, vocab_size, train_data, val_data = load_and_prepare_data(config)

    token_embedding = TokenEmbedding(vocab_size, model_config.d_model, scale=False).to(device)
    model = BigramLanguageModel(vocab_size, model_config.d_model).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n".replace(",", "."))

    print_training_statistics(config, train_data.numel())

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(token_embedding.parameters()),
        lr=config.training.learning_rate,
    )

    forward_pass = create_forward_pass(token_embedding)
    data = {DataSplitEnum.TRAIN: train_data, DataSplitEnum.VAL: val_data}
    metrics = train_loop(
        model,
        forward_pass,
        data,
        optimizer,
        model_config.seq_len,
        train_config.batch_size,
        train_config.max_iters,
        train_config.eval_interval,
        train_config.eval_iters,
        device,
    )

    checkpoint = {
        BigramCheckpointEnum.MODEL: model.state_dict(),
        BigramCheckpointEnum.TOKEN_EMBEDDING: token_embedding.state_dict(),
        BigramCheckpointEnum.VOCAB_SIZE: vocab_size,
        BigramCheckpointEnum.CONFIG: asdict(config),
        BigramCheckpointEnum.TOKENIZER: tokenizer,
    }

    model_save_path = save_model_checkpoint(config, total_params, checkpoint)
    save_metrics(metrics, model_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bigram Language Model")
    parser.add_argument("config", type=str, help="Config name")

    args = parser.parse_args()

    config = load_config(args.config)
    assert isinstance(config.model, BigramModelConfig)
    print(f"Loading config: {args.config}\n")

    if config.model.type != ModelTypeEnum.BIGRAM:
        raise ValueError(f"Config '{args.config}' is not a bigram config")

    main(config)
