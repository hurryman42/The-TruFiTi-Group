import json
from dataclasses import asdict

import torch
import argparse

from src.config import MODEL_DIR, get_data_path
from src.dto.config import Config
from src.enums import BigramCheckpointEnum, DataSplitEnum, ModelTypeEnum, TokenizerTypeEnum
from src.models.bigram_language_model import BigramLanguageModel
from src.models.embeddings.positional_encoding import PositionalEncoding
from src.models.embeddings.token_embedding import TokenEmbedding
from src.training.trainer import TrainingMetrics, train_loop
from src.utils.data_loader import read_file_only_reviews
from src.utils.device import get_device
from src.utils.encoding import encode_texts
from src.utils.tokenizer_loader import load_tokenizer
from src.utils.training import train_val_test_split


def create_forward_pass(token_embedding, pos_encoding):
    def forward_pass(model, x, y):
        tok_emb = token_embedding(x)
        embeddings = pos_encoding(tok_emb)
        _, loss = model(embeddings, y)
        return loss

    return forward_pass


def save_model(model, token_embedding, pos_encoding, vocab_size: int, config: Config, tokenizer):
    if config.tokenizer.type == TokenizerTypeEnum.BPE_HUGGING_FACE:
        save_path = MODEL_DIR / "bigram_model_bpe_hugging_face.pt"
    else:
        save_path = MODEL_DIR / "bigram_model.pt"

    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        BigramCheckpointEnum.MODEL: model.state_dict(),
        BigramCheckpointEnum.TOKEN_EMBEDDING: token_embedding.state_dict(),
        BigramCheckpointEnum.POS_ENCODING: pos_encoding.state_dict(),
        BigramCheckpointEnum.VOCAB_SIZE: vocab_size,
        BigramCheckpointEnum.CONFIG: asdict(config),
        BigramCheckpointEnum.TOKENIZER: tokenizer,
    }

    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")

    return save_path


def save_metrics(metrics: TrainingMetrics, tokenizer_type: TokenizerTypeEnum):
    match tokenizer_type:
        case TokenizerTypeEnum.BPE_HUGGING_FACE:
            metrics_path = MODEL_DIR / "bigram_model_bpe_hf_metrics.json"
        case TokenizerTypeEnum.BPE:
            metrics_path = MODEL_DIR / "bigram_model_bpe_custom_metrics.json"
        case _:
            metrics_path = MODEL_DIR / "bigram_model_metrics.json"

    with open(metrics_path, "w") as f:
        json.dump(
            {
                "steps": metrics.steps,
                "train_loss": metrics.train_loss,
                "val_loss": metrics.val_loss,
                "train_perplexity": metrics.train_perplexity,
                "val_perplexity": metrics.val_perplexity,
            },
            f,
            indent=2,
        )

    print(f"Metrics saved to {metrics_path}")


def main(config: Config):
    device = get_device()

    print(f"Using device: {device}")
    print(f"Tokenizer: {config.tokenizer.type}\n")

    tokenizer = load_tokenizer(config.tokenizer)
    vocab_size = tokenizer.get_vocab_size()

    texts = read_file_only_reviews(get_data_path(config.data.file))
    encoded = encode_texts(texts, tokenizer, config.tokenizer.type)
    print(f"Total tokens: {len(encoded):,}".replace(",", "."))

    train_texts, val_texts, _ = train_val_test_split(
        texts,
        config.data.train_size,
        config.data.val_size,
        config.data.test_size,
    )

    train_data = torch.tensor(encode_texts(train_texts, tokenizer, config.tokenizer.type), dtype=torch.long)
    val_data = torch.tensor(encode_texts(val_texts, tokenizer, config.tokenizer.type), dtype=torch.long)

    token_embedding = TokenEmbedding(vocab_size, config.model.d_model, scale=False).to(device)
    pos_encoding = PositionalEncoding(config.model.d_model, max_seq_len=config.model.seq_len).to(device)
    model = BigramLanguageModel(vocab_size, config.model.d_model).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n".replace(",", "."))

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(token_embedding.parameters()),
        lr=config.training.learning_rate,
    )

    forward_pass = create_forward_pass(token_embedding, pos_encoding)

    data = {DataSplitEnum.TRAIN: train_data, DataSplitEnum.VAL: val_data}
    metrics = train_loop(
        model,
        forward_pass,
        data,
        optimizer,
        config.model.seq_len,
        config.training.batch_size,
        config.training.max_iters,
        config.training.eval_interval,
        config.training.eval_iters,
        device,
    )

    save_model(model, token_embedding, pos_encoding, vocab_size, config, tokenizer)
    save_metrics(metrics, config.tokenizer.type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bigram Language Model")
    parser.add_argument("config", type=str, help="Config file path")

    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    print(f"Loading config: {args.config}\n")

    if config.model.type != ModelTypeEnum.BIGRAM:
        raise ValueError(f"Config '{args.config}' is not a bigram config")

    main(config)
