import json
import torch
import argparse

from tokenizers import Tokenizer as HFTokenizer
from src.tokenizer.bpe_tokenizer import BPETokenizer
from src.tokenizer.char_tokenizer import CharTokenizer

from src.config import MODEL_DIR, get_data_path, get_model_type, get_tokenizer_path, get_tokenizer_type, load_config
from src.enums import (
    BigramModelEnum,
    CheckpointEnum,
    DataConfigEnum,
    DataSplitEnum,
    ModelTypeEnum,
    SectionEnum,
    TokenizerTypeEnum,
    TrainingEnum,
)
from src.models.bigram_language_model import BigramLanguageModel
from src.models.embeddings.positional_encoding import PositionalEncoding
from src.models.embeddings.token_embedding import TokenEmbedding
from src.training.trainer import TrainingMetrics, train_loop
from src.utils.data_loader import read_file_only_reviews
from src.utils.device import get_device
from src.utils.encoding import encode_texts
from src.utils.tokenizer_loader import load_bpe_hugging_face_tokenizer, load_char_tokenizer, load_bpe_custom_tokenizer
from src.utils.training import train_val_test_split

type AnyTokenizer = CharTokenizer | HFTokenizer | BPETokenizer


def create_forward_pass(token_embedding, pos_encoding):
    def forward_pass(model, x, y):
        tok_emb = token_embedding(x)
        embeddings = pos_encoding(tok_emb)
        _, loss = model(embeddings, y)
        return loss

    return forward_pass


def save_model(model, token_embedding, pos_encoding, vocab_size: int, config: dict):
    tokenizer_type = get_tokenizer_type(config)

    if tokenizer_type == TokenizerTypeEnum.BPE_HUGGING_FACE:
        save_path = MODEL_DIR / "bigram_model_bpe_hugging_face.pt"
    else:
        save_path = MODEL_DIR / "bigram_model.pt"

    save_path.parent.mkdir(parents=True, exist_ok=True)

    model_cfg = config[SectionEnum.MODEL]

    checkpoint = {
        CheckpointEnum.MODEL: model.state_dict(),
        CheckpointEnum.TOKEN_EMBEDDING: token_embedding.state_dict(),
        CheckpointEnum.POS_ENCODING: pos_encoding.state_dict(),
        CheckpointEnum.VOCAB_SIZE: vocab_size,
        CheckpointEnum.D_MODEL: model_cfg[BigramModelEnum.D_MODEL],
        CheckpointEnum.SEQ_LEN: model_cfg[BigramModelEnum.SEQ_LEN],
        CheckpointEnum.TOKENIZER_TYPE: str(tokenizer_type),
    }

    torch.save({str(k): v for k, v in checkpoint.items()}, save_path)
    print(f"Model saved to {save_path}")

    return save_path


def save_metrics(metrics: TrainingMetrics, tokenizer_type: TokenizerTypeEnum):
    if tokenizer_type == TokenizerTypeEnum.BPE_HUGGING_FACE:
        metrics_path = MODEL_DIR / "bigram_model_bpe_hf_metrics.json"
    elif tokenizer_type == TokenizerTypeEnum.BPE:
        metrics_path = MODEL_DIR / "bigram_model_bpe_custom_metrics.json"
    else:
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


def main(config: dict):
    device = get_device()
    tokenizer_type = get_tokenizer_type(config)
    tokenizer_path = get_tokenizer_path(config)

    print(f"Using device: {device}")
    print(f"Tokenizer: {tokenizer_type}\n")

    tokenizer: AnyTokenizer
    match tokenizer_type:
        case TokenizerTypeEnum.CHAR:
            tokenizer = load_char_tokenizer(tokenizer_path)
            vocab_size = tokenizer.get_vocab_size
        case TokenizerTypeEnum.BPE_HUGGING_FACE:
            tokenizer = load_bpe_hugging_face_tokenizer(tokenizer_path)
            vocab_size = tokenizer.get_vocab_size()
        case TokenizerTypeEnum.BPE:
            tokenizer = load_bpe_custom_tokenizer(tokenizer_path)
            vocab_size = tokenizer.get_vocab_size
        case _:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    data_path = get_data_path(config)
    texts = read_file_only_reviews(data_path)
    encoded = encode_texts(texts, tokenizer, tokenizer_type)
    print(f"Total tokens: {len(encoded):,}".replace(",", "."))

    data_cfg = config[SectionEnum.DATA]
    train_texts, val_texts, _ = train_val_test_split(
        texts,
        data_cfg[DataConfigEnum.TRAIN_SIZE],
        data_cfg[DataConfigEnum.VAL_SIZE],
        data_cfg[DataConfigEnum.TEST_SIZE],
    )

    train_data = torch.tensor(encode_texts(train_texts, tokenizer, tokenizer_type), dtype=torch.long)
    val_data = torch.tensor(encode_texts(val_texts, tokenizer, tokenizer_type), dtype=torch.long)

    model_cfg = config[SectionEnum.MODEL]
    d_model = model_cfg[BigramModelEnum.D_MODEL]
    seq_len = model_cfg[BigramModelEnum.SEQ_LEN]

    token_embedding = TokenEmbedding(vocab_size, d_model, scale=False).to(device)
    pos_encoding = PositionalEncoding(d_model, max_seq_len=seq_len).to(device)
    model = BigramLanguageModel(vocab_size, d_model).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n".replace(",", "."))

    training_cfg = config[SectionEnum.TRAINING]
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(token_embedding.parameters()),
        lr=training_cfg[TrainingEnum.LEARNING_RATE],
    )

    forward_pass = create_forward_pass(token_embedding, pos_encoding)

    data = {DataSplitEnum.TRAIN: train_data, DataSplitEnum.VAL: val_data}
    metrics = train_loop(
        model,
        forward_pass,
        data,
        optimizer,
        seq_len,
        training_cfg[TrainingEnum.BATCH_SIZE],
        training_cfg[TrainingEnum.MAX_ITERS],
        training_cfg[TrainingEnum.EVAL_INTERVAL],
        training_cfg[TrainingEnum.EVAL_ITERS],
        device,
    )

    save_model(model, token_embedding, pos_encoding, vocab_size, config)
    save_metrics(metrics, tokenizer_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bigram Language Model")
    parser.add_argument("config", type=str, help="Config file path")

    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Loading config: {args.config}\n")

    if get_model_type(config) != ModelTypeEnum.BIGRAM:
        raise ValueError(f"Config '{args.config}' is not a bigram config")

    main(config)
