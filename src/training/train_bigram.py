import argparse

import torch

from src.config import BigramConfig
from src.enums import TokenizerType
from src.models.bigram_language_model import BigramLanguageModel
from src.models.embeddings.positional_encoding import PositionalEncoding
from src.models.embeddings.token_embedding import TokenEmbedding
from src.training.trainer import train_loop
from src.utils.data_loader import read_file_only_reviews
from src.utils.device import get_device
from src.utils.encoding import encode_texts
from src.utils.tokenizer_loader import load_bpe_hugging_face_tokenizer, load_char_tokenizer
from src.utils.training import train_val_test_split


def create_forward_pass(token_embedding, pos_encoding):
    def forward_pass(model, x, y):
        tok_emb = token_embedding(x)
        embeddings = pos_encoding(tok_emb)
        _, loss = model(embeddings, y)
        return loss

    return forward_pass


def save_model(model, token_embedding, pos_encoding, vocab_size: int, config: BigramConfig):
    config.model_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "token_embedding": token_embedding.state_dict(),
            "pos_encoding": pos_encoding.state_dict(),
            "vocab_size": vocab_size,
            "d_model": config.d_model,
            "seq_len": config.seq_len,
            "tokenizer_type": config.tokenizer_type.value,
        },
        config.model_save_path,
    )
    print(f"Model saved to {config.model_save_path}")


def main(config: BigramConfig):
    device = get_device()
    print(f"Using device: {device}")
    print(f"Tokenizer: {config.tokenizer_type.value}\n")

    if config.tokenizer_type == TokenizerType.CHAR:
        tokenizer = load_char_tokenizer(config.tokenizer_path)
        vocab_size = tokenizer.get_vocab_size
    else:
        tokenizer = load_bpe_hugging_face_tokenizer(config.tokenizer_path)
        vocab_size = tokenizer.get_vocab_size()

    texts = read_file_only_reviews(config.data_path)
    encoded = encode_texts(texts, tokenizer, config.tokenizer_type)
    print(f"Total tokens: {len(encoded):,}".replace(",", "."))

    train_data, val_data, _ = train_val_test_split(encoded, config.train_size, config.val_size)

    token_embedding = TokenEmbedding(vocab_size, config.d_model, scale=False).to(device)
    pos_encoding = PositionalEncoding(config.d_model, max_seq_len=config.seq_len).to(device)
    model = BigramLanguageModel(vocab_size, config.d_model).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n".replace(",", "."))

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(token_embedding.parameters()),
        lr=config.learning_rate,
    )

    forward_pass = create_forward_pass(token_embedding, pos_encoding)

    train_loop(
        model,
        forward_pass,
        train_data,
        val_data,
        optimizer,
        config.seq_len,
        config.batch_size,
        config.max_iters,
        config.eval_interval,
        config.eval_iters,
        device,
    )

    save_model(model, token_embedding, pos_encoding, vocab_size, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bigram Language Model")
    parser.add_argument(
        "--tokenizer",
        type=str,
        choices=["bpe", "char"],
        default="bpe",
        help="Tokenizer type: 'bpe' (default) or 'char'",
    )
    parser.add_argument("--max-iters", type=int, help="Maximum training iterations")
    parser.add_argument("--batch-size", type=int, help="Batch size")

    args = parser.parse_args()

    tokenizer_type = TokenizerType.BPE_HUGGING_FACE if args.tokenizer == "bpe" else TokenizerType.CHAR
    config = BigramConfig(tokenizer_type=tokenizer_type)

    if args.max_iters:
        config.max_iters = args.max_iters
    if args.batch_size:
        config.batch_size = args.batch_size

    main(config)
