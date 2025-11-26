import argparse

import torch

from src.config import TransformerConfig
from src.enums import TokenizerType
from src.models.transformer.transformer import TransformerDecoderOnly
from src.training.trainer import train_loop
from src.utils.data_loader import read_file_synopsis_review_pairs
from src.utils.device import get_device
from src.utils.encoding import encode_texts
from src.utils.tokenizer_loader import load_bpe_hugging_face_tokenizer, load_char_tokenizer
from src.utils.training import train_val_test_split


def create_forward_pass():
    def forward_pass(model, x, y):
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        return loss

    return forward_pass


def save_model(model, vocab_size: int, num_params: int, config: TransformerConfig):
    save_path = config.get_model_save_path(num_params)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "num_params": num_params,
            "vocab_size": vocab_size,
            "d_model": config.d_model,
            "seq_len": config.seq_len,
            "num_heads": config.num_heads,
            "num_blocks": config.num_blocks,
            "ff_hidden_dim": config.ff_hidden_dim,
            "dropout": config.dropout,
            "tokenizer_type": config.tokenizer_type.value,
        },
        save_path,
    )
    print(f"Model saved to {save_path}")


def main(config: TransformerConfig):
    device = get_device()
    print(f"Using device: {device}")
    print(f"Tokenizer: {config.tokenizer_type.value}\n")

    if config.tokenizer_type == TokenizerType.CHAR:
        tokenizer = load_char_tokenizer(config.tokenizer_path)
        vocab_size = tokenizer.get_vocab_size
    else:
        tokenizer = load_bpe_hugging_face_tokenizer(config.tokenizer_path)
        vocab_size = tokenizer.get_vocab_size()

    texts = read_file_synopsis_review_pairs(config.data_path)
    encoded = encode_texts(texts, tokenizer, config.tokenizer_type)
    print(f"Total tokens: {len(encoded):,}".replace(",", "."))

    train_data, val_data, _ = train_val_test_split(encoded, config.train_size, config.val_size)

    model = TransformerDecoderOnly(
        vocab_size,
        embedding_dimension=config.d_model,
        num_blocks=config.num_blocks,
        num_heads=config.num_heads,
        head_dimension=config.head_dim,
        max_seq_len=config.seq_len,
        ff_hidden_dimension=config.ff_hidden_dim,
        dropout=config.dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1_000_000:.1f}M)\n".replace(",", "."))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    forward_pass = create_forward_pass()

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

    save_model(model, vocab_size, num_params, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer Language Model")
    parser.add_argument(
        "--tokenizer",
        type=str,
        choices=["bpe", "char"],
        default="bpe",
        help="Tokenizer type: 'bpe' (default) or 'char'",
    )
    parser.add_argument("--num-blocks", type=int, help="Number of transformer blocks")
    parser.add_argument("--num-heads", type=int, help="Number of attention heads")
    parser.add_argument("--max-iters", type=int, help="Maximum training iterations")
    parser.add_argument("--batch-size", type=int, help="Batch size")

    args = parser.parse_args()

    tokenizer_type = TokenizerType.BPE_HUGGING_FACE if args.tokenizer == "bpe" else TokenizerType.CHAR
    config = TransformerConfig(tokenizer_type=tokenizer_type)

    if args.num_blocks:
        config.num_blocks = args.num_blocks
    if args.num_heads:
        config.num_heads = args.num_heads
    if args.max_iters:
        config.max_iters = args.max_iters
    if args.batch_size:
        config.batch_size = args.batch_size

    main(config)
