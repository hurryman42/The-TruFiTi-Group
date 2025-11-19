from pathlib import Path

import torch

from src.models.embeddings.positional_encoding import PositionalEncoding
from src.models.embeddings.token_embedding import TokenEmbedding
from src.models.bigram_language_model import BigramLanguageModel
from src.scripts.read_file import read_file
from src.tokenizer.char_tokenizer import CharTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
D_MODEL = 256
SEQ_LEN = 128
BATCH_SIZE = 32
EVAL_ITERS = 50  # Number of batches to average for loss estimation
MAX_ITERS = 3000
EVAL_INTERVAL = 1000  # Interval of printing estimated loss
LEARNING_RATE = 1e-3


def load_char_tokenizer() -> CharTokenizer:
    char_tokenizer_path = BASE_DIR.parent / "tokenizer" / "char_tokenizer.json"
    tokenizer = CharTokenizer.load(str(char_tokenizer_path))
    print(f"Vocabulary size: {tokenizer.get_vocab_size}\n")
    return tokenizer


def load_text() -> str:
    input_file = (
        BASE_DIR.parent / "data" / "letterboxd_filtered_short_synopsis_film.jsonl"
    )
    texts = read_file(input_file)

    print(f"Number of reviews: {len(texts):,}".replace(",", "."))
    text_string = "\n".join(texts)
    print(f"Total number of chars: {len(text_string):,}\n".replace(",", "."))

    return text_string


def encode_text(text: str, tokenizer: CharTokenizer) -> list[int]:
    encoded_text = tokenizer.encode(text)
    print(f"Number of tokens: {len(encoded_text):,}".replace(",", "."))
    return encoded_text


def train_val_split(
    data: list[int], train_size: float
) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.tensor(data, dtype=torch.long)
    n = int(train_size * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data


def get_batch(
    split: str,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    seq_len: int,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == "train" else val_data

    # Generate random starting positions
    ix = torch.randint(len(data) - seq_len, (batch_size,))  # (batch_size,)

    # (batch_size, seq_len)
    x = torch.stack([data[i : i + seq_len] for i in ix])
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix])

    x, y = x.to(device), y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss(
    model,
    token_embedding,
    pos_encoding,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    seq_len: int,
    batch_size: int,
    eval_iters: int,
    device: str,
) -> dict:
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split, train_data, val_data, seq_len, batch_size, device)

            tok_emb = token_embedding(x)
            embeddings = pos_encoding(tok_emb)

            logits, loss = model(embeddings, y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out


def train(
    model: BigramLanguageModel,
    token_embedding: TokenEmbedding,
    pos_encoding: PositionalEncoding,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    optimizer,
    max_iters: int,
    device: str,
):
    print("Starting training...\n")

    for iter in range(max_iters):
        # Evaluation
        if iter % EVAL_INTERVAL == 0 or iter == max_iters - 1:
            losses = estimate_loss(
                model,
                token_embedding,
                pos_encoding,
                train_data,
                val_data,
                SEQ_LEN,
                BATCH_SIZE,
                EVAL_ITERS,
                device,
            )
            print(
                f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        xb, yb = get_batch("train", train_data, val_data, SEQ_LEN, BATCH_SIZE, device)

        tok_emb = token_embedding(xb)
        embeddings = pos_encoding(tok_emb)

        # Forward pass
        logits, loss = model(embeddings, yb)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("\nTraining completed!")


def save_model(
    model: BigramLanguageModel,
    token_embedding: TokenEmbedding,
    pos_encoding: PositionalEncoding,
    char_tokenizer: CharTokenizer,
):
    save_path = BASE_DIR.parent / "models" / "bigram_model.pt"

    torch.save(
        {
            "model": model.state_dict(),
            "token_embedding": token_embedding.state_dict(),
            "pos_encoding": pos_encoding.state_dict(),
            "vocab_size": char_tokenizer.get_vocab_size,
            "d_model": D_MODEL,
            "seq_len": SEQ_LEN,
        },
        save_path,
    )

    print(f"\nModel saved to {save_path}")


def main():
    # Setup
    char_tokenizer = load_char_tokenizer()
    vocab_size = char_tokenizer.get_vocab_size
    text = load_text()
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}\n")

    # Prepare data
    encoded_texts = encode_text(text, char_tokenizer)
    train_data, val_data = train_val_split(encoded_texts, 0.9)

    # Initialize models
    token_embedding = TokenEmbedding(vocab_size, D_MODEL, scale=False).to(device)
    pos_encoding = PositionalEncoding(D_MODEL, max_seq_len=SEQ_LEN).to(device)
    model = BigramLanguageModel(vocab_size, D_MODEL).to(device)
    print("Initialize models finnish")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}\n".replace(",", "."))

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(token_embedding.parameters()),
        lr=LEARNING_RATE,
    )

    train(
        model,
        token_embedding,
        pos_encoding,
        train_data,
        val_data,
        optimizer,
        MAX_ITERS,
        device,
    )

    save_model(model, token_embedding, pos_encoding, char_tokenizer)


if __name__ == "__main__":
    main()
