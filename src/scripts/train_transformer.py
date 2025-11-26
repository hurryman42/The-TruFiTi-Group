from pathlib import Path
import torch

#from tqdm import tqdm

from src.tokenizer.char_tokenizer import CharTokenizer
from src.scripts.read_file import read_file_synopsis_review_pairs
from src.models.transformer.transformer import TransformerDecoderOnly

BASE_DIR = Path(__file__).resolve().parent.parent

DIMENSION_MODEL = 256
NUM_HEADS = 8
MAX_SEQ_LEN = 128
FF_HIDDEN_DIMENSION = 4 * DIMENSION_MODEL

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_ITERS = 3000

EVAL_INTERVAL = 1000  # interval of printing estimated loss
EVAL_ITERS = 20  # number of batches to average for loss estimation
WEIGHT_DECAY = 0 # for AdamW


def load_char_tokenizer() -> CharTokenizer:
    char_tokenizer_path = BASE_DIR.parent / "tokenizer" / "char_tokenizer.json"
    tokenizer = CharTokenizer.load(str(char_tokenizer_path))
    print(f"Vocabulary size: {tokenizer.get_vocab_size}\n")
    return tokenizer

def load_text() -> str:
    input_file = BASE_DIR.parent / "data" / "letterboxd_filtered_short_synopsis_film.jsonl"
    texts = read_file_synopsis_review_pairs(input_file)
    print(f"Number of reviews: {len(texts):,}".replace(",", "."))
    text_string = "\n".join(texts)
    print(f"Total number of chars: {len(text_string):,}\n".replace(",", "."))
    return text_string

def encode_text(text: str, tokenizer: CharTokenizer) -> list[int]:
    encoded_text = tokenizer.encode(text)
    print(f"Number of tokens: {len(encoded_text):,}".replace(",", "."))
    return encoded_text

def train_val_split(data: list[int], train_size: float) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.tensor(data, dtype=torch.long)
    n = int(train_size * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data

def get_batch(
    split: str,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    seq_length: int,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - seq_length, (batch_size,))
    x = torch.stack([data[i : i + seq_length] for i in ix])
    y = torch.stack([data[i + 1 : i + seq_length + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(
    model,
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
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train(
    model,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    optimizer,
    max_iters: int,
    device: str,
):
    print("Starting training...\n")
    #progress_bar = tqdm(range(max_iters), desc="Training", unit="step")
    #for iter in progress_bar:
    for iter in range(max_iters):
        # evaluation
        if iter % EVAL_INTERVAL == 0 or iter == max_iters - 1:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                MAX_SEQ_LEN,
                BATCH_SIZE,
                EVAL_ITERS,
                device,
            )
            print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch("train", train_data, val_data, MAX_SEQ_LEN, BATCH_SIZE, device)
        logits = model(xb)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            yb.view(-1)
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        #progress_bar.set_postfix({"loss": loss.item()})
    print("\nTraining completed!")

def save_model(
    model,
    char_tokenizer: CharTokenizer,
):
    save_path = BASE_DIR.parent / "models" / "transformer_language_model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "vocab_size": char_tokenizer.get_vocab_size,
            "dimension_model": DIMENSION_MODEL,
            "seq_len": MAX_SEQ_LEN,
        },
        save_path,
    )
    print(f"\nModel saved to {save_path}")

def main():
    char_tokenizer = load_char_tokenizer()
    vocab_size = char_tokenizer.get_vocab_size
    text = load_text()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}\n")

    encoded_texts = encode_text(text, char_tokenizer)
    train_data, val_data = train_val_split(encoded_texts, 0.9)

    model = TransformerDecoderOnly(
        vocab_size, embedding_dimension=DIMENSION_MODEL, num_blocks=6, num_heads=NUM_HEADS,
        head_dimension=DIMENSION_MODEL // NUM_HEADS, max_seq_len=MAX_SEQ_LEN, ff_hidden_dimension=FF_HIDDEN_DIMENSION, dropout=0.1
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}\n".replace(",", "."))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train(
        model,
        train_data,
        val_data,
        optimizer,
        MAX_ITERS,
        device,
    )
    save_model(model, char_tokenizer)

if __name__ == "__main__":
    main()