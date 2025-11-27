from collections.abc import Callable

import torch

from src.utils.training import get_batch


@torch.no_grad()
def estimate_loss(
    model,
    forward_pass: Callable,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    seq_len: int,
    batch_size: int,
    eval_iters: int,
    device: str,
) -> dict[str, float]:
    out = {}
    model.eval()

    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, seq_len, batch_size, device)
            loss = forward_pass(model, x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()

    model.train()
    return out


def train_loop(
    model,
    forward_pass: Callable,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    optimizer,
    seq_len: int,
    batch_size: int,
    max_iters: int,
    eval_interval: int,
    eval_iters: int,
    device: str,
):
    print("Starting training...\n")

    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, forward_pass, train_data, val_data, seq_len, batch_size, eval_iters, device)
            print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        x, y = get_batch(train_data, seq_len, batch_size, device)
        loss = forward_pass(model, x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("\nTraining completed!")
