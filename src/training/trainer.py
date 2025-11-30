import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from src.enums import DataSplitEnum
from src.utils.training import get_batch


@dataclass
class TrainingMetrics:
    steps: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)

    def add(self, step: int, train_loss: float, val_loss: float):
        self.steps.append(step)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)

    @property
    def train_perplexity(self) -> list[float]:
        return [math.exp(loss) for loss in self.train_loss]

    @property
    def val_perplexity(self) -> list[float]:
        return [math.exp(loss) for loss in self.val_loss]


@torch.no_grad()
def estimate_loss(
    model,
    forward_pass: Callable,
    data: dict[DataSplitEnum, torch.Tensor],
    seq_len: int,
    batch_size: int,
    eval_iters: int,
    device: str,
) -> dict[DataSplitEnum, float]:
    model.eval()

    losses = {}
    for split in DataSplitEnum:
        split_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data[split], seq_len, batch_size, device)
            loss = forward_pass(model, x, y)
            split_losses[k] = loss.item()
        losses[split] = split_losses.mean().item()

    model.train()
    return losses


def create_scheduler(optimizer, warmup_iters: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if warmup_iters > 0 and step < warmup_iters:
            return (step + 1) / warmup_iters
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def check_early_stopping(
    val_loss: float,
    best_val_loss: float,
    patience_counter: int,
    patience: int,
    min_delta: float,
) -> tuple[float, int, bool]:
    if val_loss < best_val_loss - min_delta:
        return val_loss, 0, False

    new_counter = patience_counter + 1
    return best_val_loss, new_counter, new_counter >= patience


def should_eval(step: int, eval_interval: int, max_iters: int) -> bool:
    if step % eval_interval == 0:
        return True
    if step == max_iters - 1:
        return True
    if step <= 1000 and step % 100 == 0:
        return True
    return False


def train_loop(
    model,
    forward_pass: Callable,
    data: dict[DataSplitEnum, torch.Tensor],
    optimizer,
    seq_len: int,
    batch_size: int,
    max_iters: int,
    eval_interval: int,
    eval_iters: int,
    device: str,
    wandb_run=None,
    patience: int = 5,
    min_delta: float = 1e-3,
    warmup_iters: int = 0,
) -> TrainingMetrics:
    metrics = TrainingMetrics()
    best_val_loss = float("inf")
    patience_counter = 0
    scheduler = create_scheduler(optimizer, warmup_iters)

    print("Starting training...\n")

    with tqdm(range(max_iters), desc="Training Progress", unit="step") as pbar:
        for step in pbar:
            if should_eval(step, eval_interval, max_iters):
                losses = estimate_loss(model, forward_pass, data, seq_len, batch_size, eval_iters, device)
                metrics.add(step, losses[DataSplitEnum.TRAIN], losses[DataSplitEnum.VAL])
                pbar.set_postfix(train_loss=losses[DataSplitEnum.TRAIN], val_loss=losses[DataSplitEnum.VAL])

                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train_loss": losses[DataSplitEnum.TRAIN],
                            "val_loss": losses[DataSplitEnum.VAL],
                            "train_perplexity": math.exp(losses[DataSplitEnum.TRAIN]),
                            "val_perplexity": math.exp(losses[DataSplitEnum.VAL]),
                        },
                        step=step,
                    )

                best_val_loss, patience_counter, should_stop = check_early_stopping(
                    losses[DataSplitEnum.VAL], best_val_loss, patience_counter, patience, min_delta
                )
                if should_stop:
                    print(f"\nEarly stopping at step {step}")
                    break

            x, y = get_batch(data[DataSplitEnum.TRAIN], seq_len, batch_size, device)
            loss = forward_pass(model, x, y)
            optimizer.zero_grad(set_to_none=True)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()  # adjusting learning rate for next step

    print("\nTraining completed!")
    return metrics
