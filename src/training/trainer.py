import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
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
) -> TrainingMetrics:
    metrics = TrainingMetrics()

    print("Starting training...\n")

    with tqdm(range(max_iters), desc="Training Progress", unit="step") as pbar:
        for step in pbar:
            # eval step
            if step % eval_interval == 0 or step == max_iters - 1:
                losses = estimate_loss(model, forward_pass, data, seq_len, batch_size, eval_iters, device)
                metrics.add(step, losses[DataSplitEnum.TRAIN], losses[DataSplitEnum.VAL])

                #print(
                #    f"Step {step}: "
                #    f"train loss {losses[DataSplitEnum.TRAIN]:.4f} (ppl {math.exp(losses[DataSplitEnum.TRAIN]):.2f}), "
                #    f"val loss {losses[DataSplitEnum.VAL]:.4f} (ppl {math.exp(losses[DataSplitEnum.VAL]):.2f})"
                #)
                pbar.set_postfix(train_loss=losses[DataSplitEnum.TRAIN], val_loss=losses[DataSplitEnum.VAL])
                if wandb_run is not None:
                    wandb_run.log({
                        "train_loss": losses[DataSplitEnum.TRAIN],
                        "val_loss": losses[DataSplitEnum.VAL],
                        "step": step,
                    })

            # training step
            x, y = get_batch(data[DataSplitEnum.TRAIN], seq_len, batch_size, device)
            loss = forward_pass(model, x, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    print("\nTraining completed!")
    return metrics
