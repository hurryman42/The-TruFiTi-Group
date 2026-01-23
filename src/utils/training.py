import torch


def train_val_test_split(
    data: list[str],
    train_size: float = 0.9,
    val_size: float = 0.1,
    test_size: float | None = None,
) -> tuple[list[str], list[str], list[str] | None]:
    if test_size is None:
        assert train_size + val_size == 1.0, f"train_size + val_size must equal 1.0, got {train_size + val_size}"
    else:
        assert train_size + val_size + test_size == 1.0, (
            f"train_size + val_size + test_size must equal 1.0, got {train_size + val_size + test_size}"
        )

    n = len(data)
    train_end = int(train_size * n)
    val_end = train_end + int(val_size * n)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:] if test_size is not None else None

    return train_data, val_data, test_data


def get_batch(
    data: torch.Tensor,
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    # ix = torch.tensor([i + seq_len + 1 for i in range(batch_size)], dtype=torch.long, device=device)

    x = torch.stack([data[i : i + seq_len] for i in ix])
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix])

    x, y = x.to(device), y.to(device)

    return x, y
