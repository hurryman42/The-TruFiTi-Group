import json
import matplotlib.pyplot as plt
import os

metric_files = [
    "gru_L2_1.9M_26-01-26_23-56-51_metrics.json",
    "transformer_rope_ffn4x_w-init_L2_0.7M_26-03-12_13-44-27_metrics.json",
]

labels = {
    "gru_L2_1.9M_26-01-26_23-56-51_metrics.json": "GRU (L2 1.9M)",
    "transformer_rope_ffn4x_w-init_L2_0.7M_26-03-12_13-44-27_metrics.json": "Transformer RoPE (0.7M)",
}


def load_metrics(path):
    with open(path) as f:
        data = json.load(f)
    return data["steps"], data["train_loss"], data["val_loss"]


fig, axs = plt.subplots(1, 2, figsize=(15, 5))
ax_train, ax_val = axs

for file in metric_files:
    steps, train_loss, val_loss = load_metrics(file)
    label = labels.get(file, os.path.basename(file).replace("_metrics.json", ""))

    ax_train.plot(steps, train_loss, label=label)
    ax_val.plot(steps, val_loss, label=label)

ax_train.set_xlim(left=0, right=25000)
ax_train.minorticks_on()
ax_train.grid(which="both", axis="y", linestyle="--", color="gray", alpha=0.3)
ax_train.set_title("Training Loss")
ax_train.set_xlabel("Steps")
ax_train.set_ylabel("Loss")
ax_train.legend()

ax_val.set_xlim(left=0, right=25000)
ax_val.minorticks_on()
ax_val.grid(which="both", axis="y", linestyle="--", color="gray", alpha=0.3)
ax_val.set_title("Validation Loss")
ax_val.set_xlabel("Steps")
ax_val.set_ylabel("Loss")
ax_val.legend()

plt.tight_layout()
plt.savefig("gru-compare-1x2.pdf")
plt.close()
