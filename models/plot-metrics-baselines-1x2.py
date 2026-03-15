import json
import matplotlib.pyplot as plt
import os

metric_files = [
    "bigram_L2_0.5M_26-01-24_14-12-22_metrics.json",
    "transformer_sinusoidal_ffn4x_no-init_L2_0.7M_26-03-12_14-00-41_metrics.json",
    "transformer_rope_ffn4x_no-init_L2_0.7M_26-01-24_15-10-31_metrics.json",
]

labels = {
    "bigram_L2_0.5M_26-01-24_14-12-22_metrics.json": "Bigram (L2 0.5M)",
    "transformer_sinusoidal_ffn4x_no-init_L2_0.7M_26-03-12_14-00-41_metrics.json": "Transformer (Sinusoidal)",
    "transformer_rope_ffn4x_no-init_L2_0.7M_26-01-24_15-10-31_metrics.json": "Transformer (RoPE)",
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
plt.savefig("baselines-1x2.pdf")
plt.close()
