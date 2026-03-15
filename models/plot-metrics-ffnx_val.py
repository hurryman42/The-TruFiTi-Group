import json
import matplotlib.pyplot as plt
import os

metric_files = [
    "transformer_rope_ffn1x_w-init_L2_0.6M_26-03-11_21-43-37_metrics.json",
    "transformer_rope_ffn2x_w-init_L2_0.6M_26-03-12_15-01-23_metrics.json",
    "transformer_rope_ffn4x_w-init_L2_0.7M_26-03-12_13-44-27_metrics.json",
]

labels = {
    "transformer_rope_ffn1x_w-init_L2_0.6M_26-03-11_21-43-37_metrics.json": "Transformer FFN 1x L2 0.6M",
    "transformer_rope_ffn2x_w-init_L2_0.6M_26-03-12_15-01-23_metrics.json": "Transformer FFN 2x L2 0.6M",
    "transformer_rope_ffn4x_w-init_L2_0.7M_26-03-12_13-44-27_metrics.json": "Transformer FFN 4x L2 0.7M",
}

metric_files_big = [
    "transformer_big_rope_ffn1x_w-init_L2_5.2M_26-03-13_12-12-39_metrics.json",
    "transformer_big_rope_ffn2x_w-init_L2_6.3M_26-03-13_11-26-26_metrics.json",
    "transformer_big_rope_ffn4x_w-init_L2_8.4M_26-03-13_13-19-14_metrics.json",
]

labels_big = {
    "transformer_big_rope_ffn1x_w-init_L2_5.2M_26-03-13_12-12-39_metrics.json": "Transformer FFN 1x L2 5.2M",
    "transformer_big_rope_ffn2x_w-init_L2_6.3M_26-03-13_11-26-26_metrics.json": "Transformer FFN 2x L2 6.3M",
    "transformer_big_rope_ffn4x_w-init_L2_8.4M_26-03-13_13-19-14_metrics.json": "Transformer FFN 4x L2 8.4M",
}


def load_metrics(path):
    with open(path) as f:
        data = json.load(f)
    return data["steps"], data["train_loss"], data["val_loss"]


fig, axs = plt.subplots(1, 2, figsize=(15, 5))
ax_val, ax_big = axs

for file in metric_files:
    steps, train_loss, val_loss = load_metrics(file)
    label = labels.get(file, os.path.basename(file).replace("_metrics.json", ""))

    ax_val.plot(steps, val_loss, label=label)

ax_val.set_xlim(left=0, right=25000)
ax_val.set_ylim(bottom=3.2, top=8.5)
ax_val.minorticks_on()
ax_val.grid(which="both", axis="y", linestyle="--", color="gray", alpha=0.3)
ax_val.set_title("Validation Loss Transformers FFN 1x-4x")
ax_val.set_xlabel("Steps")
ax_val.set_ylabel("Loss")
ax_val.legend()

for file in metric_files_big:
    steps, train_loss, val_loss = load_metrics(file)
    label = labels_big.get(file, os.path.basename(file).replace("_metrics.json", ""))

    ax_big.plot(steps, val_loss, label=label)

ax_big.set_xlim(left=0, right=25000)
ax_big.set_ylim(bottom=3.2, top=8.5)
ax_big.minorticks_on()
ax_big.grid(which="both", axis="y", linestyle="--", color="gray", alpha=0.3)
ax_big.set_title("Validation Loss Big Transformers FFN 1x-4x")
ax_big.set_xlabel("Steps")
ax_big.set_ylabel("Loss")
ax_big.legend()

plt.tight_layout()
plt.savefig("ffnx-compare-1x2.pdf")
plt.close()
