# ruff: noqa

import json
import matplotlib.pyplot as plt
import os

metric_files = [
    "transformer_no-filter_rope_ffn4x_w-init_L2_0.7M_26-03-14_23-42-08_metrics.json",
    "transformer_no-llm_rope_ffn4x_w-init_L2_0.7M_26-03-15_00-40-13_metrics.json",
    "transformer_rope_ffn4x_w-init_L2_0.7M_26-03-12_13-44-27_metrics.json",
]

labels = {
    "transformer_no-filter_rope_ffn4x_w-init_L2_0.7M_26-03-14_23-42-08_metrics.json": "Transformer on not filtered dataset (L2 0.7M)",
    "transformer_no-llm_rope_ffn4x_w-init_L2_0.7M_26-03-15_00-40-13_metrics.json": "Transformer on dataset after mid-filter (L2 0.7M)",
    "transformer_rope_ffn4x_w-init_L2_0.7M_26-03-12_13-44-27_metrics.json": "Transformer on final dataset (L2 0.7M)",
}


def load_metrics(path):
    with open(path) as f:
        data = json.load(f)
    return data["steps"], data["train_loss"], data["val_loss"]


fig, ax_val = plt.subplots(1, 1, figsize=(10, 5))

for file in metric_files:
    steps, train_loss, val_loss = load_metrics(file)
    label = labels.get(file, os.path.basename(file).replace("_metrics.json", ""))

    ax_val.plot(steps, val_loss, label=label)

ax_val.set_xlim(left=0, right=25000)
ax_val.minorticks_on()
ax_val.grid(which="both", axis="y", linestyle="--", color="gray", alpha=0.3)
ax_val.set_title("Validation Loss")
ax_val.set_xlabel("Steps")
ax_val.set_ylabel("Loss")
ax_val.legend()

plt.tight_layout()
plt.savefig("datasets.pdf")
plt.close()
