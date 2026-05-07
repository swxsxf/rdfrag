"""Plot LoRA training loss from HuggingFace Trainer state.

The plot is intended for the thesis chapter with the fine-tuning experiment.
It uses the trainer_state.json saved by Kaggle/Trainer and does not rerun
training or inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_STATE = Path("kaggle/working/qwen3-rdfrag-lora/checkpoint-20/trainer_state.json")
DEFAULT_OUTPUT = Path("artifacts/plots/training/qwen3_lora_training_loss.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    state = json.loads(args.state.read_text(encoding="utf-8"))
    log_history = state.get("log_history", [])
    points = [
        (int(item["step"]), float(item["loss"]))
        for item in log_history
        if "loss" in item and "step" in item
    ]
    if not points:
        raise RuntimeError(f"No loss points found in {args.state}")

    steps = [step for step, _ in points]
    losses = [loss for _, loss in points]
    quality_proxy = [1 / (1 + loss) for loss in losses]

    color_bg = "#ffffff"
    color_text = "#0f172a"
    color_subtext = "#475569"
    color_grid = "#cbd5e1"
    color_blue = "#2563eb"
    color_green = "#10b981"

    fig, ax = plt.subplots(figsize=(9.5, 5.2), facecolor=color_bg)
    ax.set_facecolor(color_bg)
    ax.plot(steps, losses, marker="o", linewidth=2.6, markersize=8, color=color_blue, label="Training loss")
    ax.set_title("Динамика loss при LoRA-дообучении Qwen3:8B", fontsize=17, fontweight="bold", color=color_text, pad=14)
    ax.set_xlabel("Шаг обучения", fontsize=12, color=color_text)
    ax.set_ylabel("Loss", fontsize=12, color=color_text)
    ax.grid(axis="both", color=color_grid, alpha=0.7, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=color_text, labelsize=11)
    ax.set_xticks(steps)
    for spine in ax.spines.values():
        spine.set_color(color_grid)

    for step, loss in points:
        ax.annotate(
            f"{loss:.3f}",
            xy=(step, loss),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            color=color_text,
        )

    ax2 = ax.twinx()
    ax2.plot(steps, quality_proxy, marker="s", linewidth=2.0, markersize=6, color=color_green, linestyle="--", label="1 / (1 + loss)")
    ax2.set_ylabel("Условный показатель качества, 1 / (1 + loss)", fontsize=11, color=color_subtext)
    ax2.tick_params(colors=color_subtext, labelsize=10)
    for spine in ax2.spines.values():
        spine.set_color(color_grid)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, facecolor=color_bg)
    plt.close(fig)
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
