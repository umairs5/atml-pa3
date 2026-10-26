"""Learning curves for the five task 1 runs.

python -m task1.plot_curves --runs_dir RUNS --out report/figures
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

RUNS = [("full_ft", "Full FT"), ("full_ft_l2sp", "Full FT + L2-SP"), ("lora_r4", "LoRA-4"),
        ("lora_r16", "LoRA-16"), ("oplora_r16", "OPLoRA-16")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--out", default="report/figures")
    ap.add_argument("--smooth", type=int, default=25)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    logs = {}
    for run, label in RUNS:
        f = os.path.join(args.runs_dir, run, "train_log.jsonl")
        if os.path.exists(f):
            logs[label] = pd.read_json(f, lines=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for label, df in logs.items():
        axes[0].plot(df["update"], df["sft_loss"].rolling(args.smooth, min_periods=1).mean(), label=label)
        axes[1].plot(df["update"], df["lr"], label=label)
    axes[0].set(xlabel="optimizer update", ylabel=f"SFT loss ({args.smooth}-update mean)", title="SFT loss")
    axes[1].set(xlabel="optimizer update", ylabel="learning rate", title="Learning rate", yscale="log")
    axes[0].legend(fontsize=8)

    if "Full FT + L2-SP" in logs:
        df = logs["Full FT + L2-SP"]
        ax = axes[2]
        ax.plot(df["update"], df["sft_loss"].rolling(args.smooth, min_periods=1).mean(), label="SFT loss")
        ax.plot(df["update"], df["lam_omega"], label=r"$\lambda\Omega$")
        ax.plot(df["update"], df["total_loss"].rolling(args.smooth, min_periods=1).mean(), label="total")
        ax.set(xlabel="optimizer update", title="L2-SP objective terms")
        ax.legend(fontsize=8)
    else:
        axes[2].axis("off")

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(args.out, f"task1_learning_curves.{ext}"), dpi=200)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
