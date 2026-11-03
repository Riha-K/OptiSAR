"""Plot train/val curves from history.json (sir-facing learning curve).

Example:
  python -m multisenge_utae.plot_history \\
    multisenge_utae/checkpoints/run_c6_head_v0/history.json

  python -m multisenge_utae.plot_history \\
    multisenge_utae/checkpoints/run_c6_full_v0/history.json \\
    --title "U-TAE P5 full fine-tune"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot_history(history: list[dict], out: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib not installed - on PARAM try: pip install --user matplotlib"
        ) from exc

    epochs = [int(r["epoch"]) for r in history]
    train_loss = [float(r["train_loss"]) for r in history]
    val_wf1 = [float(r["val_weighted_f1"]) for r in history]
    val_kappa = [float(r.get("val_kappa", 0)) for r in history]
    has_mean_f1 = all("val_mean_f1" in r for r in history)
    val_mean_f1 = [float(r["val_mean_f1"]) for r in history] if has_mean_f1 else None

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    fig.suptitle(title, fontsize=12)

    ax0 = axes[0]
    ax0.plot(epochs, train_loss, "o-", color="#2563eb", linewidth=1.5, markersize=4, label="train loss (mean/batch)")
    ax0.set_ylabel("Train loss")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper right")

    ax1 = axes[1]
    ax1.plot(epochs, val_wf1, "o-", color="#16a34a", linewidth=1.5, markersize=4, label="val weighted F1")
    if val_mean_f1 is not None:
        ax1.plot(epochs, val_mean_f1, "s--", color="#ca8a04", linewidth=1.2, markersize=3, label="val mean F1")
    ax1.plot(epochs, val_kappa, "^--", color="#9333ea", linewidth=1.2, markersize=3, label="val kappa")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Val score")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right")

    best_i = max(range(len(val_wf1)), key=lambda i: val_wf1[i])
    ax1.axvline(epochs[best_i], color="#16a34a", linestyle=":", alpha=0.5)
    ax1.annotate(
        f"best ep {epochs[best_i]}\nwF1={val_wf1[best_i]:.4f}",
        xy=(epochs[best_i], val_wf1[best_i]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=8,
        color="#166534",
    )

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main() -> int:
    p = argparse.ArgumentParser(description="Plot training curves from history.json")
    p.add_argument("history", type=Path, help="Path to history.json")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG (default: same dir as history, history_plot.png)",
    )
    p.add_argument("--title", type=str, default=None, help="Figure title")
    args = p.parse_args()

    if not args.history.is_file():
        raise SystemExit(f"history not found: {args.history}")

    history = json.loads(args.history.read_text(encoding="utf-8"))
    if not history:
        raise SystemExit("history.json is empty")

    mode = history[0].get("mode", "")
    title = args.title or f"Training curves ({mode or 'run'})"
    out = args.out or args.history.with_name("history_plot.png")
    plot_history(history, out, title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
