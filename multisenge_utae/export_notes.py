"""Export saved metrics JSON to markdown tables for log.md / sir notes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multisenge_seg.taxonomy import CLASS6_NAMES, CLASS10_NAMES


def _class_names(num_classes: int) -> dict[int, str]:
    return CLASS6_NAMES if num_classes == 6 else CLASS10_NAMES


def table5_markdown(scores: dict, num_classes: int, title: str = "Table 5 style") -> str:
    names = _class_names(num_classes)
    p = scores["per_class_precision"]
    r = scores["per_class_recall"]
    s = scores.get("per_class_sensitivity", r)
    spec = scores.get("per_class_specificity", [0.0] * num_classes)
    f = scores["per_class_f1"]
    lines = [
        f"## {title}",
        "",
        "| Class | Name | Precision | Recall | Sensitivity | Specificity | F1 | Support |",
        "|-------|------|-----------|--------|-------------|-------------|-----|---------|",
    ]
    support = scores.get("support") or [0] * num_classes
    for i in range(num_classes):
        cid = i + 1
        lines.append(
            f"| {cid} | {names.get(cid, '?')} | {p[i]:.4f} | {r[i]:.4f} | {s[i]:.4f} | "
            f"{spec[i]:.4f} | {f[i]:.4f} | {int(support[i])} |"
        )
    w_spec = scores.get("weighted_specificity", 0.0)
    w_sens = scores.get("weighted_sensitivity", scores["weighted_recall"])
    lines.extend(
        [
            f"| **W-Avg** | | {scores['weighted_precision']:.4f} | "
            f"{scores['weighted_recall']:.4f} | {w_sens:.4f} | {w_spec:.4f} | "
            f"**{scores['weighted_f1']:.4f}** | |",
            "",
            f"Accuracy: {scores['accuracy']:.4f} · Kappa: {scores['kappa']:.4f} · "
            f"Mean F1: {scores['mean_f1']:.4f} · Mean Sens: {scores.get('mean_sensitivity', 0):.4f} · "
            f"Mean Spec: {scores.get('mean_specificity', 0):.4f}",
            "",
        ]
    )
    cm = scores.get("confusion_matrix")
    if cm:
        lines.append("### Confusion matrix (rows=GT, cols=pred)")
        lines.append("")
        header = "| GT \\ Pred | " + " | ".join(str(i + 1) for i in range(num_classes)) + " |"
        sep = "|---|" + "|".join("---:" for _ in range(num_classes)) + "|"
        lines.extend([header, sep])
        for i, row in enumerate(cm):
            lines.append(f"| **{i + 1}** | " + " | ".join(str(v) for v in row) + " |")
        lines.append("")
    return "\n".join(lines)


def probe_summary_markdown(summary: dict, probe: str) -> str:
    lines = [
        f"## Layer probes ({probe})",
        "",
        "| Level | W-F1 | W-Sens | W-Spec | Accuracy | Kappa |",
        "|-------|------|--------|--------|----------|-------|",
    ]
    for level, scores in summary.items():
        lines.append(
            f"| {level} | {scores['weighted_f1']:.4f} | "
            f"{scores.get('weighted_sensitivity', scores['weighted_recall']):.4f} | "
            f"{scores.get('weighted_specificity', 0):.4f} | "
            f"{scores['accuracy']:.4f} | {scores['kappa']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Write markdown tables from U-TAE metrics JSON")
    p.add_argument("--metrics", type=Path, required=True, help="e.g. test_metrics.json or best_metrics.json")
    p.add_argument("--num-classes", type=int, default=6, choices=[6, 10])
    p.add_argument("--title", type=str, default="U-TAE test metrics")
    p.add_argument("--out", type=Path, default=None, help="Default: same dir as metrics, .md suffix")
    p.add_argument(
        "--probe-summary",
        type=Path,
        default=None,
        help="Optional probe_summary_linear.json for layer table",
    )
    args = p.parse_args()

    scores = json.loads(args.metrics.read_text(encoding="utf-8"))
    out = args.out or args.metrics.with_suffix(".md")
    parts = [table5_markdown(scores, args.num_classes, title=args.title)]
    if args.probe_summary and args.probe_summary.is_file():
        summary = json.loads(args.probe_summary.read_text(encoding="utf-8"))
        probe = args.probe_summary.stem.replace("probe_summary_", "")
        parts.append(probe_summary_markdown(summary, probe))
    out.write_text("\n".join(parts), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
