"""Train U-TAE on MultiSenGE (A5 breast-paper style: head then full).

Phases:
  --mode head  : freeze encoder + L-TAE, train decoder/head only (paper P4)
  --mode full  : fine-tune all weights (paper P5)

Geographic tile split matches multisenge_seg (train 8 tiles / val 31UFP+31UGP / test 31UEQ).
Start with --num-classes 6; repeat with 10 for Table 6.

Example (smoke):
  python -m multisenge_utae.train \\
    --index multisenge_seg/artifacts/patch_index.json \\
    --num-classes 6 --mode head --epochs 2 \\
    --max-train 8 --max-val 4 \\
    --out-dir multisenge_utae/checkpoints/run_c6_head_smoke
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from multisenge_seg.build_index import records_from_json
from multisenge_seg.dataset import MultiSenGETemporalDataset, PatchRecord, build_patch_index
from multisenge_seg.metrics import (
    accumulate_confusion,
    class_weights_from_counts,
    format_prf_table,
    scores_from_cm,
)
from multisenge_seg.taxonomy import num_output_classes
from multisenge_seg.train_utils import (
    apply_class_boost,
    estimate_channel_stats,
    estimate_class_counts,
    estimate_class_counts_from_gr,
    parse_class_boost,
    set_seed,
    _seed_worker,
)
from multisenge_utae.data import batch_positions, collate_utae, modality_input_dim, select_modality
from multisenge_utae.losses import SegCompositeLoss
from multisenge_utae.models import UTAE


@torch.no_grad()
def evaluate(
    model: UTAE,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    modality: str = "both",
) -> dict:
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for batch in loader:
        x = select_modality(batch["x"].to(device), modality)
        mask = batch["mask"].numpy()
        bp = batch_positions(x.shape[0], device)
        logits = model(x, batch_positions=bp)
        pred = logits.argmax(dim=1).cpu().numpy()
        cm += accumulate_confusion(pred, mask, num_classes=num_classes, ignore_index=255)
    return scores_from_cm(cm)


def train_one_epoch(
    model: UTAE,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    accum_steps: int = 1,
    modality: str = "both",
) -> float:
    model.train()
    accum_steps = max(int(accum_steps), 1)
    total = 0.0
    n = 0
    opt.zero_grad(set_to_none=True)
    for i, batch in enumerate(loader):
        x = select_modality(batch["x"].to(device), modality)
        mask = batch["mask"].to(device)
        bp = batch_positions(x.shape[0], device)
        logits = model(x, batch_positions=bp)
        loss = criterion(logits, mask) / accum_steps
        loss.backward()
        if (i + 1) % accum_steps == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)
        total += float(loss.item()) * accum_steps
        n += 1
    if n % accum_steps != 0:
        opt.step()
        opt.zero_grad(set_to_none=True)
    return total / max(n, 1)


def build_model(
    input_dim: int,
    num_classes: int,
    mode: str,
    init_ckpt: Path | None,
    device: torch.device,
    activation: str = "relu",
) -> UTAE:
    model = UTAE(input_dim=input_dim, num_classes=num_classes, activation=activation).to(device)
    if init_ckpt is not None:
        if not init_ckpt.is_file():
            raise FileNotFoundError(f"init checkpoint not found: {init_ckpt}")
        try:
            ckpt = torch.load(init_ckpt, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(init_ckpt, map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"loaded init_ckpt={init_ckpt} missing={len(missing)} unexpected={len(unexpected)}")
    model.set_train_mode(mode)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"train mode={mode} trainable={trainable:,} / {total:,}")
    return model


def _run_eval(args, records: list[PatchRecord], n_cls: int) -> int:
    ckpt_path = args.eval_ckpt
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    device = torch.device(args.device)
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    stats = ckpt.get("norm_stats")
    if not stats:
        stats_path = ckpt_path.parent / "norm_stats.json"
        if not stats_path.is_file():
            raise RuntimeError("checkpoint has no norm_stats; expected sibling norm_stats.json")
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    n_cls = int(ckpt.get("num_classes", n_cls))
    meta = ckpt.get("args") or {}
    modality = str(meta.get("modality", args.modality))
    input_dim = int(ckpt.get("input_dim", modality_input_dim(modality)))
    activation = str(meta.get("activation", "relu"))
    ds = MultiSenGETemporalDataset(
        records,
        args.eval_split,
        num_classes=6 if n_cls == 6 else 10,
        augment=False,
        s1_mean=stats["s1_mean"],
        s1_std=stats["s1_std"],
        s2_mean=stats["s2_mean"],
        s2_std=stats["s2_std"],
    )
    if len(ds) == 0:
        raise RuntimeError(f"no patches for split={args.eval_split}")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_utae,
    )
    model = UTAE(input_dim=input_dim, num_classes=n_cls, activation=activation).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"eval ckpt={ckpt_path} split={args.eval_split} n={len(ds)} classes={n_cls} modality={modality}")
    scores = evaluate(model, loader, device, n_cls, modality=modality)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{args.eval_split}_metrics.json"
    dest.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(
        f"{args.eval_split} wP={scores['weighted_precision']:.4f} "
        f"wR={scores['weighted_recall']:.4f} wSens={scores['weighted_sensitivity']:.4f} "
        f"wSpec={scores['weighted_specificity']:.4f} wF1={scores['weighted_f1']:.4f} "
        f"acc={scores['accuracy']:.4f} kappa={scores['kappa']:.4f}"
    )
    print(format_prf_table(scores))
    print("wrote", dest)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge"))
    p.add_argument("--index", type=Path, default=None)
    p.add_argument("--num-classes", type=int, default=6, choices=[6, 10])
    p.add_argument("--mode", type=str, default="head", choices=["head", "full"])
    p.add_argument(
        "--init-ckpt",
        type=Path,
        default=None,
        help="Optional weights for head/full start (e.g. random-init head run or probe encoder)",
    )
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--plateau-patience", type=int, default=5)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--stats-patches", type=int, default=64)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--full-class-weights", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--monitor", type=str, default="weighted_f1", choices=["weighted_f1", "kappa", "mean_f1"])
    p.add_argument("--class-boost", type=str, default="")
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-val", type=int, default=None)
    p.add_argument(
        "--modality",
        type=str,
        default="both",
        choices=["both", "s2", "s1"],
        help="both=S2+S1 (12ch); s2=optical-only (10ch); s1=SAR-only (2ch) - paper-style ablation",
    )
    p.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu", "silu", "starrelu"])
    p.add_argument(
        "--loss",
        type=str,
        default="ce",
        choices=["ce", "ce_dice", "ce_gdl"],
        help="ce=weighted CE (quoted phases). ce_dice=Composite A. ce_gdl=Composite B",
    )
    p.add_argument("--dice-lambda", type=float, default=1.0, help="lambda on the Dice / GDL term")
    p.add_argument("--out-dir", type=Path, default=Path("multisenge_utae/checkpoints/run_c6_head_v0"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--eval-ckpt", type=Path, default=None)
    p.add_argument("--eval-split", type=str, default="test", choices=["train", "val", "test"])
    args = p.parse_args()

    n_cls = num_output_classes(args.num_classes)
    if args.index and args.index.is_file():
        records = records_from_json(args.index)
        print("loaded index", args.index, "n=", len(records))
    else:
        print("building index from", args.data_root)
        records = build_patch_index(args.data_root)

    if args.eval_ckpt:
        return _run_eval(args, records, n_cls)

    if args.seed is not None:
        set_seed(args.seed)
        print("seed", args.seed)

    print("estimating channel stats…")
    stats = estimate_channel_stats(records, args.num_classes, max_patches=args.stats_patches)

    train_ds = MultiSenGETemporalDataset(
        records,
        "train",
        num_classes=args.num_classes,
        augment=not args.no_augment,
        s1_mean=stats["s1_mean"],
        s1_std=stats["s1_std"],
        s2_mean=stats["s2_mean"],
        s2_std=stats["s2_std"],
    )
    val_ds = MultiSenGETemporalDataset(
        records,
        "val",
        num_classes=args.num_classes,
        augment=False,
        s1_mean=stats["s1_mean"],
        s1_std=stats["s1_std"],
        s2_mean=stats["s2_mean"],
        s2_std=stats["s2_std"],
    )
    if args.max_train:
        train_ds.records = train_ds.records[: args.max_train]
    if args.max_val:
        val_ds.records = val_ds.records[: args.max_val]

    loader_kw = dict(num_workers=args.workers, collate_fn=collate_utae)
    if args.seed is not None:
        g = torch.Generator()
        g.manual_seed(args.seed)
        loader_kw["generator"] = g
        loader_kw["worker_init_fn"] = _seed_worker
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kw)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_utae,
    )

    input_dim = modality_input_dim(args.modality)
    print(
        "modality", args.modality, "input_dim", input_dim,
        "train/val", len(train_ds), len(val_ds), "mode", args.mode,
        "activation", args.activation,
        "loss", args.loss, "dice_lambda", args.dice_lambda,
    )

    device = torch.device(args.device)
    model = build_model(input_dim, n_cls, args.mode, args.init_ckpt, device, activation=args.activation)

    if args.full_class_weights:
        print("estimating class weights (all train GR masks)…")
        counts = estimate_class_counts_from_gr(records, "train", args.num_classes)
    else:
        print("estimating class weights (train loader subset)…", flush=True)
        counts = estimate_class_counts(train_loader, n_cls, max_batches=80)
    print("counts", counts.tolist(), flush=True)
    weights = class_weights_from_counts(counts)
    boost = parse_class_boost(args.class_boost, n_cls)
    if np.any(boost != 1.0):
        weights = apply_class_boost(weights, boost)
    criterion = SegCompositeLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        ignore_index=255,
        mode=args.loss,
        dice_lambda=args.dice_lambda,
    )
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.1, patience=args.plateau_patience
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "norm_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    monitor_key = args.monitor
    history = []
    best_mon = -1.0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"epoch {epoch}/{args.epochs} train…", flush=True)
        loss = train_one_epoch(
            model, train_loader, opt, criterion, device,
            accum_steps=args.accum_steps, modality=args.modality,
        )
        print(f"epoch {epoch}/{args.epochs} val…", flush=True)
        val = evaluate(model, val_loader, device, n_cls, modality=args.modality)
        mon = float(val[monitor_key])
        scheduler.step(mon)
        row = {
            "epoch": epoch,
            "train_loss": loss,
            "val_weighted_f1": val["weighted_f1"],
            "val_weighted_precision": val["weighted_precision"],
            "val_weighted_recall": val["weighted_recall"],
            "val_weighted_sensitivity": val["weighted_sensitivity"],
            "val_weighted_specificity": val["weighted_specificity"],
            "val_accuracy": val["accuracy"],
            "val_kappa": val["kappa"],
            "val_mean_f1": val["mean_f1"],
            "lr": opt.param_groups[0]["lr"],
            "mode": args.mode,
            "modality": args.modality,
            "sec": round(time.time() - t0, 1),
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} loss={loss:.4f} val_wF1={val['weighted_f1']:.4f} "
            f"wSens={val['weighted_sensitivity']:.4f} wSpec={val['weighted_specificity']:.4f} "
            f"kappa={val['kappa']:.4f} mon={monitor_key}:{mon:.4f} lr={row['lr']:.1e}"
        )
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "num_classes": n_cls,
            "input_dim": input_dim,
            "modality": args.modality,
            "val": val,
            "norm_stats": stats,
            "mode": args.mode,
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        }
        torch.save(ckpt, args.out_dir / "last.pt")
        if mon > best_mon:
            best_mon = mon
            stale = 0
            torch.save(ckpt, args.out_dir / "best.pt")
            (args.out_dir / "best_metrics.json").write_text(json.dumps(val, indent=2), encoding="utf-8")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"EarlyStopping after {args.patience} epochs without val {monitor_key} improvement")
                break

    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("done. best val", monitor_key, best_mon, "dir", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
