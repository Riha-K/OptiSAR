"""Build / load a JSON patch index (avoids re-scanning labels every train start)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multisenge_seg.dataset import PatchRecord, build_patch_index, summarize_splits


def records_to_jsonable(records: list[PatchRecord]) -> list[dict]:
    out = []
    for r in records:
        out.append(
            {
                "patch_id": r.patch_id,
                "tile": r.tile,
                "split": r.split,
                "label_json": str(r.label_json),
                "gr_path": str(r.gr_path),
                "s2_by_month": {str(k): str(v) for k, v in r.s2_by_month.items()},
                "s1_by_month": {str(k): str(v) for k, v in r.s1_by_month.items()},
            }
        )
    return out


def records_from_json(path: Path) -> list[PatchRecord]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    records: list[PatchRecord] = []
    for row in raw["records"]:
        records.append(
            PatchRecord(
                patch_id=row["patch_id"],
                tile=row["tile"],
                split=row["split"],
                label_json=Path(row["label_json"]),
                gr_path=Path(row["gr_path"]),
                s2_by_month={int(k): Path(v) for k, v in row["s2_by_month"].items()},
                s1_by_month={int(k): Path(v) for k, v in row["s1_by_month"].items()},
            )
        )
    return records


def main() -> int:
    p = argparse.ArgumentParser(description="Cache MultiSenGE CNN validation patch index")
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("LULCDial-s1/data/lulcdial_s1/ai4lcc/multisenge"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("multisenge_seg/artifacts/patch_index.json"),
    )
    args = p.parse_args()
    records = build_patch_index(args.data_root)
    summary = summarize_splits(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data_root": str(args.data_root.resolve()),
        "summary": summary,
        "num_records": len(records),
        "records": records_to_jsonable(records),
    }
    args.out.write_text(json.dumps(payload), encoding="utf-8")
    print("wrote", args.out, "n=", len(records), summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
