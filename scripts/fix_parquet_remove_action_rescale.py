#!/usr/bin/env python3
"""Remove action.rescale column from all parquet files under a data directory."""
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def iter_parquet_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.parquet") if p.is_file())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--column", default="action.rescale")
    ap.add_argument("--backup", action="store_true", help="create .bak copy before overwrite")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = args.data_dir
    if not root.exists():
        raise SystemExit(f"Not found: {root}")

    files = iter_parquet_files(root)
    if not files:
        print(f"No parquet files under {root}")
        return 0

    modified = 0
    scanned = 0

    for fpath in files:
        scanned += 1
        try:
            table = pq.read_table(fpath)
        except Exception as exc:
            print(f"[ERROR] failed to read {fpath}: {exc}")
            continue

        if args.column not in table.schema.names:
            continue

        new_table = table.drop([args.column])
        modified += 1

        if args.dry_run:
            continue

        tmp_path = fpath.with_suffix(fpath.suffix + ".tmp")
        pq.write_table(new_table, tmp_path)

        if args.backup:
            bak_path = fpath.with_suffix(fpath.suffix + ".bak")
            if not bak_path.exists():
                fpath.replace(bak_path)
            else:
                # Avoid clobbering existing backups.
                i = 1
                while True:
                    cand = fpath.with_suffix(fpath.suffix + f".bak{i}")
                    if not cand.exists():
                        fpath.replace(cand)
                        break
                    i += 1
        else:
            fpath.unlink()

        tmp_path.replace(fpath)

    print(f"scanned={scanned} modified={modified} column={args.column}")
    if args.dry_run:
        print("dry-run: no files modified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
