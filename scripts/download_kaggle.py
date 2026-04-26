#!/usr/bin/env python3
"""Download a Kaggle dataset via kagglehub and normalize the first CSV
into the project format (data/raw/<name>.csv with columns text,label).

Usage:
    python scripts/download_kaggle.py \
        --slug abdmental01/email-spam-dedection \
        --output data/raw/email_spam.csv

Optional:
    --csv-name SPECIFIC.csv     Pick a specific CSV when the dataset has many.
    --encoding latin-1          Override CSV encoding.
    --text-col / --label-col    Force column names (skip auto-detect).
    --no-dedup                  Keep duplicates.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--slug", required=True, help="Kaggle dataset slug, e.g. 'user/dataset-name'.")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/kaggle_spam.csv"),
        help="Where to write the normalized CSV.",
    )
    p.add_argument("--csv-name", default=None, help="Specific filename inside the dataset.")
    p.add_argument("--encoding", default="utf-8", help="Source CSV encoding (default: utf-8).")
    p.add_argument("--text-col", default=None)
    p.add_argument("--label-col", default=None)
    p.add_argument("--no-dedup", action="store_true", help="Keep duplicate rows.")
    args = p.parse_args()

    try:
        import kagglehub
    except ImportError:
        print(
            "[error] kagglehub not installed. Run: pip install kagglehub",
            file=sys.stderr,
        )
        return 1

    print(f"[info] downloading '{args.slug}' via kagglehub ...")
    download_dir = Path(kagglehub.dataset_download(args.slug))
    print(f"[info] kagglehub cache: {download_dir}")

    csvs = sorted(download_dir.rglob("*.csv"))
    if not csvs:
        print(f"[error] no CSV files found under {download_dir}", file=sys.stderr)
        return 2

    if args.csv_name:
        match = [c for c in csvs if c.name == args.csv_name]
        if not match:
            print(
                f"[error] csv-name '{args.csv_name}' not in {[c.name for c in csvs]}",
                file=sys.stderr,
            )
            return 3
        chosen = match[0]
    elif len(csvs) > 1:
        print(f"[info] multiple CSVs found: {[c.name for c in csvs]}")
        print(f"[info] picking first: {csvs[0].name} (use --csv-name to override)")
        chosen = csvs[0]
    else:
        chosen = csvs[0]
    print(f"[info] using CSV: {chosen}")

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "prepare_kaggle.py"),
        "--input",
        str(chosen),
        "--output",
        str(args.output),
        "--encoding",
        args.encoding,
    ]
    if not args.no_dedup:
        cmd.append("--dedup")
    if args.text_col:
        cmd += ["--text-col", args.text_col]
    if args.label_col:
        cmd += ["--label-col", args.label_col]

    print(f"[info] running: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
