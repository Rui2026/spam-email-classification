#!/usr/bin/env python3
"""Convert a Kaggle spam CSV into the project format (columns: text,label).

Handles common variants:
    - SMS Spam Collection (uciml/sms-spam-collection-dataset): v1=label, v2=text
    - "Category, Message"
    - "Class, EmailText"
    - already-normalized "text, label"

Usage:
    python scripts/prepare_kaggle.py --input /path/to/spam.csv \
        --output data/raw/sms_spam.csv

Optional:
    --text-col / --label-col to override auto-detection
    --encoding (default: latin-1; SMS Spam Collection ships as latin-1)
    --dedup to drop duplicate rows after cleaning
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Common (text_col, label_col) candidates seen in Kaggle / public spam datasets.
KNOWN_COLUMN_PAIRS: list[tuple[str, str]] = [
    ("text", "label"),
    ("Message", "Category"),
    ("message", "category"),
    ("EmailText", "Class"),
    ("v2", "v1"),
    ("sms", "label"),
    ("body", "label"),
]


def auto_detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Return (text_col, label_col) using a list of known patterns; else heuristics."""
    cols = {c.lower(): c for c in df.columns}
    for text_c, label_c in KNOWN_COLUMN_PAIRS:
        if text_c.lower() in cols and label_c.lower() in cols:
            return cols[text_c.lower()], cols[label_c.lower()]

    # Heuristic fallback: short label column (few unique strings), the other long-ish text.
    candidates = []
    for c in df.columns:
        s = df[c].dropna().astype(str)
        if s.empty:
            continue
        nuniq = s.nunique()
        avg_len = s.str.len().mean()
        candidates.append((c, nuniq, avg_len))

    if len(candidates) < 2:
        raise ValueError(
            f"Could not detect text/label columns. Available columns: {list(df.columns)}. "
            f"Use --text-col and --label-col."
        )

    label_col = min(candidates, key=lambda t: t[1])[0]
    text_col = max((c for c in candidates if c[0] != label_col), key=lambda t: t[2])[0]
    return text_col, label_col


def normalize(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "text": df[text_col].astype(str).str.replace("\r", " ").str.replace("\n", " "),
            "label": df[label_col].astype(str).str.lower().str.strip(),
        }
    )
    out = out.dropna()
    out = out[out["text"].str.strip() != ""]

    # Map common label aliases to {ham, spam}; leave others as-is for utils.encode_labels.
    alias_to_spam = {"spam", "1", "true", "yes"}
    alias_to_ham = {"ham", "0", "false", "no", "not_spam", "legit", "legitimate"}
    def _norm(v: str) -> str:
        if v in alias_to_spam:
            return "spam"
        if v in alias_to_ham:
            return "ham"
        return v

    out["label"] = out["label"].map(_norm)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", "-i", type=Path, required=True, help="Path to the Kaggle CSV.")
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/raw/kaggle_spam.csv"),
        help="Where to write the normalized CSV (project root relative).",
    )
    p.add_argument("--text-col", default=None, help="Override text column name.")
    p.add_argument("--label-col", default=None, help="Override label column name.")
    p.add_argument("--encoding", default="latin-1", help="Source CSV encoding (default: latin-1).")
    p.add_argument("--dedup", action="store_true", help="Drop duplicate (text,label) rows.")
    args = p.parse_args()

    if not args.input.exists():
        print(f"[error] input not found: {args.input}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.input, encoding=args.encoding)
    print(f"[info] loaded {len(df)} rows; columns: {list(df.columns)}")

    if args.text_col and args.label_col:
        text_col, label_col = args.text_col, args.label_col
    else:
        text_col, label_col = auto_detect_columns(df)
        print(f"[info] auto-detected text='{text_col}', label='{label_col}'")

    out = normalize(df, text_col, label_col)
    if args.dedup:
        before = len(out)
        out = out.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
        print(f"[info] dedup: {before} -> {len(out)} rows")

    counts = out["label"].value_counts().to_dict()
    print(f"[info] label distribution: {counts}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"[info] wrote {len(out)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
