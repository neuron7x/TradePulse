# SPDX-License-Identifier: MIT
import sys, os, glob, pandas as pd
from pathlib import Path

def analyze_csv(path: str):
    df = pd.read_csv(path)
    report = []
    report.append(f"# File: {path}")
    report.append(f"- rows: {len(df)}; cols: {len(df.columns)}")
    na = df.isna().mean().mean()
    report.append(f"- NaN ratio (avg): {na:.4f}")
    if "ts" in df.columns:
        ts = pd.to_datetime(df["ts"], errors="coerce")
        gaps = ts.diff().dt.total_seconds().dropna()
        if len(gaps)>0:
            report.append(f"- median gap (s): {gaps.median():.3f}; max gap (s): {gaps.max():.3f}")
    dups = df.duplicated().sum()
    report.append(f"- duplicates: {dups}")
    return "\n".join(report)

def main():
    root = Path("data")
    out = []
    if not root.exists():
        print("No data/ directory — nothing to check (OK).")
        return 0
    csvs = glob.glob("data/**/*.csv", recursive=True)
    if not csvs:
        print("No CSV files in data/ — nothing to check (OK).")
        return 0
    for c in csvs:
        try:
            out.append(analyze_csv(c))
        except Exception as e:
            out.append(f"# File: {c}\n- ERROR: {e}")
    print("\n\n".join(out))
    return 0

if __name__ == "__main__":
    sys.exit(main())
