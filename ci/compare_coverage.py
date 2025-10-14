#!/usr/bin/env python3
import argparse, sys, xml.etree.ElementTree as ET

def read_rate(path: str) -> float:
    root = ET.parse(path).getroot()
    covered = float(root.get("lines-covered", "0"))
    valid = float(root.get("lines-valid", "0"))
    if valid == 0:
        return 100.0
    return (covered / valid) * 100.0

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--current", required=True)
    p.add_argument("--mode", choices=["strict"], default="strict")
    args = p.parse_args()

    base = read_rate(args.base)
    cur = read_rate(args.current)
    delta = cur - base

    print(f"[coverage] base={base:.2f}%  current={cur:.2f}%  delta={delta:+.2f}%")
    if delta < -1e-6:
        print("::error ::Project coverage decreased vs base. Failing.", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
