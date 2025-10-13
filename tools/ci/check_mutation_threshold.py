#!/usr/bin/env python3
"""Mutation score gate for CI pipelines.

The script reads the metadata emitted by mutmut (``mutants/*.meta``) and
computes the mutation score.  If the score drops below the configured
threshold the process exits with a non-zero status so CI can fail fast.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

STATUS_BY_EXIT_CODE: Mapping[int | None, str] = {
    1: "killed",
    3: "killed",
    -24: "timeout",
    24: "timeout",
    33: "no tests",
    34: "skipped",
    35: "suspicious",
    36: "timeout",
    152: "timeout",
    255: "timeout",
    0: "survived",
    2: "interrupted",
    5: "no tests",
    -11: "segfault",
    None: "not checked",
}

SUCCESS_STATUSES = {"killed"}
FAILURE_STATUSES = {
    "timeout",
    "interrupted",
    "not checked",
    "segfault",
    "unknown",
}


@dataclass
class MutationSummary:
    counts: Counter[str]

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def killed(self) -> int:
        return sum(self.counts[status] for status in SUCCESS_STATUSES)

    @property
    def survived(self) -> int:
        return self.total - self.killed

    @property
    def score(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.killed / self.total) * 100

    def as_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "total": self.total,
            "killed": self.killed,
            "survived": self.survived,
            "status_breakdown": dict(sorted(self.counts.items())),
        }


def iter_exit_codes(meta_files: Iterable[Path]) -> Iterable[int | None]:
    for meta_file in meta_files:
        with meta_file.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        exit_codes = meta.get("exit_code_by_key", {})
        for value in exit_codes.values():
            if value is None or value == "None":
                yield None
                continue
            try:
                yield int(value)
            except (TypeError, ValueError):
                # mutmut may already store the textual status; surface it directly.
                yield value if isinstance(value, str) else None


def build_summary(exit_codes: Iterable[int | None]) -> MutationSummary:
    counts: Counter[str] = Counter()
    for code in exit_codes:
        status = STATUS_BY_EXIT_CODE.get(code)
        if status is None:
            # fall back to textual status if mutmut ever switches to that format
            status = str(code) if isinstance(code, str) else "unknown"
        counts[status] += 1
    return MutationSummary(counts=counts)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enforce minimum mutation score")
    parser.add_argument(
        "--mutants-dir",
        default="mutants",
        type=Path,
        help="Directory that stores mutmut metadata (*.meta)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Minimum acceptable mutation score (percentage)",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Optional path to dump a JSON payload with the collected statistics.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    mutants_dir: Path = args.mutants_dir

    if not mutants_dir.exists():
        print(f"[mutation-gate] No mutants directory found at '{mutants_dir}'.", file=sys.stderr)
        return 1

    meta_files = sorted(mutants_dir.glob("**/*.meta"))
    if not meta_files:
        print(
            f"[mutation-gate] No mutation metadata files (*.meta) located in '{mutants_dir}'.",
            file=sys.stderr,
        )
        return 1

    summary = build_summary(iter_exit_codes(meta_files))
    score = summary.score

    print("[mutation-gate] Mutation testing summary:")
    for status, count in sorted(summary.counts.items()):
        print(f"  - {status:>12}: {count}")
    print(f"  => killed: {summary.killed} / total: {summary.total} => score: {score:.2f}%")

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_json.open("w", encoding="utf-8") as fh:
            json.dump(summary.as_dict(), fh, indent=2)

    if summary.total == 0:
        print("[mutation-gate] No mutants were evaluated. Treating this as a failure.", file=sys.stderr)
        return 1

    if score < args.threshold:
        print(
            (
                f"[mutation-gate] Mutation score {score:.2f}% fell below the required"
                f" threshold of {args.threshold:.2f}%."
            ),
            file=sys.stderr,
        )
        return 1

    failing_statuses = summary.counts.keys() & FAILURE_STATUSES
    critical_failures = sum(summary.counts[status] for status in failing_statuses)
    if critical_failures:
        print(
            (
                "[mutation-gate] Critical failures detected during mutation testing."
                " Investigate the statuses above before merging."
            ),
            file=sys.stderr,
        )
        return 1

    print("[mutation-gate] Mutation score gate satisfied.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
