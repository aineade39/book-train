#!/usr/bin/env python3
"""The acceptance gate (Phase 2d): the actual "loss function" for the
bounded matcher-improvement loop (Phase 4), made explicit and reusable
instead of left to a model's judgment call.

A candidate change is accepted only if, versus the baseline, on the *same*
fold:

- ``identity_recall`` does not decrease
- ``conflict`` (raw count, on ISBN-gold pairs) does not increase
- ``false_merges`` (adversarial negative set) does not increase
- ``suspicious_duplicate_targets`` (zero-label canary) does not increase
- all existing unit tests still pass

This is deliberately asymmetric, not a simple "did recall go up" check.
For this pipeline a false merge is worse than a miss:
``build_ios_en_from_goodreads.gap_fill_unmatched`` already inserts a
synthetic row carrying the *correct* title for any popular
``unmatched``/``ambiguous`` book, so a miss degrades gracefully. A false
merge does the opposite -- it transfers that book's popularity onto the
*wrong* OL work and seeds the OCR shortlist with a wrong title. A change
trading +1.0pt ``identity_recall`` for any increase in ``false_merges`` or
``suspicious_duplicate_targets`` is rejected; a change trading -0.2pt
``identity_recall`` for a measurable drop in either is acceptable.

``exact_recall`` is tracked for visibility only and never gates here -- see
``bibliographic_join.evaluate_title_author_against_isbn``'s docstring for
why it's noisier than ``identity_recall`` (it penalizes OL's own duplicate
work-key splits, which the matcher can't fix).

Missing keys in either report default to 0 -- this lets the same function
serve both the lightweight per-iteration check (identity_recall/conflict/
false_merges only, per Phase 4's "never regenerate the full 50k output
inside the loop" guardrail) and the fuller end-of-batch check that adds
``suspicious_duplicate_targets`` (which needs the full matched output).

Usage:
    python tools/catalog/acceptance_gate.py \\
        --baseline matcher_eval/<baseline>.json \\
        --candidate matcher_eval/<candidate>.json \\
        [--canaries-baseline canaries_baseline.json --canaries-candidate canaries_candidate.json] \\
        [--unit-tests-failed]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

RECALL_TOLERANCE = 1e-9


@dataclass(frozen=True)
class AcceptanceResult:
    accepted: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"accepted": self.accepted, "reasons": list(self.reasons)}


def evaluate_acceptance(
    baseline: dict[str, object],
    candidate: dict[str, object],
    *,
    unit_tests_passed: bool = True,
    recall_tolerance: float = RECALL_TOLERANCE,
) -> AcceptanceResult:
    reasons: list[str] = []

    if not unit_tests_passed:
        reasons.append("unit tests failed")

    base_recall = float(baseline.get("identity_recall", 0.0))
    cand_recall = float(candidate.get("identity_recall", 0.0))
    if cand_recall < base_recall - recall_tolerance:
        reasons.append(f"identity_recall decreased ({base_recall:.4f} -> {cand_recall:.4f})")

    base_conflict = int(baseline.get("conflict", 0))
    cand_conflict = int(candidate.get("conflict", 0))
    if cand_conflict > base_conflict:
        reasons.append(f"conflict increased ({base_conflict} -> {cand_conflict})")

    base_false_merges = int(baseline.get("false_merges", 0))
    cand_false_merges = int(candidate.get("false_merges", 0))
    if cand_false_merges > base_false_merges:
        reasons.append(f"false_merges increased ({base_false_merges} -> {cand_false_merges})")

    base_dup = int(baseline.get("suspicious_duplicate_targets", 0))
    cand_dup = int(candidate.get("suspicious_duplicate_targets", 0))
    if cand_dup > base_dup:
        reasons.append(f"suspicious_duplicate_targets increased ({base_dup} -> {cand_dup})")

    return AcceptanceResult(accepted=not reasons, reasons=tuple(reasons))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline eval report JSON")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate eval report JSON, same shape")
    parser.add_argument(
        "--canaries-baseline",
        type=Path,
        default=None,
        help="Optional eval_canaries.py --out report for the baseline (adds suspicious_duplicate_targets to the gate)",
    )
    parser.add_argument(
        "--canaries-candidate",
        type=Path,
        default=None,
        help="Optional eval_canaries.py --out report for the candidate",
    )
    parser.add_argument(
        "--unit-tests-failed", action="store_true", help="Pass if the candidate's unit test run failed"
    )
    args = parser.parse_args(argv)

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    if args.canaries_baseline:
        baseline = {**baseline, **json.loads(args.canaries_baseline.read_text(encoding="utf-8"))}
    if args.canaries_candidate:
        candidate = {**candidate, **json.loads(args.canaries_candidate.read_text(encoding="utf-8"))}

    result = evaluate_acceptance(baseline, candidate, unit_tests_passed=not args.unit_tests_failed)
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
