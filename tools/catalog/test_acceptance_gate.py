#!/usr/bin/env python3
"""Unit tests for tools/catalog/acceptance_gate.py
(run: python tools/catalog/test_acceptance_gate.py)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.acceptance_gate import evaluate_acceptance  # noqa: E402

BASELINE = {"identity_recall": 0.90, "conflict": 5, "false_merges": 0, "suspicious_duplicate_targets": 2}


class TestEvaluateAcceptance(unittest.TestCase):
    def test_identical_report_is_accepted(self) -> None:
        result = evaluate_acceptance(BASELINE, dict(BASELINE))
        self.assertTrue(result.accepted)
        self.assertEqual(result.reasons, ())

    def test_improved_recall_with_no_other_regressions_is_accepted(self) -> None:
        candidate = {**BASELINE, "identity_recall": 0.95}
        result = evaluate_acceptance(BASELINE, candidate)
        self.assertTrue(result.accepted)

    def test_decreased_recall_is_rejected(self) -> None:
        candidate = {**BASELINE, "identity_recall": 0.85}
        result = evaluate_acceptance(BASELINE, candidate)
        self.assertFalse(result.accepted)
        self.assertTrue(any("identity_recall" in r for r in result.reasons))

    def test_recall_gain_with_more_conflicts_is_rejected(self) -> None:
        # The asymmetry rule: a recall win never buys an increase in
        # conflict/false_merges/suspicious_duplicate_targets.
        candidate = {**BASELINE, "identity_recall": 0.99, "conflict": 6}
        result = evaluate_acceptance(BASELINE, candidate)
        self.assertFalse(result.accepted)
        self.assertTrue(any("conflict" in r for r in result.reasons))

    def test_recall_gain_with_more_false_merges_is_rejected(self) -> None:
        candidate = {**BASELINE, "identity_recall": 0.99, "false_merges": 1}
        result = evaluate_acceptance(BASELINE, candidate)
        self.assertFalse(result.accepted)
        self.assertTrue(any("false_merges" in r for r in result.reasons))

    def test_recall_gain_with_more_suspicious_duplicates_is_rejected(self) -> None:
        candidate = {**BASELINE, "identity_recall": 0.99, "suspicious_duplicate_targets": 3}
        result = evaluate_acceptance(BASELINE, candidate)
        self.assertFalse(result.accepted)
        self.assertTrue(any("suspicious_duplicate_targets" in r for r in result.reasons))

    def test_small_recall_loss_with_fewer_false_merges_is_accepted(self) -> None:
        candidate = {**BASELINE, "identity_recall": 0.88, "false_merges": 0}
        # Recall only decreased, false_merges did not increase (unchanged
        # here since baseline already had 0) -- still accepted because no
        # gate condition fired; the "trade a small recall loss for fewer
        # false merges" framing only matters when recall *also* stays flat
        # or improves elsewhere, not as a magic exemption from the recall
        # check itself. This test documents that the recall check is a
        # hard floor, not a suggestion.
        result = evaluate_acceptance(BASELINE, candidate)
        self.assertFalse(result.accepted)

    def test_failed_unit_tests_is_rejected_even_with_perfect_metrics(self) -> None:
        result = evaluate_acceptance(BASELINE, dict(BASELINE), unit_tests_passed=False)
        self.assertFalse(result.accepted)
        self.assertIn("unit tests failed", result.reasons)

    def test_missing_keys_default_to_zero(self) -> None:
        # Lightweight per-iteration reports (no suspicious_duplicate_targets,
        # per Phase 4's "don't regenerate the full 50k output per iteration")
        # must not spuriously fail the gate just because a key is absent.
        baseline = {"identity_recall": 0.9, "conflict": 0}
        candidate = {"identity_recall": 0.92, "conflict": 0}
        result = evaluate_acceptance(baseline, candidate)
        self.assertTrue(result.accepted)


if __name__ == "__main__":
    unittest.main()
