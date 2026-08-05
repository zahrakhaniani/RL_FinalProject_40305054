"""Tests for the transfer scenarios and the logging they depend on."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from environments.generator import MazeGenerator
from environments.variants import make_variant, unchanged_neighbourhood_mask
from transfer.transfer_learning import build_initial_q, classify_transfer, scenario_specs

CONFIG = {
    "transfer": {
        "scenarios": ["scratch", "full", "scaled", "selective"],
        "betas": [0.25, 0.5, 0.75],
    }
}


class TestScenarioSpecs(unittest.TestCase):
    def test_scaled_family_expands_over_betas(self):
        names = [name for name, _ in scenario_specs(CONFIG)]
        self.assertEqual(
            names,
            ["scratch", "full", "scaled_0.25", "scaled_0.5", "scaled_0.75", "selective"],
        )

    def test_every_spec_carries_its_family(self):
        for name, options in scenario_specs(CONFIG):
            with self.subTest(scenario=name):
                self.assertIn(options["family"], ("scratch", "full", "scaled", "selective"))


class TestInitialQ(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = MazeGenerator(student_id="40305054").generate(reward_mode="shaped")
        target = make_variant(source, "similar", seed=87_654_321)
        cls.mask = unchanged_neighbourhood_mask(source, target)
        rng = np.random.default_rng(0)
        cls.source_q = rng.normal(size=(source.rows, source.cols, 2, 4, 4))

    def build(self, scenario, **options):
        return build_initial_q(scenario, options, self.source_q, self.mask)

    def test_scratch_starts_empty(self):
        self.assertEqual(np.count_nonzero(self.build("scratch", family="scratch")), 0)

    def test_full_transfer_copies_everything(self):
        np.testing.assert_array_equal(
            self.build("full", family="full"), self.source_q
        )

    def test_full_transfer_does_not_alias_the_source(self):
        initial = self.build("full", family="full")
        initial[0, 0, 0, 0, 0] += 1.0
        self.assertNotEqual(initial[0, 0, 0, 0, 0], self.source_q[0, 0, 0, 0, 0])

    def test_scaled_transfer_shrinks_by_beta(self):
        for beta in (0.25, 0.5, 0.75):
            with self.subTest(beta=beta):
                np.testing.assert_allclose(
                    self.build("scaled", family="scaled", beta=beta),
                    self.source_q * beta,
                )

    def test_selective_transfer_only_keeps_unchanged_neighbourhoods(self):
        initial = self.build("selective", family="selective")
        np.testing.assert_array_equal(initial[self.mask], self.source_q[self.mask])
        self.assertEqual(np.count_nonzero(initial[~self.mask]), 0)

    def test_selective_transfer_is_between_scratch_and_full(self):
        initial = self.build("selective", family="selective")
        self.assertGreater(np.count_nonzero(initial), 0)
        self.assertLess(np.count_nonzero(initial), self.source_q.size)

    def test_unknown_scenario_is_rejected(self):
        with self.assertRaises(KeyError):
            self.build("teleport", family="teleport")


class TestClassification(unittest.TestCase):
    baseline = {
        "zero_shot_success": 0.0,
        "early_success": 0.30,
        "final_success": 0.80,
        "episodes_to_threshold": 900,
        "episodes": 1500,
    }

    def classify(self, **overrides):
        record = {**self.baseline, **overrides}
        return classify_transfer(record, self.baseline)["verdict"]

    def test_matching_the_baseline_is_neutral(self):
        self.assertEqual(self.classify(), "neutral")

    def test_faster_early_learning_is_positive(self):
        self.assertEqual(self.classify(early_success=0.60), "positive")

    def test_higher_final_performance_is_positive(self):
        self.assertEqual(self.classify(final_success=0.95), "positive")

    def test_worse_final_performance_is_negative(self):
        self.assertEqual(self.classify(final_success=0.50), "negative")

    def test_slower_early_learning_is_negative(self):
        self.assertEqual(self.classify(early_success=0.05), "negative")

    def test_a_worse_ending_outweighs_a_better_start(self):
        """A head start that ends up worse than scratch is still negative transfer."""
        self.assertEqual(
            self.classify(early_success=0.70, final_success=0.40), "negative"
        )

    def test_never_reaching_the_threshold_counts_as_slower(self):
        record = {**self.baseline, "episodes_to_threshold": None}
        self.assertLess(classify_transfer(record, self.baseline)["speed_delta"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
