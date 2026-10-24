import unittest

from facenet_auth.evaluation import (
    evaluate_trial,
    summarize_trial_results,
    threshold_search,
)


class EvaluationTests(unittest.TestCase):
    def test_identity_and_pad_both_needed(self):
        self.assertTrue(evaluate_trial(0.90, 0.90, 0.80, 0.80))
        self.assertFalse(evaluate_trial(0.90, 0.40, 0.80, 0.80))
        self.assertFalse(evaluate_trial(0.70, 0.90, 0.80, 0.80))

    def test_threshold_search_finds_reasonable_cutoff(self):
        genuine_scores = [0.90, 0.92, 0.94, 0.88]
        impostor_scores = [0.40, 0.45, 0.50, 0.42]
        tau_id = threshold_search(
            genuine_scores,
            impostor_scores,
            min_score=0.0,
            max_score=1.0,
            step=0.01,
        )
        self.assertGreaterEqual(tau_id, 0.6)
        self.assertLessEqual(tau_id, 0.9)

    def test_summary_tracks_counts(self):
        results = [
            {"identity_score": 0.95, "pad_score": 0.90, "is_attack": False, "accepted": True},
            {"identity_score": 0.82, "pad_score": 0.85, "is_attack": False, "accepted": True},
            {"identity_score": 0.92, "pad_score": 0.65, "is_attack": True, "accepted": False},
            {"identity_score": 0.97, "pad_score": 0.92, "is_attack": True, "accepted": True},
        ]
        summary = summarize_trial_results(results)
        self.assertEqual(summary["total_trials"], 4)
        self.assertEqual(summary["accepted_trials"], 3)
        self.assertEqual(summary["attack_successes"], 1)
        self.assertEqual(summary["genuine_accepts"], 2)


if __name__ == "__main__":
    unittest.main()
