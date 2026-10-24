import unittest

from facenet_auth.experiment import run_experiment


class ExperimentTests(unittest.TestCase):
    def test_run_experiment_returns_summary(self):
        summary = run_experiment(tau_id=0.80, tau_pad=0.80, use_demo=True)
        self.assertIn("total_trials", summary)
        self.assertIn("selected_tau_id", summary)
        self.assertIn("selected_tau_pad", summary)
        self.assertGreater(summary["total_trials"], 0)


if __name__ == "__main__":
    unittest.main()
