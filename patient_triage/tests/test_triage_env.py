import unittest

from env.triage_env import TriageEnv


class TestTriageEnv(unittest.TestCase):
    def test_seed_reproducibility(self):
        env1 = TriageEnv(max_patients=4, seed=42, arrival_probability=0.0)
        env2 = TriageEnv(max_patients=4, seed=42, arrival_probability=0.0)

        s1 = env1.reset()
        s2 = env2.reset()

        self.assertEqual(s1["patients"], s2["patients"])

    def test_doctor_capacity_respected(self):
        env = TriageEnv(max_patients=5, doctors=2, seed=1, arrival_probability=0.0)
        state = env.reset()

        candidate_ids = [p["id"] for p in state["patients"][:3]]
        _, _, _, info = env.step({"patient_ids": candidate_ids})

        self.assertEqual(info["treated_this_step"], 2)

    def test_invalid_action_penalty(self):
        env = TriageEnv(max_patients=3, doctors=1, seed=2, arrival_probability=0.0)
        env.reset()

        _, reward, _, info = env.step({"wrong": 1})

        self.assertLess(reward, 0)
        self.assertEqual(info["invalid_action_reason"], "missing_or_malformed_action")

    def test_explicit_outcomes_include_deceased(self):
        env = TriageEnv(
            max_time=20,
            max_patients=1,
            doctors=0,
            seed=3,
            arrival_probability=0.0,
            deterioration_wait=0,
            deterioration_step=0.5,
            death_severity_threshold=0.5,
            death_wait_threshold=0,
        )
        env.reset()

        done = False
        while not done:
            state, _, done, _ = env.step({"patient_ids": []})

        statuses = {p["status"] for p in state["patients"]}
        self.assertIn("deceased", statuses)


if __name__ == "__main__":
    unittest.main()
