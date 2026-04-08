from env.triage_env import TriageEnv

TASK_SPEC = {
    "id": "medium",
    "difficulty": "medium",
    "objective": "Sustain treatment throughput under moderate arrivals while minimizing preventable deaths.",
}

def make_env():
    return TriageEnv(max_time=50, max_patients=8, doctors=2, seed=202, task_id="medium")


def grade_episode(final_state, total_reward):
    metrics = final_state["metrics"]
    arrivals = max(int(metrics["arrivals"]), 1)
    treated = int(metrics["treated"])
    deceased = int(metrics["deceased"])
    invalid_actions = int(metrics["invalid_actions"])

    treated_rate = min(1.0, treated / arrivals)
    mortality_penalty = min(1.0, deceased / arrivals)
    invalid_penalty = min(1.0, invalid_actions / (arrivals * 0.75 + 1))

    raw = 0.6 * treated_rate + 0.3 * (1.0 - mortality_penalty) + 0.1 * (1.0 - invalid_penalty)
    return round(max(0.0, min(1.0, raw)), 4)