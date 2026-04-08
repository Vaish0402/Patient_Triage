from env.triage_env import TriageEnv

TASK_SPEC = {
    "id": "easy",
    "difficulty": "easy",
    "objective": "Prioritize urgent patients while keeping average wait low in a low-load clinic.",
}

def make_env():
    return TriageEnv(max_time=30, max_patients=5, doctors=1, seed=101, task_id="easy")


def grade_episode(final_state, total_reward):
    metrics = final_state["metrics"]
    arrivals = max(int(metrics["arrivals"]), 1)
    treated = int(metrics["treated"])
    deceased = int(metrics["deceased"])
    cumulative_wait = int(metrics["cumulative_wait"])

    treated_rate = min(1.0, treated / arrivals)
    mortality_penalty = min(1.0, deceased / arrivals)
    wait_penalty = min(1.0, cumulative_wait / (arrivals * 8.0))

    raw = 0.65 * treated_rate + 0.2 * (1.0 - wait_penalty) + 0.15 * (1.0 - mortality_penalty)
    return round(max(0.0, min(1.0, raw)), 4)