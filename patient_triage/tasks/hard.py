from env.triage_env import TriageEnv

TASK_SPEC = {
    "id": "hard",
    "difficulty": "hard",
    "objective": "Manage a crisis surge with constrained doctors and high queue pressure without overload.",
}

def make_env():
    return TriageEnv(
        max_time=70,
        max_patients=12,
        doctors=3,
        seed=303,
        arrival_probability=0.22,
        deterioration_wait=3,
        deterioration_step=0.06,
        death_wait_threshold=4,
        task_id="hard",
    )


def grade_episode(final_state, total_reward):
    metrics = final_state["metrics"]
    arrivals = max(int(metrics["arrivals"]), 1)
    treated = int(metrics["treated"])
    deceased = int(metrics["deceased"])
    cumulative_wait = int(metrics["cumulative_wait"])
    terminated_by = metrics.get("terminated_by")

    treated_rate = min(1.0, treated / arrivals)
    mortality_penalty = min(1.0, deceased / arrivals)
    wait_penalty = min(1.0, cumulative_wait / (arrivals * 12.0))
    overload_penalty = 1.0 if terminated_by == "system_overload" else 0.0

    raw = (
        0.5 * treated_rate
        + 0.25 * (1.0 - mortality_penalty)
        + 0.2 * (1.0 - wait_penalty)
        + 0.05 * (1.0 - overload_penalty)
    )
    return round(max(0.0, min(1.0, raw)), 4)