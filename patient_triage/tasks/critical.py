# from env.triage_env import TriageEnv

# TASK_SPEC = {
#     "id": "critical",
#     "difficulty": "critical",
#     "objective": "Manage mass casualty crisis with triage conflicts: some low-severity high-wait patients will die unless treated immediately.",
# }

# def make_env():
#     return TriageEnv(
#         max_time=40,
#         max_patients=20,
#         doctors=2,
#         seed=404,
#         arrival_probability=0.30,
#         deterioration_wait=2,
#         deterioration_step=0.08,
#         death_severity_threshold=0.85,
#         death_wait_threshold=3,
#         critical_threshold=0.7,
#         task_id="critical",
#     )


# def grade_episode(final_state, total_reward):
#     metrics = final_state["metrics"]
#     arrivals = max(int(metrics["arrivals"]), 1)
#     treated = int(metrics["treated"])
#     deceased = int(metrics["deceased"])
#     cumulative_wait = int(metrics["cumulative_wait"])

#     treated_rate = min(1.0, treated / arrivals)
#     mortality_penalty = min(1.0, deceased / arrivals)
#     wait_penalty = min(1.0, cumulative_wait / (arrivals * 10.0))

#     raw = (
#         0.4 * treated_rate
#         + 0.35 * (1.0 - mortality_penalty)
#         + 0.25 * (1.0 - wait_penalty)
#     )
#     return round(max(0.0, min(1.0, raw)), 4)

from env.triage_env import TriageEnv

TASK_SPEC = {
    "id": "critical",
    "difficulty": "critical",
    "objective": "Manage mass casualty crisis with triage conflicts: some low-severity high-wait patients will die unless treated immediately.",
}

def make_env():
    return TriageEnv(
        max_time=60,
        max_patients=18,
        doctors=2,
        seed=404,
        arrival_probability=0.30,
        deterioration_wait=2,
        deterioration_step=0.08,
        death_severity_threshold=0.85,
        death_wait_threshold=3,
        critical_threshold=0.7,
        task_id="critical",
    )


def grade_episode(final_state, total_reward):
    metrics = final_state["metrics"]
    arrivals = max(int(metrics["arrivals"]), 1)
    treated = int(metrics["treated"])
    deceased = int(metrics["deceased"])
    cumulative_wait = int(metrics["cumulative_wait"])

    treated_rate = min(1.0, treated / arrivals)
    mortality_penalty = min(1.0, deceased / arrivals)
    wait_penalty = min(1.0, cumulative_wait / (arrivals * 10.0))

    raw = (
        0.4 * treated_rate
        + 0.35 * (1.0 - mortality_penalty)
        + 0.25 * (1.0 - wait_penalty)
    )
    return round(max(0.0, min(1.0, raw)), 4)