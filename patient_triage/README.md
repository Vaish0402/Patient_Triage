# patient-triage-env

An OpenEnv-compliant reinforcement learning environment for hospital patient triage optimization. Agents must learn to allocate limited medical staff to incoming patients by jointly reasoning about clinical severity and wait-time mortality risk.

---

## Motivation

Hospital triage is a high-stakes, real-world decision problem: with more patients than available doctors, every allocation decision carries life-or-death consequences. A purely severity-based policy fails when low-severity patients deteriorate after long waits. This environment models that tension, requiring agents to develop nuanced triage policies that balance urgency, throughput, and mortality prevention — tasks that are genuinely hard to solve with simple heuristics.

---

## Environment Description

The environment simulates a hospital emergency department over discrete time steps. At each step:

- New patients may arrive with randomized severity scores
- The agent selects which waiting patients to assign to available doctors
- Patients deteriorate over time if untreated; severe deterioration leads to death
- The episode ends when all patients are processed or the time limit is reached

The core challenge: a high-severity new arrival may seem more urgent than a low-severity patient who has been waiting for 8 steps — but the latter may be minutes from dying. The optimal policy must weigh both dimensions.

---

## Action Space

```
Action: { "patient_ids": [int, ...] }
```

A list of patient IDs to assign to available doctors this step. The number of selected patients must not exceed `available_doctors`. Submitting more IDs than available doctors results in only the first N being treated. Submitting an empty list or a malformed action incurs a penalty reward.

---

## Observation Space

Each step returns a state dictionary with the following fields:

| Field | Type | Description |
|---|---|---|
| `time` | int | Current time step |
| `available_doctors` | int | Doctors free to treat patients this step |
| `patients` | list | All patients (waiting, in-treatment, treated, deceased) |

Each patient object contains:

| Field | Type | Description |
|---|---|---|
| `id` | int | Unique patient identifier |
| `severity` | float (0–1) | Clinical severity score at arrival |
| `wait_time` | int | Steps spent waiting without treatment |
| `risk` | float (0–1) | Current deterioration risk score |
| `urgency_score` | float (0–1) | Combined urgency: `min(1.0, severity + wait_time * 0.15)` |
| `status` | str | One of: `waiting`, `in-treatment`, `treated`, `deceased` |

---

## Reward Function

The reward function provides dense signal across the full trajectory — not just at episode end.

- **Positive reward** per patient treated, scaled by severity and wait time
- **Negative reward** for invalid actions (malformed or empty when patients are waiting)
- **Negative reward** when a patient dies (mortality penalty)
- **Cumulative wait penalty** applied over the episode

This ensures agents receive learning signal at every step rather than a sparse end-of-episode score.

---

## Tasks

### Easy
**Objective:** Prioritize urgent patients while keeping average wait low in a low-load clinic.

- `max_time=30`, `max_patients=5`, `doctors=1`, `seed=101`
- Small queue, single doctor, straightforward severity-based decisions
- Expected difficulty: solvable by a greedy severity heuristic
- **Baseline score: 0.9406**

### Medium
**Objective:** Sustain treatment throughput under moderate arrivals while minimizing preventable deaths.

- `max_time=50`, `max_patients=8`, `doctors=2`, `seed=202`
- Two doctors, larger queue, begins to require throughput management
- Invalid action penalty introduced to discourage random policies
- **Baseline score: 1.0**

### Hard
**Objective:** Manage a crisis surge with constrained doctors and high queue pressure without overload.

- `max_time=70`, `max_patients=12`, `doctors=3`, `seed=303`, `arrival_probability=0.22`
- High arrival rate, large queue, system overload penalty active
- Requires balancing throughput against mortality risk under sustained pressure
- **Baseline score: 0.975**

### Critical
**Objective:** Manage mass casualty crisis with triage conflicts — low-severity high-wait patients will die unless treated immediately.

- `max_time=60`, `max_patients=18`, `doctors=2`, `seed=404`, `arrival_probability=0.30`
- High arrival rate, tight doctor constraint, rapid deterioration
- Core challenge: pure severity-greedy policies fail because long-wait low-severity patients die
- Requires genuine urgency reasoning combining severity + wait time
- **Baseline score: 0.767**

---

## Grader Details

All graders return a score in `[0.0, 1.0]` and are fully deterministic given the same seed.

| Metric | Easy | Medium | Hard | Critical |
|---|---|---|---|---|
| Treated rate weight | 0.65 | 0.60 | 0.50 | 0.40 |
| Mortality penalty weight | 0.15 | 0.30 | 0.25 | 0.35 |
| Wait penalty weight | 0.20 | — | 0.20 | 0.25 |
| Invalid action penalty | — | 0.10 | — | — |
| Overload penalty | — | — | 0.05 | — |

---

## Baseline Results

Model: `Qwen/Qwen2.5-7B-Instruct` via HuggingFace Inference API

| Task | Difficulty | LLM Score | Heuristic Score | LLM Reward | Heuristic Reward |
|---|---|---|---|---|---|
| easy | Easy | 0.9406 | 0.9406 | 14.05 | 14.05 |
| medium | Medium | 1.0 | 1.0 | 18.90 | 18.90 |
| hard | Hard | 0.975 | 0.975 | 31.95 | 31.95 |
| critical | Critical | 0.767 | 0.7659 | 33.25 | 34.70 |
| **Aggregate** | | **0.9206** | **0.9204** | | |

The LLM marginally outperforms the heuristic on aggregate (+0.0002), with the critical task being the primary differentiator where urgency-aware reasoning is required.

---

## Project Structure

```
patient_triage/
├── env/
│   └── triage_env.py        # Core TriageEnv, Observation, Action, Reward models
├── tasks/
│   ├── easy.py              # Easy task spec, make_env, grade_episode
│   ├── medium.py            # Medium task spec, make_env, grade_episode
│   ├── hard.py              # Hard task spec, make_env, grade_episode
│   └── critical.py          # Critical task spec, make_env, grade_episode
├── inference.py             # Baseline LLM inference script
├── test_triage_env.py       # Unit tests
├── openenv.yaml             # OpenEnv spec metadata
├── Dockerfile               # Container definition
└── README.md
```

---

## Setup & Usage

### Prerequisites
- Python 3.10+
- Docker (for containerized execution)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Baseline Inference

```bash
export OPENAI_API_KEY=your_api_key_here
export API_BASE_URL=https://api.groq.com/openai/v1   # or any OpenAI-compatible endpoint
export MODEL_NAME=llama-3.1-8b-instant               # or any compatible model

python inference.py
```

### Run Tests

```bash
python -m pytest test_triage_env.py -v
```

### Docker

```bash
docker build -t patient-triage-env .
docker run -e OPENAI_API_KEY=your_key -e API_BASE_URL=your_url -e MODEL_NAME=your_model patient-triage-env
```

### OpenEnv Validation

```bash
openenv validate
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | API key for the inference provider |
| `API_BASE_URL` | No | Base URL for OpenAI-compatible API (default: HuggingFace router) |
| `MODEL_NAME` | No | Model to use for baseline inference (default: `Qwen/Qwen2.5-7B-Instruct`) |

---

## OpenEnv Spec Compliance

| Requirement | Status |
|---|---|
| Typed `Observation`, `Action`, `Reward` Pydantic models | ✅ |
| `reset()` → initial observation | ✅ |
| `step(action)` → observation, reward, done, info | ✅ |
| `state()` → current state | ✅ |
| `openenv.yaml` with metadata | ✅ |
| Minimum 3 tasks with graders | ✅ (4 tasks) |
| Graders return 0.0–1.0 | ✅ |
| Deterministic with fixed seed | ✅ |
| Baseline inference script | ✅ |
| Dockerfile | ✅ |