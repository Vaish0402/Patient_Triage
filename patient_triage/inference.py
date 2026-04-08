import json
import os
import re
import time

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from tasks.easy import TASK_SPEC as EASY_SPEC
from tasks.easy import grade_episode as grade_easy
from tasks.easy import make_env as make_easy
from tasks.medium import TASK_SPEC as MEDIUM_SPEC
from tasks.medium import grade_episode as grade_medium
from tasks.medium import make_env as make_medium
from tasks.hard import TASK_SPEC as HARD_SPEC
from tasks.hard import grade_episode as grade_hard
from tasks.hard import make_env as make_hard
from tasks.critical import TASK_SPEC as CRITICAL_SPEC
from tasks.critical import grade_episode as grade_critical
from tasks.critical import make_env as make_critical

from openai import OpenAI

TASKS = [
    (EASY_SPEC, make_easy, grade_easy),
    (MEDIUM_SPEC, make_medium, grade_medium),
    (HARD_SPEC, make_hard, grade_hard),
    (CRITICAL_SPEC, make_critical, grade_critical),
]

BENCHMARK_NAME = "patient-triage-env"
SYSTEM_PROMPT = (
    "You are a hospital triage policy. Choose patients by jointly reasoning about severity and wait-time mortality risk. "
    "A moderate-severity patient with very long wait time may be more urgent than a high-severity patient who just arrived. "
    "Use a combined urgency view such as severity * 0.7 + wait_time * 0.3 (or similar), and use urgency_score as a key signal. "
    "Ignoring wait time causes preventable deaths. Return JSON only in this exact format: {\"patient_ids\": [int, ...]}. "
    "Select at most available_doctors patients."
)


def _load_env_file(path=".env"):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


if load_dotenv is not None:
    load_dotenv()
else:
    _load_env_file()


def _extract_waiting_patients(state):
    return [p for p in state["patients"] if p["status"] == "waiting"]


def _urgency_score(patient):
    return min(1.0, round(patient["severity"] + patient["wait_time"] * 0.15, 3))


def _urgency_fallback_action(state):
    waiting = _extract_waiting_patients(state)
    if not waiting:
        return {"patient_ids": []}

    selected = sorted(waiting, key=_urgency_score, reverse=True)[: state["available_doctors"]]
    return {"patient_ids": [p["id"] for p in selected]}


def _parse_action_response(text):
    action = json.loads(text)
    if not isinstance(action, dict):
        return None, "non_dict_action"

    ids = action.get("patient_ids", [])
    if not isinstance(ids, list):
        return None, "patient_ids_not_list"

    return {"patient_ids": ids}, None


def _create_completion_with_429_retry(client, model_name, messages):
    try:
        return client.chat.completions.create(
            model=model_name,
            temperature=0,
            timeout=30,
            messages=messages,
        )
    except Exception as exc:
        error_text = str(exc)
        if "429" not in error_text:
            raise

        match = re.search(r"Please try again in\s+([0-9]+(?:\.[0-9]+)?)\s+seconds", error_text, re.IGNORECASE)
        if match:
            retry_delay = float(match.group(1)) + 1.0
            time.sleep(retry_delay)
            return client.chat.completions.create(
                model=model_name,
                temperature=0,
                timeout=30,
                messages=messages,
            )
        raise


def _llm_action(client, model_name, task_spec, state):
    waiting = _extract_waiting_patients(state)
    if not waiting:
        return {"patient_ids": []}, None

    compact = [
        {
            "id": p["id"],
            "severity": p["severity"],
            "wait_time": p["wait_time"],
            "risk": p["risk"],
            "urgency_score": _urgency_score(p),
        }
        for p in waiting
    ]
    # Cap to top 10 by severity to keep prompt size manageable for smaller models
    compact = sorted(compact, key=lambda p: p["severity"], reverse=True)[:10]
    payload = {
        "task": task_spec["id"],
        "objective": task_spec["objective"],
        "time": state["time"],
        "available_doctors": state["available_doctors"],
        "patients": compact,
    }

    primary_messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": json.dumps(payload),
        },
    ]

    try:
        response = _create_completion_with_429_retry(client, model_name, primary_messages)
        time.sleep(12)
    except Exception as exc:
        return {"patient_ids": []}, str(exc)

    text = response.choices[0].message.content or "{}"
    try:
        action, parse_error = _parse_action_response(text)
        if parse_error is None and action.get("patient_ids"):
            return action, None
    except json.JSONDecodeError as exc:
        parse_error = str(exc)

    plain_list = "\n".join(f"{idx + 1}. id={p['id']}, urgency_score={p['urgency_score']}" for idx, p in enumerate(compact))
    retry_messages = [
        {
            "role": "system",
            "content": "Return JSON only in this exact format: {\"patient_ids\": [int, ...]}.",
        },
        {
            "role": "user",
            "content": (
                f"Pick top {state['available_doctors']} patient ids by urgency_score from this list:\n"
                f"{plain_list}\n"
                "Return only JSON with patient_ids."
            ),
        },
    ]

    try:
        retry_response = _create_completion_with_429_retry(client, model_name, retry_messages)
        time.sleep(12)
        retry_text = retry_response.choices[0].message.content or "{}"
        retry_action, retry_error = _parse_action_response(retry_text)
        if retry_error is None and retry_action.get("patient_ids"):
            return retry_action, None
        if retry_error is None:
            return retry_action, "empty_patient_ids"
        return {"patient_ids": []}, retry_error
    except json.JSONDecodeError as exc:
        return {"patient_ids": []}, str(exc)
    except Exception as exc:
        return {"patient_ids": []}, str(exc)


def _fmt_bool(value):
    return "true" if value else "false"


def _fmt_reward(value):
    return f"{value:.2f}"


def run_heuristic_baseline(task_spec, env_factory, grader):
    env = env_factory()
    state = env.reset()
    total_reward = 0.0

    while True:
        waiting = [p for p in state["patients"] if p["status"] == "waiting"]
        if not waiting:
            action = {"patient_ids": []}
        else:
            selected = sorted(waiting, key=lambda p: p["severity"], reverse=True)[: state["available_doctors"]]
            action = {"patient_ids": [p["id"] for p in selected]}

        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break

    score = grader(state, total_reward)
    env.close()
    return {
        "task": task_spec["id"],
        "difficulty": task_spec["difficulty"],
        "total_reward": round(total_reward, 4),
        "score": score,
        "termination": state["metrics"]["terminated_by"],
    }


def run_task(task_spec, env_factory, grader, client, model_name):
    env = env_factory()
    rewards = []
    step_idx = 0
    done = False
    success = False

    print(f"[START] task={task_spec['id']} env={BENCHMARK_NAME} model={model_name}")

    try:
        state = env.reset()
        while True:
            action, action_error = _llm_action(client, model_name, task_spec, state)
            waiting = _extract_waiting_patients(state)
            if action_error is not None or (waiting and not action.get("patient_ids")):
                action = _urgency_fallback_action(state)
                if action_error is None:
                    action_error = "empty_patient_ids"

            action_str = json.dumps(action, separators=(",", ":"))

            state, reward, done, info = env.step(action)
            step_idx += 1
            rewards.append(reward)

            last_action_error = info.get("invalid_action_reason")
            if last_action_error is None:
                last_action_error = action_error

            error_value = "null" if last_action_error is None else str(last_action_error)
            print(
                f"[STEP]  step={step_idx} action={action_str} reward={_fmt_reward(reward)} "
                f"done={_fmt_bool(done)} error={error_value}"
            )

            if done:
                break

        total_reward = sum(rewards)
        score = grader(state, total_reward)
        success = True

        return {
            "task": task_spec["id"],
            "difficulty": task_spec["difficulty"],
            "total_reward": round(total_reward, 4),
            "score": score,
            "termination": state["metrics"]["terminated_by"],
        }
    finally:
        env.close()
        rewards_str = ",".join(_fmt_reward(r) for r in rewards)
        print(f"[END]   success={_fmt_bool(success)} steps={step_idx} rewards={rewards_str}")


def run():
    api_base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if openai_api_key is None or openai_api_key.strip() == "":
        raise ValueError("OPENAI_API_KEY environment variable is required")

    client = OpenAI(api_key=openai_api_key, base_url=api_base_url)
    system_prompt = SYSTEM_PROMPT
    print("[CONFIG] system_prompt=", system_prompt[:120])

    results = []
    for spec, make_env, grader in TASKS:
        results.append(run_task(spec, make_env, grader, client, model_name))
        time.sleep(3)
    return results


if __name__ == "__main__":
    results = run()
    heuristic_results = [
        run_heuristic_baseline(spec, make_env, grader)
        for spec, make_env, grader in TASKS
    ]

    print("\n=== Baseline Results: LLM vs Heuristic ===")
    for llm, heur in zip(results, heuristic_results):
        improvement = round(llm["score"] - heur["score"], 4)
        sign = "+" if improvement >= 0 else ""
        print(f"\n{llm['task'].upper()} ({llm['difficulty']}):")
        print(f"  LLM:       score={llm['score']}, reward={llm['total_reward']}")
        print(f"  Heuristic: score={heur['score']}, reward={heur['total_reward']}")
        print(f"  Improvement: {sign}{improvement}")

    llm_agg = round(sum(r["score"] for r in results) / len(results), 4)
    heur_agg = round(sum(r["score"] for r in heuristic_results) / len(heuristic_results), 4)
    advantage = round(llm_agg - heur_agg, 4)
    sign = "+" if advantage >= 0 else ""
    print(f"\nAggregate Score:")
    print(f"  LLM:       {llm_agg}")
    print(f"  Heuristic: {heur_agg}")
    print(f"  LLM Advantage: {sign}{advantage}")