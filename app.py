import os

from flask import Flask, jsonify, request

from tasks.critical import make_env as make_critical
from tasks.easy import make_env as make_easy
from tasks.hard import make_env as make_hard
from tasks.medium import make_env as make_medium

app = Flask(__name__)

TASK_ENV_FACTORIES = {
    "easy": make_easy,
    "medium": make_medium,
    "hard": make_hard,
    "critical": make_critical,
}


@app.get("/")
def root():
    return jsonify({"status": "ok", "service": "patient-triage-env"}), 200


@app.get("/health")
def health():
    return jsonify({"status": "healthy"}), 200


@app.post("/reset")
def reset():
    payload = request.get_json(silent=True) or {}
    task = payload.get("task", "easy")

    if task not in TASK_ENV_FACTORIES:
        return jsonify({"error": f"unknown task '{task}'", "allowed_tasks": list(TASK_ENV_FACTORIES.keys())}), 400

    env = TASK_ENV_FACTORIES[task]()
    try:
        state = env.reset()
        return jsonify({"task": task, "observation": state}), 200
    finally:
        env.close()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    app.run(host="0.0.0.0", port=port)
