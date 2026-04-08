import os
import threading
import traceback
import uuid

from flask import Flask, jsonify, request

import inference
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

_jobs = {}
_jobs_lock = threading.Lock()


def _run_inference_job(job_id):
    try:
        with _jobs_lock:
            _jobs[job_id]["status"] = "running"

        results = inference.run()

        with _jobs_lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["results"] = results
    except Exception as exc:
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(exc)
            _jobs[job_id]["traceback"] = traceback.format_exc()


def _enqueue_inference_job(job_id=None):
    if job_id is None:
        job_id = str(uuid.uuid4())

    with _jobs_lock:
        if job_id in _jobs:
            return job_id
        _jobs[job_id] = {
            "status": "queued",
            "results": None,
            "error": None,
            "traceback": None,
        }

    thread = threading.Thread(target=_run_inference_job, args=(job_id,), daemon=True)
    thread.start()
    return job_id


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


@app.post("/run_inference")
def run_inference():
    job_id = _enqueue_inference_job()

    return jsonify({"job_id": job_id, "status": "queued"}), 202


@app.get("/run_inference/<job_id>")
def run_inference_status(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        return jsonify({"error": f"job '{job_id}' not found"}), 404

    response = {
        "job_id": job_id,
        "status": job["status"],
        "results": job["results"],
        "error": job["error"],
    }
    return jsonify(response), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    auto_run = os.getenv("RUN_INFERENCE_ON_STARTUP", "1").strip().lower()
    if auto_run in {"1", "true", "yes", "on"}:
        startup_job_id = _enqueue_inference_job(job_id="startup")
        print(f"[BOOT] queued inference job id={startup_job_id}", flush=True)
    app.run(host="0.0.0.0", port=port)
