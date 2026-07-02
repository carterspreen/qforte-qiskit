"""Stage 3: retrieve IBM Runtime Sampler counts for a submitted UQK run.

Manual run:
  conda activate qiskit_env_v1
  python hardware_runs/3_retrieve.py dry_run

This script only needs saved job IDs. The original submit process does not need
to stay alive while IBM hardware jobs are queued or running.
"""

from __future__ import annotations

from hardware_common import (
    get_profile,
    get_run_id,
    load_json,
    load_runtime_service,
    now_utc,
    retrieve_dir,
    retrieval_summary_path,
    sampler_pub_counts,
    save_json,
    submitted_jobs_path,
)


def status_name(status):
    return getattr(status, "name", str(status))


def job_result_is_ready(status):
    name = status_name(status).upper()
    return name in {"DONE", "COMPLETED"}


def main():
    profile = get_profile()
    run_id = get_run_id(profile)
    submitted = load_json(submitted_jobs_path(run_id))
    service = load_runtime_service()
    jobs_dir = retrieve_dir(run_id) / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "schema_version": 1,
        "generated_at_utc": now_utc(),
        "profile": profile,
        "run_id": run_id,
        "submitted_jobs_path": str(submitted_jobs_path(run_id)),
        "jobs": [],
    }

    for job_record in submitted["jobs"]:
        job_id = job_record["job_id"]
        job = service.job(job_id)
        status = status_name(job.status())
        retrieved_record = {
            **job_record,
            "retrieved_at_utc": now_utc(),
            "status_at_retrieve": status,
            "result_file": None,
            "result_ready": False,
        }
        print("Job", job_id, "status", status)
        if job_result_is_ready(job.status()):
            result = job.result()
            counts_by_circuit = []
            for pub_index, circuit_index in enumerate(job_record["circuit_indices"]):
                counts_by_circuit.append(
                    {
                        "pub_index": int(pub_index),
                        "circuit_index": int(circuit_index),
                        "counts": sampler_pub_counts(result[pub_index]),
                    }
                )
            result_payload = {
                "schema_version": 1,
                "job_id": job_id,
                "retrieved_at_utc": now_utc(),
                "job_group_id": job_record["job_group_id"],
                "point_id": job_record["point_id"],
                "shots": job_record["shots"],
                "circuit_indices": job_record["circuit_indices"],
                "counts_by_circuit": counts_by_circuit,
            }
            result_file = jobs_dir / f"{job_record['job_group_id']}_{job_id}.json"
            save_json(result_file, result_payload)
            retrieved_record["result_file"] = str(result_file)
            retrieved_record["result_ready"] = True
        summary["jobs"].append(retrieved_record)

    save_json(retrieval_summary_path(run_id), summary)
    ready = sum(1 for job in summary["jobs"] if job["result_ready"])
    print("Retrieved ready jobs:", ready, "/", len(summary["jobs"]))
    print("Retrieval summary:", retrieval_summary_path(run_id))


if __name__ == "__main__":
    main()
