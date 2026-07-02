"""Stage 2: submit a planned UQK hardware run to IBM Runtime Sampler.

Manual run:
  conda activate qiskit_env_v1
  python hardware_runs/2_submit.py dry_run

This script is intentionally hard to submit accidentally. To submit real QPU
jobs, first generate a runtime-backed plan with 1_plan_only.py, then edit both
SUBMIT_TO_QPU and CONFIRMATION_TEXT below.
"""

from __future__ import annotations

from hardware_common import (
    IBM_USE_FRACTIONAL_GATES,
    configured_backend_name,
    execution_groups_from_plan,
    get_profile,
    get_run_id,
    load_json,
    load_qpy_circuits,
    load_runtime_service,
    now_utc,
    plan_metadata_path,
    qpy_path,
    save_json,
    submit_dir,
    submitted_jobs_path,
)


SUBMIT_TO_QPU = False
CONFIRMATION_TEXT = "SUBMIT_QFORTE_HARDWARE"
REQUIRED_CONFIRMATION_TEXT = "SUBMIT_QFORTE_HARDWARE"
ALLOW_RESUBMIT = False


def assert_submit_is_intentional(plan_metadata):
    if not SUBMIT_TO_QPU:
        raise SystemExit(
            "SUBMIT_TO_QPU is False. Edit hardware_runs/2_submit.py only when "
            "you are ready to spend IBM Runtime budget."
        )
    if CONFIRMATION_TEXT != REQUIRED_CONFIRMATION_TEXT:
        raise SystemExit(
            "CONFIRMATION_TEXT does not match the required confirmation string."
        )
    if plan_metadata["backend"]["backend_source"] != "runtime":
        raise SystemExit(
            "The plan was not generated from a real Runtime backend. Re-run "
            "1_plan_only.py with QFORTE_IBM_BACKEND_SOURCE=runtime or edit "
            "BACKEND_SOURCE='runtime' before submitting."
        )


def main():
    profile = get_profile()
    run_id = get_run_id(profile)
    plan_metadata = load_json(plan_metadata_path(run_id))
    assert_submit_is_intentional(plan_metadata)

    if submitted_jobs_path(run_id).exists() and not ALLOW_RESUBMIT:
        raise SystemExit(
            f"{submitted_jobs_path(run_id)} already exists. Refusing to resubmit "
            "unless ALLOW_RESUBMIT=True."
        )

    from qiskit_ibm_runtime import SamplerV2 as Sampler

    circuits = load_qpy_circuits(qpy_path(run_id))
    execution_groups = execution_groups_from_plan(plan_metadata)
    service = load_runtime_service()
    backend_name = configured_backend_name()
    backend = service.backend(
        backend_name,
        use_fractional_gates=IBM_USE_FRACTIONAL_GATES,
    )
    sampler = Sampler(mode=backend)

    submitted = {
        "schema_version": 1,
        "generated_at_utc": now_utc(),
        "profile": profile,
        "run_id": run_id,
        "plan_metadata_path": str(plan_metadata_path(run_id)),
        "qpy_path": str(qpy_path(run_id)),
        "backend_name": backend_name,
        "jobs": [],
    }
    submit_dir(run_id).mkdir(parents=True, exist_ok=True)

    for group in execution_groups:
        group_circuits = [circuits[index] for index in group["circuit_indices"]]
        job = sampler.run(group_circuits, shots=int(group["shots"]))
        job_record = {
            **group,
            "job_id": job.job_id(),
            "submitted_at_utc": now_utc(),
            "status_at_submit": str(job.status()),
        }
        submitted["jobs"].append(job_record)
        save_json(submitted_jobs_path(run_id), submitted)
        print(
            "Submitted",
            group["job_group_id"],
            "job_id=",
            job_record["job_id"],
            "circuits=",
            len(group["circuit_indices"]),
            "shots=",
            group["shots"],
        )

    print("Submission manifest:", submitted_jobs_path(run_id))


if __name__ == "__main__":
    main()
