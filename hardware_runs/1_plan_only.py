"""Stage 1: build and transpile a hardware UQK circuit plan.

Manual run:
  conda activate qiskit_env_v1
  python hardware_runs/1_plan_only.py dry_run
  python hardware_runs/1_plan_only.py production_run

For real submission planning, set BACKEND_SOURCE = "runtime" below after your
IBM credentials are saved. The default "fake" backend is for local validation
only and cannot be submitted to IBM hardware.
"""

from __future__ import annotations

import os

from hardware_common import (
    IBM_BACKEND_SOURCE,
    build_hardware_circuit_plan,
    dump_qpy_circuits,
    get_profile,
    get_run_id,
    log_progress,
    plan_metadata_path,
    qpy_path,
    save_json,
)


BACKEND_SOURCE = os.environ.get("QFORTE_IBM_BACKEND_SOURCE", IBM_BACKEND_SOURCE)


def main():
    profile = get_profile()
    run_id = get_run_id(profile)
    metadata_path = plan_metadata_path(run_id)
    if metadata_path.exists():
        log_progress(f"Removing stale plan metadata before rebuild: {metadata_path}")
        metadata_path.unlink()
    circuits, metadata = build_hardware_circuit_plan(
        profile,
        backend_source=BACKEND_SOURCE,
    )
    metadata["run_id"] = run_id
    metadata["qpy_path"] = str(qpy_path(run_id))
    metadata["plan_metadata_path"] = str(metadata_path)

    dump_qpy_circuits(qpy_path(run_id), circuits)
    log_progress(f"Writing plan metadata: {metadata_path}")
    save_json(metadata_path, metadata)
    log_progress(f"Finished writing plan metadata: {metadata_path}")

    print("Hardware UQK plan written")
    print("  profile:", profile)
    print("  run_id:", run_id)
    print("  backend source:", metadata["backend"]["backend_source"])
    print("  backend:", metadata["backend"]["resolved_backend_name"])
    print("  circuits:", metadata["num_circuits"])
    print("  total executions:", metadata["total_executions"])
    print("  QPY:", qpy_path(run_id))
    print("  metadata:", metadata_path)
    if metadata["backend"]["backend_source"] != "runtime":
        print("  NOTE: this is a local/fake-backend plan and submit will reject it.")


if __name__ == "__main__":
    main()
