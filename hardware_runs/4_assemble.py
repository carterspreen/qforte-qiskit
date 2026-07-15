"""Stage 4: assemble retrieved hardware counts into UQK overlap matrices.

Manual run:
  conda activate qiskit_env_v1
  python hardware_runs/4_assemble.py dry_run
"""

from __future__ import annotations

import numpy as np

from hardware_common import (
    ENFORCE_C0_EXACT,
    KRYLOV_DIMENSION,
    MAX_CORRELATION_POWER,
    MFE_LABELS,
    assemble_dir,
    complex_array_to_records,
    complex_matrix_to_nested_records,
    complex_record,
    get_profile,
    get_run_id,
    load_json,
    load_reference_matrix,
    matrix_error_metrics,
    now_utc,
    plan_metadata_path,
    retrieval_summary_path,
    save_json,
    sum_counts,
    write_csv,
)


def import_mfe_helpers():
    import sys
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts" / "qiskit"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from mfe_measurement_building_blocks import estimate_z_from_counts

    return estimate_z_from_counts


def assemble_overlap_matrix(correlations):
    matrix = np.empty((KRYLOV_DIMENSION, KRYLOV_DIMENSION), dtype=np.complex128)
    for m in range(KRYLOV_DIMENSION):
        for n in range(KRYLOV_DIMENSION):
            diff = n - m
            if diff >= 0:
                matrix[m, n] = correlations[diff]
            else:
                matrix[m, n] = np.conjugate(correlations[-diff])
    return matrix


def load_counts_by_circuit(retrieval_summary):
    counts_by_circuit = {}
    pending_jobs = []
    for job_record in retrieval_summary["jobs"]:
        if not job_record.get("result_ready"):
            pending_jobs.append(job_record)
            continue
        payload = load_json(job_record["result_file"])
        for item in payload["counts_by_circuit"]:
            counts_by_circuit[int(item["circuit_index"])] = item["counts"]
    return counts_by_circuit, pending_jobs


def concise_job_record(job_record):
    return {
        "job_group_id": job_record.get("job_group_id"),
        "job_id": job_record.get("job_id"),
        "point_id": job_record.get("point_id"),
        "result_ready": bool(job_record.get("result_ready")),
        "status_at_retrieve": job_record.get("status_at_retrieve"),
        "shots": job_record.get("shots"),
        "total_executions": job_record.get("total_executions"),
    }


def entries_by_point(plan_metadata):
    grouped = {}
    for entry in plan_metadata["circuit_index"]:
        grouped.setdefault(entry["point_id"], []).append(entry)
    return grouped


def first_entry(entries):
    return sorted(entries, key=lambda item: int(item["circuit_index"]))[0]


def assemble_point(point_id, entries, counts_by_circuit, plan_metadata, estimate_z_from_counts):
    dt = float(plan_metadata["dt"])
    scalar_energy = float(plan_metadata["sampling_model"]["scalar_energy"])
    hf_count_key = plan_metadata["hf_count_key"]
    point0 = first_entry(entries)

    measured_correlations = np.zeros(MAX_CORRELATION_POWER + 1, dtype=np.complex128)
    stored_correlations = np.zeros(MAX_CORRELATION_POWER + 1, dtype=np.complex128)
    per_power_metadata = []
    if ENFORCE_C0_EXACT:
        measured_correlations[0] = 1.0 + 0.0j
        stored_correlations[0] = 1.0 + 0.0j

    for power in range(1, MAX_CORRELATION_POWER + 1):
        counts_by_label = {}
        for label in MFE_LABELS:
            matching = [
                entry
                for entry in entries
                if int(entry["power"]) == power and entry["mfe_label"] == label
            ]
            if not matching:
                raise RuntimeError(f"No counts for {point_id}, k={power}, {label}.")
            counts_by_label[label] = sum_counts(
                [counts_by_circuit[int(entry["circuit_index"])] for entry in matching]
            )
        estimate = estimate_z_from_counts(counts_by_label, hf_count_key, verbose=False)
        non_scalar = complex(estimate.z)
        scalar_phase = np.exp(-1j * scalar_energy * power * dt)
        measured = scalar_phase * non_scalar
        stored = measured
        measured_correlations[power] = measured
        stored_correlations[power] = stored
        per_power_metadata.append(
            {
                "power": int(power),
                "counts": counts_by_label,
                "fidelities": {
                    "F1": float(estimate.f1),
                    "F2_plus": float(estimate.f2_plus),
                    "F2_i": float(estimate.f2_i),
                },
                "non_scalar_measured_correlation": complex_record(non_scalar),
                "scalar_phase_applied_analytically": complex_record(scalar_phase),
                "measured_correlation": complex_record(measured),
                "stored_correlation": complex_record(stored),
            }
        )

    overlap_matrix = assemble_overlap_matrix(stored_correlations)
    return {
        "point_id": point_id,
        "mode": point0["mode"],
        "Nmfe": point0["Nmfe"],
        "Nd": point0.get("Nd"),
        "Nw": point0.get("Nw"),
        "sipc": point0.get("sipc"),
        "total_shots_per_correlation": point0.get("total_shots_per_correlation"),
        "ratio_nmfe_over_nw": point0.get("ratio_nmfe_over_nw"),
        "S": overlap_matrix,
        "measured_correlations": measured_correlations,
        "stored_correlations": stored_correlations,
        "per_power_metadata": per_power_metadata,
    }


def save_point_outputs(run_id, point_result, plan_metadata, reference_matrix, reference_path):
    point_dir = assemble_dir(run_id) / point_result["point_id"]
    point_dir.mkdir(parents=True, exist_ok=True)
    npz_path = point_dir / f"{point_result['point_id']}_hardware_uqk_overlap_matrix.npz"
    metadata_path = point_dir / f"{point_result['point_id']}_hardware_uqk_overlap_matrix_metadata.json"
    np.savez(
        npz_path,
        S=point_result["S"],
        correlations=point_result["stored_correlations"],
        measured_correlations=point_result["measured_correlations"],
        correlation_powers=np.arange(MAX_CORRELATION_POWER + 1, dtype=int),
        krylov_dimension=np.array(KRYLOV_DIMENSION, dtype=int),
        dt=np.array(float(plan_metadata["dt"]), dtype=float),
    )
    diagnostics = {}
    if reference_matrix is not None:
        diagnostics.update(matrix_error_metrics(reference_matrix, point_result["S"]))
    metadata = {
        "schema_version": 1,
        "generated_at_utc": now_utc(),
        "point_id": point_result["point_id"],
        "mode": point_result["mode"],
        "options": {
            "krylov_dimension": KRYLOV_DIMENSION,
            "max_correlation_power": MAX_CORRELATION_POWER,
            "dt": plan_metadata["dt"],
            "backend_name": plan_metadata["backend"]["backend_name"],
            "resolved_backend_name": plan_metadata["backend"]["resolved_backend_name"],
            "Nmfe": point_result["Nmfe"],
            "Nd": point_result["Nd"],
            "Nw": point_result["Nw"],
            "sipc": point_result["sipc"],
        },
        "outputs": {
            "npz": str(npz_path),
            "metadata_json": str(metadata_path),
        },
        "reference": {
            "S_reference_npz": str(reference_path) if reference_path else None,
        },
        "diagnostics": diagnostics,
        "correlations": complex_array_to_records(point_result["stored_correlations"]),
        "measured_correlations": complex_array_to_records(
            point_result["measured_correlations"]
        ),
        "overlap_matrix": complex_matrix_to_nested_records(point_result["S"]),
        "mfe_by_power": point_result["per_power_metadata"],
    }
    save_json(metadata_path, metadata)
    return npz_path, metadata_path, diagnostics


def main():
    profile = get_profile()
    run_id = get_run_id(profile)
    estimate_z_from_counts = import_mfe_helpers()
    plan_metadata = load_json(plan_metadata_path(run_id))
    retrieval_summary = load_json(retrieval_summary_path(run_id))
    counts_by_circuit, pending_jobs = load_counts_by_circuit(retrieval_summary)
    reference_matrix, reference_path = load_reference_matrix()
    pending_jobs_by_point = {}
    for job_record in pending_jobs:
        pending_jobs_by_point.setdefault(job_record.get("point_id"), []).append(
            concise_job_record(job_record)
        )

    records = []
    skipped_points = []
    for point_id, entries in sorted(entries_by_point(plan_metadata).items()):
        missing_circuit_indices = [
            int(entry["circuit_index"])
            for entry in entries
            if int(entry["circuit_index"]) not in counts_by_circuit
        ]
        if missing_circuit_indices:
            skipped = {
                "point_id": point_id,
                "reason": "retrieved_counts_incomplete",
                "expected_circuit_count": len(entries),
                "retrieved_circuit_count": len(entries) - len(missing_circuit_indices),
                "missing_circuit_count": len(missing_circuit_indices),
                "pending_jobs": pending_jobs_by_point.get(point_id, []),
            }
            skipped_points.append(skipped)
            print(
                "Skipping incomplete",
                point_id,
                "retrieved_circuits=",
                skipped["retrieved_circuit_count"],
                "expected_circuits=",
                skipped["expected_circuit_count"],
            )
            continue
        point_result = assemble_point(
            point_id,
            entries,
            counts_by_circuit,
            plan_metadata,
            estimate_z_from_counts,
        )
        npz_path, metadata_path, diagnostics = save_point_outputs(
            run_id,
            point_result,
            plan_metadata,
            reference_matrix,
            reference_path,
        )
        record = {
            "plot": "hardware",
            "uqk_mode": point_result["mode"],
            "backend_mode": "ibm_hardware",
            "hardware_backend": plan_metadata["backend"]["backend_name"],
            "resolved_backend_name": plan_metadata["backend"]["resolved_backend_name"],
            "profile": profile,
            "run_id": run_id,
            "point_id": point_id,
            "krylov_dimension": KRYLOV_DIMENSION,
            "dt": plan_metadata["dt"],
            "Nd": point_result["Nd"],
            "sipc": point_result["sipc"],
            "Nw": point_result["Nw"],
            "Nmfe": point_result["Nmfe"],
            "total_shots_per_correlation": point_result["total_shots_per_correlation"],
            "ratio_nmfe_over_nw": point_result["ratio_nmfe_over_nw"],
            "npz_path": str(npz_path),
            "metadata_path": str(metadata_path),
        }
        record.update(diagnostics)
        records.append(record)
        print("Assembled", point_id, "->", npz_path)

    summary = {
        "schema_version": 1,
        "generated_at_utc": now_utc(),
        "profile": profile,
        "run_id": run_id,
        "plan_metadata_path": str(plan_metadata_path(run_id)),
        "retrieval_summary_path": str(retrieval_summary_path(run_id)),
        "S_reference_npz": str(reference_path) if reference_path else None,
        "retrieval_complete": not pending_jobs,
        "pending_job_count": len(pending_jobs),
        "pending_jobs": [concise_job_record(job_record) for job_record in pending_jobs],
        "assembled_point_count": len(records),
        "skipped_points": skipped_points,
        "records": records,
    }
    save_json(assemble_dir(run_id) / "hardware_overlap_records.json", summary)
    write_csv(assemble_dir(run_id) / "hardware_overlap_records.csv", records)
    print("Hardware records:", assemble_dir(run_id) / "hardware_overlap_records.json")


if __name__ == "__main__":
    main()
