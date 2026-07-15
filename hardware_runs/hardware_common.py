"""Shared helpers for qforte-qiskit IBM hardware UQK runs.

The four stage scripts in this directory use a saved plan as their contract:

1. plan_only builds/transpiles circuits and writes a QPY plus JSON index.
2. submit reads that plan and submits grouped Sampler jobs.
3. retrieve reads saved job IDs and stores raw counts.
4. assemble turns raw counts back into UQK overlap matrices.

No IBM API token or instance CRN should ever be written to a run directory.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from qiskit import qpy
from qiskit.transpiler import generate_preset_pass_manager


HARDWARE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = HARDWARE_ROOT.parent
RUNS_ROOT = HARDWARE_ROOT / "runs"
SECRETS_PATH = HARDWARE_ROOT / "secrets" / "ibm_quantum_runtime.json"

MOLECULE_NAME = "diatomic_h2_sto_3g"
EXPERIMENT_NAME = "uqk_hardware_preliminary"
KRYLOV_DIMENSION = 4
MAX_CORRELATION_POWER = KRYLOV_DIMENSION - 1
DT_FALLBACK = 0.1
RANDOM_SEED = 330623

IBM_ACCOUNT_NAME = "qforte-hardware"
IBM_BACKEND_NAME = "ibm_brisbane"
IBM_BACKEND_SOURCE = "fake"  # "fake" for local planning checks; "runtime" for real hardware plans.
IBM_FAKE_BACKEND_CLASS = "FakeBrisbane"
IBM_USE_FRACTIONAL_GATES = False
TRANSPILE_OPTIMIZATION_LEVEL = 3

MEASURE_C0 = False
ENFORCE_C0_EXACT = True
STOCHASTIC_WEIGHT_CONVENTION = "group_pauli_l1_norm"

PROFILE_CONFIGS = {
    "dry_run": {
        "standard": [
            {
                "Nmfe": 1024,
            }
        ],
        "stochastic": [
            {
                "Nd": 1,
                "Nw": 5,
                "Nmfe": 512,
            }
        ],
    },
    "production_run": {
        "standard": [
            {
                "Nmfe": 10_000,
            }
        ],
        "stochastic": [
            {
                "Nd": nd,
                "Nw": 500,
                "Nmfe": 200,
            }
            for nd in [1, 2, 4, 8]
        ],
    },
}

MFE_LABELS = ["F1", "F2_plus", "F2_i"]
MAX_EXECUTIONS_PER_JOB = 9_000_000
MAX_CIRCUITS_PER_JOB = 500
PROGRESS_LOG_EVERY_STOCHASTIC_INSTANCES = 50


def now_utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def log_progress(message):
    print(f"[{now_utc()}] {message}", flush=True)


def safe_label_part(value):
    return (
        str(value)
        .replace(".", "p")
        .replace("-", "m")
        .replace("/", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def write_csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = sorted({key for record in records for key in record})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    return path


def complex_record(value):
    value = complex(value)
    return {"real": float(value.real), "imag": float(value.imag)}


def complex_from_record(record):
    return complex(float(record["real"]), float(record["imag"]))


def complex_array_to_records(values):
    return [
        {"index": int(index), **complex_record(value)}
        for index, value in enumerate(values)
    ]


def complex_matrix_to_nested_records(matrix):
    return [
        [complex_record(matrix[row, col]) for col in range(matrix.shape[1])]
        for row in range(matrix.shape[0])
    ]


def matrix_error_metrics(reference, candidate):
    diff = np.asarray(reference) - np.asarray(candidate)
    reference_norm = float(np.linalg.norm(reference))
    absolute_fro = float(np.linalg.norm(diff))
    return {
        "absolute_frobenius_error": absolute_fro,
        "relative_frobenius_error": (
            float(absolute_fro / reference_norm)
            if reference_norm > 0.0
            else float("nan")
        ),
        "max_abs_entry_error": float(np.max(np.abs(diff))),
        "reference_frobenius_norm": reference_norm,
    }


def default_run_id(profile):
    return (
        f"{MOLECULE_NAME}_{EXPERIMENT_NAME}_{profile}"
        f"_M_{KRYLOV_DIMENSION}"
        f"_kmax_{MAX_CORRELATION_POWER}"
        f"_seed_{RANDOM_SEED}"
    )


def get_profile():
    profile = sys.argv[1] if len(sys.argv) > 1 else "dry_run"
    if profile not in PROFILE_CONFIGS:
        raise ValueError(
            f"Unknown profile {profile!r}. Expected one of {sorted(PROFILE_CONFIGS)}."
        )
    return profile


def get_run_id(profile):
    if len(sys.argv) > 2:
        return sys.argv[2]
    return default_run_id(profile)


def run_dir(run_id):
    return RUNS_ROOT / run_id


def plan_dir(run_id):
    return run_dir(run_id) / "plan"


def submit_dir(run_id):
    return run_dir(run_id) / "submit"


def retrieve_dir(run_id):
    return run_dir(run_id) / "retrieve"


def assemble_dir(run_id):
    return run_dir(run_id) / "assembled"


def qpy_path(run_id):
    return plan_dir(run_id) / "hardware_circuits.qpy"


def plan_metadata_path(run_id):
    return plan_dir(run_id) / "plan_metadata.json"


def submitted_jobs_path(run_id):
    return submit_dir(run_id) / "submitted_jobs.json"


def retrieval_summary_path(run_id):
    return retrieve_dir(run_id) / "retrieval_summary.json"


def workflow_manifest_path():
    return REPO_ROOT / "notebooks" / MOLECULE_NAME / "metadata" / "workflow_manifest.json"


def workflow_file(manifest, key):
    value = Path(manifest["files"][key])
    if not value.is_absolute():
        value = REPO_ROOT / value
    return value


def import_uqk_builder():
    scripts_dir = REPO_ROOT / "scripts" / "qiskit"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location(
        "hardware_build_uqk_overlap_matrix",
        scripts_dir / "build_uqk_overlap_matrix.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_workflow_bundle():
    builder = import_uqk_builder()
    manifest = load_json(workflow_manifest_path())
    molecule_metadata = load_json(workflow_file(manifest, "molecule_metadata_json"))
    circuit_metadata = load_json(workflow_file(manifest, "grouped_evolution_metadata_json"))
    hp_archive = load_json(workflow_file(manifest, "hermitian_pairs_json"))
    pauli_archive = load_json(workflow_file(manifest, "grouped_paulis_json"))
    group_circuits = builder.load_qpy_circuits(
        workflow_file(manifest, "grouped_evolution_qpy")
    )
    occupation, hf_count_key = builder.validate_hf_metadata(molecule_metadata)
    full_step, full_step_group_indices, skipped_scalar_group_indices = (
        builder.build_one_trotter_step(group_circuits, circuit_metadata)
    )
    sampling_model = builder.build_qdrift_sampling_model(hp_archive, pauli_archive)
    return {
        "builder": builder,
        "manifest": manifest,
        "molecule_metadata": molecule_metadata,
        "circuit_metadata": circuit_metadata,
        "hp_archive": hp_archive,
        "pauli_archive": pauli_archive,
        "group_circuits": group_circuits,
        "occupation": occupation,
        "hf_count_key": hf_count_key,
        "full_step": full_step,
        "full_step_group_indices": full_step_group_indices,
        "skipped_scalar_group_indices": skipped_scalar_group_indices,
        "sampling_model": sampling_model,
        "num_qubits": int(circuit_metadata["active_space"]["num_qubits"]),
        "basis_gates": circuit_metadata["options"]["generic_basis_gates"],
        "dt": float(circuit_metadata["options"].get("dt", DT_FALLBACK)),
    }


def load_secrets_if_present():
    if not SECRETS_PATH.exists():
        return None
    payload = load_json(SECRETS_PATH)
    required = ["token", "instance"]
    missing = [key for key in required if not payload.get(key)]
    if missing:
        raise ValueError(f"{SECRETS_PATH} is missing required keys: {missing}")
    return payload


def configured_backend_name():
    secrets = load_secrets_if_present()
    if secrets and secrets.get("backend_name"):
        return str(secrets["backend_name"])
    return IBM_BACKEND_NAME


def redacted_instance_id(instance):
    if not instance:
        return None
    digest = hashlib.sha256(str(instance).encode("utf-8")).hexdigest()[:12]
    return f"sha256:{digest}"


def load_runtime_service(account_name=IBM_ACCOUNT_NAME, allow_save_from_secrets=True):
    from qiskit_ibm_runtime import QiskitRuntimeService

    secrets = load_secrets_if_present()
    if secrets and allow_save_from_secrets:
        QiskitRuntimeService.save_account(
            token=secrets["token"],
            instance=secrets["instance"],
            name=secrets.get("account_name", account_name),
            set_as_default=False,
            overwrite=True,
        )
        account_name = secrets.get("account_name", account_name)
    return QiskitRuntimeService(name=account_name)


def load_target_backend(source=IBM_BACKEND_SOURCE):
    if source == "fake":
        import qiskit_ibm_runtime.fake_provider as fake_provider

        backend_class = getattr(fake_provider, IBM_FAKE_BACKEND_CLASS)
        return backend_class(), {
            "backend_source": "fake",
            "backend_name": IBM_FAKE_BACKEND_CLASS,
            "safe_for_submit": False,
        }
    if source == "runtime":
        service = load_runtime_service()
        backend_name_to_use = configured_backend_name()
        backend = service.backend(
            backend_name_to_use,
            use_fractional_gates=IBM_USE_FRACTIONAL_GATES,
        )
        return backend, {
            "backend_source": "runtime",
            "backend_name": backend_name_to_use,
            "account_name": IBM_ACCOUNT_NAME,
            "safe_for_submit": True,
        }
    raise ValueError("IBM_BACKEND_SOURCE must be 'fake' or 'runtime'.")


def backend_name(backend):
    name = getattr(backend, "name", None)
    if callable(name):
        return str(name())
    return str(name) if name else str(type(backend).__name__)


def make_pass_manager(backend):
    return generate_preset_pass_manager(
        optimization_level=TRANSPILE_OPTIMIZATION_LEVEL,
        backend=backend,
    )


def operation_counts(circuit):
    return {name: int(count) for name, count in circuit.count_ops().items()}


def iter_profile_points(profile):
    config = PROFILE_CONFIGS[profile]
    for standard in config["standard"]:
        nmfe = int(standard["Nmfe"])
        yield {
            "mode": "standard",
            "point_id": f"standard_Nmfe_{nmfe}",
            "Nmfe": nmfe,
            "Nd": None,
            "Nw": None,
            "shots": nmfe,
        }
    for stochastic in config["stochastic"]:
        nd = int(stochastic["Nd"])
        nw = int(stochastic["Nw"])
        nmfe = int(stochastic["Nmfe"])
        yield {
            "mode": "stochastic",
            "point_id": f"stochastic_Nd_{nd}_Nw_{nw}_Nmfe_{nmfe}",
            "Nmfe": nmfe,
            "Nd": nd,
            "Nw": nw,
            "sipc": nw,
            "shots": nmfe,
            "total_shots_per_correlation": nmfe * nw,
            "ratio_nmfe_over_nw": float(nmfe / nw),
        }


def powers_to_measure():
    start = 0 if MEASURE_C0 else 1
    return list(range(start, MAX_CORRELATION_POWER + 1))


def make_circuit_id(point, power, instance_index, mfe_label):
    if point["mode"] == "standard":
        return f"{point['point_id']}_k_{power}_{mfe_label}"
    return (
        f"{point['point_id']}_k_{power}"
        f"_sample_{instance_index}_{mfe_label}"
    )


def build_hardware_circuit_plan(profile, backend_source=IBM_BACKEND_SOURCE):
    log_progress(f"Loading workflow bundle for profile={profile!r}.")
    bundle = load_workflow_bundle()
    builder = bundle["builder"]
    log_progress(f"Loading target backend from source={backend_source!r}.")
    target_backend, backend_metadata = load_target_backend(backend_source)
    log_progress(
        "Building transpiler pass manager "
        f"optimization_level={TRANSPILE_OPTIMIZATION_LEVEL}."
    )
    pass_manager = make_pass_manager(target_backend)
    rng = np.random.default_rng(RANDOM_SEED)

    circuits = []
    circuit_index = []
    point_summaries = []
    circuit_counter = 0
    points = list(iter_profile_points(profile))
    log_progress(f"Planning {len(points)} profile point(s).")
    for point_index, point in enumerate(points, start=1):
        log_progress(
            f"Starting point {point_index}/{len(points)}: {point['point_id']} "
            f"shots={point['shots']}."
        )
        point_circuit_indices = []
        for power in powers_to_measure():
            log_progress(
                f"  Building evolution items for {point['point_id']} k={power}."
            )
            if point["mode"] == "standard":
                evolution_items = [
                    {
                        "instance_index": None,
                        "evolution_circuit": builder.build_trotter_power(
                            bundle["full_step"],
                            power,
                        ),
                        "sampled_group_history": [],
                    }
                ]
            else:
                builder.QDRIFT_SEGMENT_COUNT_ND = int(point["Nd"])
                evolution_items = []
                for instance_index in range(int(point["Nw"])):
                    if (
                        instance_index == 0
                        or (instance_index + 1) % PROGRESS_LOG_EVERY_STOCHASTIC_INSTANCES == 0
                        or instance_index + 1 == int(point["Nw"])
                    ):
                        log_progress(
                            f"    Sampling qDRIFT instance {instance_index + 1}/"
                            f"{point['Nw']} for {point['point_id']} k={power}."
                        )
                    total_time = power * bundle["dt"]
                    chunk, history = builder.build_stochastic_qdrift_instance(
                        bundle["sampling_model"],
                        total_time,
                        rng,
                        bundle["num_qubits"],
                        bundle["basis_gates"],
                        power,
                        instance_index,
                    )
                    evolution_items.append(
                        {
                            "instance_index": instance_index,
                            "evolution_circuit": chunk,
                            "sampled_group_history": history,
                        }
                    )

            log_progress(
                f"  Transpiling {len(evolution_items) * len(MFE_LABELS)} "
                f"MFE circuit(s) for {point['point_id']} k={power}."
            )
            for item in evolution_items:
                templates = builder.build_mfe_templates(
                    item["evolution_circuit"],
                    bundle["occupation"],
                    verbose=False,
                )
                for mfe_label in MFE_LABELS:
                    template = templates[mfe_label]
                    isa_circuit = pass_manager.run(template)
                    circuit_id = make_circuit_id(
                        point,
                        power,
                        item["instance_index"],
                        mfe_label,
                    )
                    isa_circuit.name = circuit_id[:100]
                    circuits.append(isa_circuit)
                    point_circuit_indices.append(circuit_counter)
                    circuit_index.append(
                        {
                            "circuit_index": circuit_counter,
                            "circuit_id": circuit_id,
                            "point_id": point["point_id"],
                            "mode": point["mode"],
                            "power": int(power),
                            "instance_index": item["instance_index"],
                            "mfe_label": mfe_label,
                            "shots": int(point["shots"]),
                            "Nmfe": int(point["Nmfe"]),
                            "Nd": point["Nd"],
                            "Nw": point["Nw"],
                            "sipc": point.get("sipc"),
                            "total_shots_per_correlation": point.get(
                                "total_shots_per_correlation"
                            ),
                            "ratio_nmfe_over_nw": point.get("ratio_nmfe_over_nw"),
                            "template_depth": int(template.depth()),
                            "template_size": int(template.size()),
                            "template_operation_counts": operation_counts(template),
                            "isa_depth": int(isa_circuit.depth()),
                            "isa_size": int(isa_circuit.size()),
                            "isa_operation_counts": operation_counts(isa_circuit),
                            "isa_num_qubits": int(isa_circuit.num_qubits),
                            "isa_num_clbits": int(isa_circuit.num_clbits),
                            "sampled_group_history": item["sampled_group_history"],
                        }
                    )
                    circuit_counter += 1
            log_progress(
                f"  Finished {point['point_id']} k={power}; "
                f"total planned circuits={circuit_counter}."
            )
        point_summaries.append(
            {
                **point,
                "num_circuits": len(point_circuit_indices),
                "circuit_indices": point_circuit_indices,
                "total_executions": int(len(point_circuit_indices) * point["shots"]),
            }
        )
        log_progress(
            f"Finished point {point_index}/{len(points)}: {point['point_id']} "
            f"circuits={len(point_circuit_indices)}."
        )

    log_progress(f"Finished circuit planning; building metadata for {len(circuits)} circuits.")
    metadata = {
        "schema_version": 1,
        "generated_at_utc": now_utc(),
        "profile": profile,
        "molecule_name": MOLECULE_NAME,
        "experiment_name": EXPERIMENT_NAME,
        "krylov_dimension": KRYLOV_DIMENSION,
        "max_correlation_power": MAX_CORRELATION_POWER,
        "powers_measured": powers_to_measure(),
        "measure_c0": MEASURE_C0,
        "enforce_c0_exact": ENFORCE_C0_EXACT,
        "random_seed": RANDOM_SEED,
        "dt": bundle["dt"],
        "hf_count_key": bundle["hf_count_key"],
        "backend": {
            **backend_metadata,
            "resolved_backend_name": backend_name(target_backend),
            "transpile_optimization_level": TRANSPILE_OPTIMIZATION_LEVEL,
            "use_fractional_gates": IBM_USE_FRACTIONAL_GATES,
        },
        "workflow_inputs": {
            "manifest": str(workflow_manifest_path()),
            "grouped_evolution_qpy": str(
                workflow_file(bundle["manifest"], "grouped_evolution_qpy")
            ),
            "grouped_evolution_metadata_json": str(
                workflow_file(bundle["manifest"], "grouped_evolution_metadata_json")
            ),
            "molecule_metadata_json": str(
                workflow_file(bundle["manifest"], "molecule_metadata_json")
            ),
            "hermitian_pairs_json": str(
                workflow_file(bundle["manifest"], "hermitian_pairs_json")
            ),
            "grouped_paulis_json": str(
                workflow_file(bundle["manifest"], "grouped_paulis_json")
            ),
        },
        "sampling_model": {
            "weight_sum_lambda": bundle["sampling_model"]["weight_sum_lambda"],
            "scalar_energy": bundle["sampling_model"]["scalar_energy"],
            "stochastic_weight_convention": STOCHASTIC_WEIGHT_CONVENTION,
            "num_sampled_non_scalar_groups": len(bundle["sampling_model"]["entries"]),
        },
        "profile_points": point_summaries,
        "num_circuits": len(circuits),
        "total_executions": int(sum(point["total_executions"] for point in point_summaries)),
        "circuit_index": circuit_index,
        "notes": [
            "Plan-only generation does not submit work to an IBM QPU.",
            "The submit stage must reject plans whose backend source is not 'runtime'.",
            "C0 is not measured by default; it is stored as exactly 1 during assembly.",
        ],
    }
    return circuits, metadata


def dump_qpy_circuits(path, circuits):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    log_progress(f"Writing {len(circuits)} circuit(s) to QPY: {path}")
    with path.open("wb") as handle:
        qpy.dump(circuits, handle)
    log_progress(f"Finished writing QPY: {path}")
    return path


def load_qpy_circuits(path):
    with Path(path).open("rb") as handle:
        return list(qpy.load(handle))


def execution_groups_from_plan(plan_metadata):
    by_point = defaultdict(list)
    for entry in plan_metadata["circuit_index"]:
        by_point[entry["point_id"]].append(entry)
    groups = []
    for point_id, entries in sorted(by_point.items()):
        entries = sorted(entries, key=lambda item: int(item["circuit_index"]))
        shots = int(entries[0]["shots"])
        current = []
        current_executions = 0
        chunk_index = 0
        for entry in entries:
            if (
                current
                and (
                    len(current) >= MAX_CIRCUITS_PER_JOB
                    or current_executions + shots > MAX_EXECUTIONS_PER_JOB
                )
            ):
                groups.append(
                    {
                        "job_group_id": f"{point_id}_chunk_{chunk_index}",
                        "point_id": point_id,
                        "shots": shots,
                        "circuit_indices": [int(item["circuit_index"]) for item in current],
                        "total_executions": int(current_executions),
                    }
                )
                chunk_index += 1
                current = []
                current_executions = 0
            current.append(entry)
            current_executions += shots
        if current:
            groups.append(
                {
                    "job_group_id": f"{point_id}_chunk_{chunk_index}",
                    "point_id": point_id,
                    "shots": shots,
                    "circuit_indices": [int(item["circuit_index"]) for item in current],
                    "total_executions": int(current_executions),
                }
            )
    return groups


def sampler_pub_counts(pub_result):
    data = pub_result.data
    for name in ["meas", "c"]:
        register = getattr(data, name, None)
        if register is not None and hasattr(register, "get_counts"):
            return {key: int(value) for key, value in register.get_counts().items()}
    keys = []
    if hasattr(data, "keys"):
        keys = list(data.keys())
    else:
        keys = [key for key in dir(data) if not key.startswith("_")]
    for key in keys:
        register = getattr(data, key, None)
        if register is not None and hasattr(register, "get_counts"):
            return {bit: int(value) for bit, value in register.get_counts().items()}
    raise ValueError("Could not find a classical register with get_counts() in PubResult.")


def sum_counts(counts_list):
    total = {}
    for counts in counts_list:
        for bitstring, count in counts.items():
            total[bitstring] = total.get(bitstring, 0) + int(count)
    return total


def find_default_s_reference_npz():
    info_path = (
        REPO_ROOT
        / "notebooks"
        / MOLECULE_NAME
        / "experiments"
        / "uqk_overlap_hyperparameters"
        / "data"
        / "S_reference_info.json"
    )
    if not info_path.exists():
        return None
    info = load_json(info_path)
    npz_path = Path(info["npz_path"])
    return npz_path if npz_path.exists() else None


def load_reference_matrix():
    npz_path = find_default_s_reference_npz()
    if npz_path is None:
        return None, None
    data = np.load(npz_path)
    return np.array(data["S"], dtype=np.complex128), npz_path
