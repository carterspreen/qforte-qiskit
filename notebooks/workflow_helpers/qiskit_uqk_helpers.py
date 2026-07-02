"""Helpers for notebook 2: grouped Pauli blocks to UQK energies.

The notebook cells should stay readable and option-driven. These helpers load
the existing script modules, set their hard-coded options from the notebook,
run them, update the notebook-local manifest, and print compact summaries.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def now_utc():
    """Return a compact UTC timestamp for metadata files."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def print_header(title):
    """Print a simple section header that reads well in notebooks."""

    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def print_kv(label, value):
    """Print one aligned key-value line for notebook summaries."""

    print(f"{label:<40} {value}")


def load_json(path):
    """Read JSON from disk."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, payload):
    """Write pretty JSON and return the path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def update_manifest(manifest_path, updates):
    """Merge updates into the notebook-local workflow manifest."""

    manifest_path = Path(manifest_path)
    manifest = load_json(manifest_path)
    manifest["updated_at_utc"] = now_utc()
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(manifest.get(key), dict):
            manifest[key].update(value)
        else:
            manifest[key] = value
    save_json(manifest_path, manifest)
    return manifest


def repo_root_from_helper():
    """Return the repository root from this helper file location."""

    return Path(__file__).resolve().parents[2]


def load_workflow_manifest(notebooks_root, molecule_name):
    """Load notebooks/<molecule_name>/metadata/workflow_manifest.json."""

    manifest_path = (
        Path(notebooks_root).resolve()
        / molecule_name
        / "metadata"
        / "workflow_manifest.json"
    )
    manifest = load_json(manifest_path)
    print_header("Loaded Workflow Manifest")
    print_kv("Molecule name:", manifest["molecule_name"])
    print_kv("Manifest JSON:", manifest_path)
    print_kv("Workflow root:", manifest["workflow_root"])
    for key, value in sorted(manifest.get("files", {}).items()):
        print_kv(f"file: {key}", value)
    return manifest_path, manifest


def load_module_from_path(name, path):
    """Import a Python script as a module without changing the old script."""

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def workflow_file(manifest, key):
    """Return a manifest file path and fail with a friendly error if missing."""

    files = manifest.get("files", {})
    if key not in files:
        raise KeyError(
            f"Manifest does not contain files[{key!r}]. Run the earlier notebook cell first."
        )
    return Path(files[key])


def assert_manifest_molecule_label(manifest, path, context):
    """Ensure a workflow JSON belongs to the manifest molecule when labeled."""

    expected = manifest.get("molecule_name")
    if not expected:
        return
    payload = load_json(path)
    actual = payload.get("molecule", {}).get("label")
    if actual and actual != expected:
        raise ValueError(
            f"{context} molecule label mismatch: manifest has {expected!r}, "
            f"but {path} records {actual!r}."
        )


def build_grouped_evolution_circuits(
    *,
    manifest_path,
    manifest,
    dt,
    evolution_method,
    target_mode,
    generic_basis_gates,
    transpiler_optimization_level,
    trotter_sequence_order,
    ibm_backend_name="ibm_brisbane",
    ibm_runtime_channel="ibm_quantum_platform",
    ibm_instance=None,
    ibm_use_fractional_gates=False,
    verbose=True,
):
    """Build one Qiskit circuit for each grouped Pauli block.

    This wraps scripts/qiskit/build_h4_grouped_evolution_circuits.py, but points
    it at the notebook-local grouped-Pauli file and notebook-local output paths.
    The saved QPY plus metadata become the circuit archive consumed by UQK.
    """

    repo_root = repo_root_from_helper()
    module = load_module_from_path(
        "notebook_build_grouped_evolution_circuits",
        repo_root / "scripts" / "qiskit" / "build_h4_grouped_evolution_circuits.py",
    )
    molecule_name = manifest["molecule_name"]
    circuits_dir = Path(manifest["directories"]["circuits_dir"])
    output_qpy = circuits_dir / f"{molecule_name}_grouped_evolution.qpy"
    output_metadata = circuits_dir / f"{molecule_name}_grouped_evolution_metadata.json"

    module.INPUT_GROUPED_PAULI_JSON = workflow_file(manifest, "grouped_paulis_json")
    module.OUTPUT_QPY = output_qpy
    module.OUTPUT_METADATA_JSON = output_metadata
    module.DT = float(dt)
    module.EVOLUTION_METHOD = evolution_method
    module.TARGET_MODE = target_mode
    module.GENERIC_BASIS_GATES = list(generic_basis_gates)
    module.TRANSPILER_OPTIMIZATION_LEVEL = int(transpiler_optimization_level)
    module.TROTTER_SEQUENCE_ORDER = int(trotter_sequence_order)
    module.IBM_BACKEND_NAME = ibm_backend_name
    module.IBM_RUNTIME_CHANNEL = ibm_runtime_channel
    module.IBM_INSTANCE = ibm_instance
    module.IBM_USE_FRACTIONAL_GATES = bool(ibm_use_fractional_gates)
    module.VERBOSE = bool(verbose)
    module.main()

    metadata = load_json(output_metadata)
    manifest = update_manifest(
        manifest_path,
        {
            "files": {
                "grouped_evolution_qpy": str(output_qpy),
                "grouped_evolution_metadata_json": str(output_metadata),
            },
            "qiskit_circuit_options": {
                "dt": float(dt),
                "evolution_method": evolution_method,
                "target_mode": target_mode,
                "generic_basis_gates": list(generic_basis_gates),
                "transpiler_optimization_level": int(transpiler_optimization_level),
                "trotter_sequence_order": int(trotter_sequence_order),
            },
        },
    )

    print_header("Circuit Archive Summary")
    print_kv("QPY:", output_qpy)
    print_kv("Metadata JSON:", output_metadata)
    print_kv("Number of circuits:", metadata["circuit_archive"]["num_circuits"])
    print_kv("Depth statistics:", metadata["depth_statistics"])
    return manifest, metadata


def run_uqk_overlap_driver(
    *,
    manifest_path,
    manifest,
    uqk_mode,
    backend_mode,
    krylov_dimension,
    max_correlation_power,
    shots_per_mfe_experiment,
    qdrift_segment_count_Nd,
    stochastic_instances_per_correlation,
    stochastic_weight_convention,
    random_seed,
    output_label,
    noisy_simulation_method="density_matrix",
    noisy_transpile_optimization_level=1,
    simple_noise_one_qubit_depolarizing_probability=1.0e-3,
    simple_noise_two_qubit_depolarizing_probability=1.0e-2,
    simple_noise_readout_error_probability=2.0e-2,
    ibm_model_source="fake_backend",
    ibm_model_fake_backend_class="FakeBrisbane",
    ibm_model_runtime_backend_name="ibm_brisbane",
    ibm_model_compress_to_active_space=True,
    mfe_verbose_for_first_nonzero_power=False,
):
    """Estimate UQK correlations C_k and assemble the overlap matrix S.

    This wraps scripts/qiskit/build_uqk_overlap_matrix.py. The helper sets all
    the script's hard-coded globals from the notebook cell, so the old CLI-style
    script remains intact while the notebook gets molecule-local outputs.
    Supported uqk_mode values are "standard", "exact_trotter", "stochastic",
    and "exact_stochastic".
    """

    repo_root = repo_root_from_helper()
    module = load_module_from_path(
        "notebook_build_uqk_overlap_matrix",
        repo_root / "scripts" / "qiskit" / "build_uqk_overlap_matrix.py",
    )
    results_dir = Path(manifest["directories"]["results_dir"])

    module.INPUT_QPY = workflow_file(manifest, "grouped_evolution_qpy")
    module.INPUT_CIRCUIT_METADATA_JSON = workflow_file(
        manifest, "grouped_evolution_metadata_json"
    )
    module.INPUT_MOLECULE_METADATA_JSON = workflow_file(
        manifest, "molecule_metadata_json"
    )
    module.INPUT_HERMITIAN_PAIR_JSON = workflow_file(manifest, "hermitian_pairs_json")
    module.INPUT_GROUPED_PAULI_JSON = workflow_file(manifest, "grouped_paulis_json")
    module.OUTPUT_DIR = results_dir

    for context, path in [
        ("Molecule metadata", module.INPUT_MOLECULE_METADATA_JSON),
        ("Hermitian-pair metadata", module.INPUT_HERMITIAN_PAIR_JSON),
        ("Grouped-Pauli metadata", module.INPUT_GROUPED_PAULI_JSON),
        ("Grouped-evolution metadata", module.INPUT_CIRCUIT_METADATA_JSON),
    ]:
        assert_manifest_molecule_label(manifest, path, context)

    module.UQK_MODE = uqk_mode
    module.KRYLOV_DIMENSION = int(krylov_dimension)
    module.MAX_CORRELATION_POWER = int(max_correlation_power)
    module.SHOTS_PER_MFE_EXPERIMENT = int(shots_per_mfe_experiment)
    module.BACKEND_MODE = backend_mode
    module.OUTPUT_FILE_STEM_PREFIX = ""
    module.OUTPUT_LABEL_OVERRIDE = output_label
    module.OUTPUT_LABEL_SUFFIX = ""
    module.NOISY_SIMULATION_METHOD = noisy_simulation_method
    module.NOISY_TRANSPILE_OPTIMIZATION_LEVEL = int(
        noisy_transpile_optimization_level
    )
    module.SIMPLE_NOISE_ONE_QUBIT_DEPOLARIZING_PROBABILITY = float(
        simple_noise_one_qubit_depolarizing_probability
    )
    module.SIMPLE_NOISE_TWO_QUBIT_DEPOLARIZING_PROBABILITY = float(
        simple_noise_two_qubit_depolarizing_probability
    )
    module.SIMPLE_NOISE_READOUT_ERROR_PROBABILITY = float(
        simple_noise_readout_error_probability
    )
    module.IBM_MODEL_SOURCE = ibm_model_source
    module.IBM_MODEL_FAKE_BACKEND_CLASS = ibm_model_fake_backend_class
    module.IBM_MODEL_RUNTIME_BACKEND_NAME = ibm_model_runtime_backend_name
    module.IBM_MODEL_COMPRESS_TO_ACTIVE_SPACE = bool(ibm_model_compress_to_active_space)
    module.QDRIFT_SEGMENT_COUNT_ND = int(qdrift_segment_count_Nd)
    module.STOCHASTIC_INSTANCES_PER_CORRELATION = int(
        stochastic_instances_per_correlation
    )
    module.STOCHASTIC_WEIGHT_CONVENTION = stochastic_weight_convention
    module.RANDOM_SEED = int(random_seed)
    module.MFE_VERBOSE_FOR_FIRST_NONZERO_POWER = bool(
        mfe_verbose_for_first_nonzero_power
    )
    module.main()

    output_npz = module.output_npz_path(module.UQK_MODE)
    output_metadata = module.output_metadata_path(module.UQK_MODE)
    data = np.load(output_npz)
    metadata = load_json(output_metadata)
    manifest = update_manifest(
        manifest_path,
        {
            "files": {
                "uqk_overlap_npz": str(output_npz),
                "uqk_overlap_metadata_json": str(output_metadata),
            },
            "uqk_overlap_options": metadata["options"],
        },
    )

    print_header("UQK Overlap Output Summary")
    print_kv("Overlap NPZ:", output_npz)
    print_kv("Metadata JSON:", output_metadata)
    print_kv("S shape:", data["S"].shape)
    print_kv("Stored C_k:", [complex(z) for z in data["correlations"]])
    print_kv("Condition number:", metadata["diagnostics"]["condition_number"])
    return manifest, data, metadata


def solve_uqk_gep_from_manifest(
    *,
    manifest_path,
    manifest,
    krylov_dimension_to_use,
    dt,
    overlap_eigenvalue_threshold,
    sort_roots_by,
    output_summary_name,
    use_direct_validation_when_available=True,
):
    """Solve the projected unitary GEV and save a JSON energy summary.

    This wraps scripts/qiskit/solve_uqk_gep.py. It reads the notebook-local C_k
    file from the manifest, assembles S and shifted U, solves U c=lambda S c,
    and converts eigenphases into UQK energy estimates.
    """

    repo_root = repo_root_from_helper()
    module = load_module_from_path(
        "notebook_solve_uqk_gep",
        repo_root / "scripts" / "qiskit" / "solve_uqk_gep.py",
    )
    summaries_dir = Path(manifest["directories"]["summaries_dir"])
    output_summary = summaries_dir / output_summary_name

    module.INPUT_CORRELATION_NPZ = workflow_file(manifest, "uqk_overlap_npz")
    module.INPUT_CORRELATION_METADATA_JSON = workflow_file(
        manifest, "uqk_overlap_metadata_json"
    )
    module.INPUT_MOLECULE_METADATA_JSON = workflow_file(
        manifest, "molecule_metadata_json"
    )
    module.INPUT_DIRECT_VALIDATION_NPZ = summaries_dir / "direct_validation_optional.npz"

    for context, path in [
        ("Correlation metadata", module.INPUT_CORRELATION_METADATA_JSON),
        ("Molecule metadata", module.INPUT_MOLECULE_METADATA_JSON),
    ]:
        assert_manifest_molecule_label(manifest, path, context)

    module.OUTPUT_SUMMARY_JSON = output_summary
    module.KRYLOV_DIMENSION_TO_USE = int(krylov_dimension_to_use)
    module.KRYLOV_DT = float(dt)
    module.OVERLAP_EIGENVALUE_THRESHOLD = overlap_eigenvalue_threshold
    module.SORT_ROOTS_BY = sort_roots_by
    module.USE_DIRECT_VALIDATION_WHEN_AVAILABLE = bool(
        use_direct_validation_when_available
    )
    module.main()

    summary = load_json(output_summary)
    manifest = update_manifest(
        manifest_path,
        {
            "files": {
                "uqk_gep_summary_json": str(output_summary),
            },
            "uqk_gep_options": summary["options"],
        },
    )

    print_header("UQK Energy Summary")
    print_kv("Summary JSON:", output_summary)
    for root in summary["input_correlation_solution"]["energies_hartree"]:
        print_kv(f"root {root['root']} energy:", f"{root['energy']:+.12f}")
    return manifest, summary
