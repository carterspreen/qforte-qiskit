# Manual run:
#   conda activate qiskit_env_v1
#   python scripts/qiskit/validate_uqk_overlap_statevector.py
#
# Summary:
#   Validate the standard UQK overlap matrix by direct noiseless statevector
#   linear algebra. This script mirrors build_uqk_overlap_matrix.py for the
#   standard path: build the physical non-scalar full Trotter block directly
#   from the grouped Pauli archive, including identity terms that Qiskit circuits
#   cannot expose as measurable global phases; apply the zero-body scalar phase
#   analytically; compute
#       C_k = <HF|U^k|HF>,
#       S_mn = C_(n-m), with C_-k = conj(C_k),
#   and compare against the saved finite-shot MFE result. The exact-probability
#   MFE check still uses the saved QPY circuits, because those are the circuits
#   actually measured. Its raw vacuum-reference estimate should match the
#   physical grouped-Pauli direct result.
#
# Hard-coded options:
#   INPUT_MOLECULE_METADATA_JSON = data/molecules/h4_linear_sto3g_metadata.json
#   INPUT_QPY = circuits/transpiled/h4_linear_sto3g_grouped_evolution.qpy
#   INPUT_CIRCUIT_METADATA_JSON = circuits/transpiled/h4_linear_sto3g_grouped_evolution_metadata.json
#   INPUT_MFE_NPZ = results/krylov/h4_standard_uqk_overlap_matrix.npz
#   INPUT_MFE_METADATA_JSON = results/krylov/h4_standard_uqk_overlap_matrix_metadata.json
#   KRYLOV_DIMENSION = 3
#   MAX_CORRELATION_POWER = KRYLOV_DIMENSION, so shifted C_M is available for
#       the projected unitary GEV post-processing script
#   KRYLOV_DT = 0.1
#   TROTTER_ORDER = 1
#   COMPUTE_EXACT_MFE_PROBABILITY_CHECK = True
#   outputs are results/validation/h4_standard_uqk_statevector_comparison.*
#
# Important dt convention:
#   Standard UQK reuses QPY circuits with dt already sealed into them. KRYLOV_DT
#   is therefore a guard value, not a knob that rescales the QPY circuits. If
#   KRYLOV_DT disagrees with the circuit metadata, this script exits rather than
#   comparing mismatched dynamics.

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.linalg import expm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from mfe_measurement_building_blocks import (  # noqa: E402
    F1_LABEL,
    F2_I_LABEL,
    F2_PLUS_LABEL,
    build_mfe_templates,
    validate_hf_metadata,
)


INPUT_MOLECULE_METADATA_JSON = (
    REPO_ROOT / "data" / "molecules" / "h4_linear_sto3g_metadata.json"
)
INPUT_QPY = (
    REPO_ROOT / "circuits" / "transpiled" / "h4_linear_sto3g_grouped_evolution.qpy"
)
INPUT_CIRCUIT_METADATA_JSON = (
    REPO_ROOT
    / "circuits"
    / "transpiled"
    / "h4_linear_sto3g_grouped_evolution_metadata.json"
)
INPUT_GROUPED_PAULI_JSON = (
    REPO_ROOT / "data" / "hamiltonians" / "h4_linear_sto3g_grouped_paulis.json"
)
INPUT_MFE_NPZ = REPO_ROOT / "results" / "krylov" / "h4_standard_uqk_overlap_matrix.npz"
INPUT_MFE_METADATA_JSON = (
    REPO_ROOT / "results" / "krylov" / "h4_standard_uqk_overlap_matrix_metadata.json"
)
OUTPUT_COMPARISON_NPZ = (
    REPO_ROOT
    / "results"
    / "validation"
    / "h4_standard_uqk_statevector_comparison.npz"
)
OUTPUT_COMPARISON_JSON = OUTPUT_COMPARISON_NPZ.with_suffix(".json")

# KRYLOV_DIMENSION is M in the notes. The S matrix is M x M and uses Krylov
# states |phi_n> = U^n |HF>, n=0,...,M-1.
KRYLOV_DIMENSION = 3

# The overlap matrix S only needs C_0 through C_(M-1), but the projected UQK
# unitary matrix uses U_mn=C_(n+1-m), which needs the shifted value C_M. Save
# direct correlations through k=M so the post-processing GEV script can verify
# both S and U from the same direct-linear-algebra data.
MAX_CORRELATION_POWER = KRYLOV_DIMENSION

# KRYLOV_DT must match the dt sealed into the saved standard QPY circuits.
# Change dt by rerunning build_h4_grouped_evolution_circuits.py, not by editing
# this value alone.
KRYLOV_DT = 0.1

# This validator currently expects first-order group sequence metadata, matching
# the standard overlap builder used so far.
TROTTER_ORDER = 1

# The saved MFE NPZ was produced with finite shots. When this option is True,
# the script also reconstructs the same MFE templates and evaluates their HF
# return probabilities exactly with Statevector. That isolates shot noise from
# circuit/formula mistakes.
COMPUTE_EXACT_MFE_PROBABILITY_CHECK = True


def now_utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def print_header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def print_kv(label, value):
    print(f"{label:<38} {value}")


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_qpy_circuits(path):
    with path.open("rb") as handle:
        return list(qpy.load(handle))


def operation_counts(circuit):
    return {name: int(count) for name, count in circuit.count_ops().items()}


def complex_record(value):
    return {"real": float(value.real), "imag": float(value.imag)}


def complex_array_records(values):
    return [
        {"index": int(index), "real": float(value.real), "imag": float(value.imag)}
        for index, value in enumerate(values)
    ]


def complex_matrix_records(matrix):
    return [
        [complex_record(matrix[row, col]) for col in range(matrix.shape[1])]
        for row in range(matrix.shape[0])
    ]


def complex_from_record(record):
    return complex(float(record["real"]), float(record["imag"]))


def pauli_label(pauli_word, num_qubits):
    letters = ["I"] * num_qubits
    for item in pauli_word:
        letters[int(item["qubit"])] = item["pauli"]
    return "".join(reversed(letters))


def prepare_hf_statevector(occupation):
    circuit = QuantumCircuit(len(occupation), name="prepare_hf")
    for qubit, occupied in enumerate(occupation):
        if int(occupied):
            circuit.x(qubit)
    return Statevector.from_instruction(circuit), circuit


def build_non_scalar_trotter_step(group_circuits, circuit_metadata):
    num_qubits = int(circuit_metadata["active_space"]["num_qubits"])
    step = QuantumCircuit(num_qubits, name="direct_non_scalar_trotter_step")
    group_metadata = {
        int(group["group_index"]): group
        for group in circuit_metadata["groups"]
    }
    included = []
    skipped_scalar = []

    for item in circuit_metadata["trotter_step_sequence"]:
        if float(item["time_multiplier"]) != 1.0:
            raise ValueError(
                "This validation script expects first-order full-dt group "
                f"circuits. Found time_multiplier={item['time_multiplier']}."
            )
        group_index = int(item["group_index"])
        metadata = group_metadata[group_index]
        if metadata["source_classification"] == "zero_body_scalar":
            skipped_scalar.append(group_index)
            continue
        step.compose(group_circuits[int(metadata["qpy_circuit_index"])], inplace=True)
        included.append(group_index)

    return step, included, skipped_scalar


def scalar_energy_from_mfe_metadata(mfe_metadata):
    sampling = mfe_metadata.get("stochastic_sampling", {})
    for key in (
        "scalar_energy_excluded_from_lambda",
        "scalar_energy_applied_analytically_to_correlations",
        "scalar_energy_applied_deterministically",
    ):
        if key in sampling:
            return float(sampling[key])
    raise KeyError(
        "Could not find the scalar energy in the MFE metadata. Rerun the "
        "standard overlap builder with the current metadata schema."
    )


def validate_inputs(circuit_metadata, mfe_metadata, mfe_npz):
    circuit_dt = float(circuit_metadata["options"]["dt"])
    if not np.isclose(circuit_dt, KRYLOV_DT, atol=0.0, rtol=1.0e-12):
        raise ValueError(
            f"KRYLOV_DT={KRYLOV_DT} does not match circuit metadata dt={circuit_dt}. "
            "The standard QPY circuits already contain dt, so rerun the circuit "
            "builder if you want a different dt."
        )

    circuit_order = int(circuit_metadata["options"]["trotter_sequence_order"])
    if circuit_order != TROTTER_ORDER:
        raise ValueError(
            f"TROTTER_ORDER={TROTTER_ORDER} does not match circuit metadata "
            f"order={circuit_order}."
        )

    if mfe_metadata["options"]["uqk_mode"] != "standard":
        raise ValueError(
            "This direct validation compares against the standard full-block MFE "
            f"result, but MFE metadata has uqk_mode={mfe_metadata['options']['uqk_mode']!r}."
        )
    if int(mfe_metadata["options"]["krylov_dimension"]) != KRYLOV_DIMENSION:
        raise ValueError(
            "KRYLOV_DIMENSION does not match the saved MFE metadata."
        )
    if not np.isclose(float(mfe_npz["dt"]), KRYLOV_DT, atol=0.0, rtol=1.0e-12):
        raise ValueError("KRYLOV_DT does not match the saved MFE NPZ dt.")
    if mfe_npz["S"].shape != (KRYLOV_DIMENSION, KRYLOV_DIMENSION):
        raise ValueError(
            f"Saved MFE S has shape {mfe_npz['S'].shape}, expected "
            f"{(KRYLOV_DIMENSION, KRYLOV_DIMENSION)}."
        )


def build_physical_non_scalar_step_matrix(pauli_archive, circuit_metadata):
    """Build one first-order Trotter step including identity Pauli terms.

    QPY circuits are what we measure, but they cannot make global identity
    phases directly observable. For direct linear-algebra validation, build the
    grouped Pauli matrices explicitly so the physical Hamiltonian represented by
    the archive is the target.
    """

    num_qubits = int(circuit_metadata["active_space"]["num_qubits"])
    pauli_by_index = {
        int(group["group_index"]): group
        for group in pauli_archive["groups"]
    }
    step = np.eye(2**num_qubits, dtype=np.complex128)

    for item in circuit_metadata["trotter_step_sequence"]:
        group_index = int(item["group_index"])
        group = pauli_by_index[group_index]
        if group["source_classification"] == "zero_body_scalar":
            continue

        labels = []
        coefficients = []
        for term in group["pauli_terms"]:
            labels.append(pauli_label(term["pauli_word"], num_qubits))
            coefficients.append(complex_from_record(term["coefficient"]))

        h_group = SparsePauliOp(labels, coeffs=coefficients).to_matrix()
        step = expm(-1j * KRYLOV_DT * h_group) @ step

    return step


def direct_correlations(hf_state, physical_non_scalar_step, scalar_energy):
    correlations = np.zeros(MAX_CORRELATION_POWER + 1, dtype=np.complex128)
    non_scalar_correlations = np.zeros(MAX_CORRELATION_POWER + 1, dtype=np.complex128)
    hf_vector = hf_state.data
    step_power = np.eye(physical_non_scalar_step.shape[0], dtype=np.complex128)

    for power in range(MAX_CORRELATION_POWER + 1):
        if power > 0:
            step_power = physical_non_scalar_step @ step_power
        non_scalar_value = np.vdot(hf_vector, step_power @ hf_vector)
        scalar_phase = np.exp(-1j * scalar_energy * power * KRYLOV_DT)
        non_scalar_correlations[power] = non_scalar_value
        correlations[power] = scalar_phase * non_scalar_value

    return correlations, non_scalar_correlations


def vacuum_branch_diagnostics(hf_state, non_scalar_step):
    """Check the simplified MFE assumptions for the non-scalar V^k circuit."""

    vacuum_state = Statevector.from_label("0" * non_scalar_step.num_qubits)
    records = []

    for power in range(MAX_CORRELATION_POWER + 1):
        power_circuit = build_power_circuit(non_scalar_step, power)
        evolved_hf = hf_state.evolve(power_circuit)
        evolved_vacuum = vacuum_state.evolve(power_circuit)
        records.append(
            {
                "power": int(power),
                "hf_hf": np.vdot(hf_state.data, evolved_hf.data),
                "vacuum_vacuum": np.vdot(vacuum_state.data, evolved_vacuum.data),
                "vacuum_hf": np.vdot(vacuum_state.data, evolved_hf.data),
                "hf_vacuum": np.vdot(hf_state.data, evolved_vacuum.data),
            }
        )

    return records


def build_power_circuit(non_scalar_step, power):
    circuit = QuantumCircuit(non_scalar_step.num_qubits, name=f"direct_power_{power}")
    for _ in range(power):
        circuit.compose(non_scalar_step, inplace=True)
    return circuit


def hf_probability_exact(circuit, hf_count_key):
    no_measurements = circuit.remove_final_measurements(inplace=False)
    state = Statevector.from_instruction(no_measurements)
    return float(state.probabilities_dict().get(hf_count_key, 0.0))


def exact_mfe_correlations(non_scalar_step, occupation, hf_count_key, scalar_energy):
    correlations = np.zeros(MAX_CORRELATION_POWER + 1, dtype=np.complex128)
    records = []

    for power in range(MAX_CORRELATION_POWER + 1):
        power_circuit = build_power_circuit(non_scalar_step, power)
        templates = build_mfe_templates(power_circuit, occupation, verbose=False)
        f1 = hf_probability_exact(templates[F1_LABEL], hf_count_key)
        f2_plus = hf_probability_exact(templates[F2_PLUS_LABEL], hf_count_key)
        f2_i = hf_probability_exact(templates[F2_I_LABEL], hf_count_key)
        non_scalar_value = complex(
            2.0 * f2_plus - (f1 + 1.0) / 2.0,
            2.0 * f2_i - (f1 + 1.0) / 2.0,
        )
        scalar_phase = np.exp(-1j * scalar_energy * power * KRYLOV_DT)
        correlations[power] = scalar_phase * non_scalar_value
        records.append(
            {
                "power": int(power),
                "F1_exact": f1,
                "F2_plus_exact": f2_plus,
                "F2_i_exact": f2_i,
                "non_scalar_mfe_correlation": complex_record(non_scalar_value),
                "scalar_phase": complex_record(scalar_phase),
                "full_correlation": complex_record(correlations[power]),
            }
        )

    return correlations, records


def assemble_overlap_matrix(correlations):
    matrix = np.empty((KRYLOV_DIMENSION, KRYLOV_DIMENSION), dtype=np.complex128)
    for m in range(KRYLOV_DIMENSION):
        for n in range(KRYLOV_DIMENSION):
            diff = n - m
            matrix[m, n] = (
                correlations[diff]
                if diff >= 0
                else np.conjugate(correlations[-diff])
            )
    return matrix


def comparison_metrics(left, right):
    delta = left - right
    return {
        "max_abs_difference": float(np.max(np.abs(delta))),
        "frobenius_difference": float(np.linalg.norm(delta)),
        "left_hermiticity_error": float(np.linalg.norm(left - left.conj().T)),
        "right_hermiticity_error": float(np.linalg.norm(right - right.conj().T)),
    }


def main():
    molecule_metadata = load_json(INPUT_MOLECULE_METADATA_JSON)
    circuit_metadata = load_json(INPUT_CIRCUIT_METADATA_JSON)
    pauli_archive = load_json(INPUT_GROUPED_PAULI_JSON)
    mfe_metadata = load_json(INPUT_MFE_METADATA_JSON)
    mfe_npz = np.load(INPUT_MFE_NPZ)
    validate_inputs(circuit_metadata, mfe_metadata, mfe_npz)

    occupation, hf_count_key = validate_hf_metadata(molecule_metadata)
    if hf_count_key != circuit_metadata["hf_reference"][
        "qiskit_counts_bitstring_if_measured_q_to_c_same_index"
    ]:
        raise ValueError("HF count key mismatch between molecule and circuit metadata.")

    group_circuits = load_qpy_circuits(INPUT_QPY)
    non_scalar_step, included_groups, skipped_scalar_groups = (
        build_non_scalar_trotter_step(group_circuits, circuit_metadata)
    )
    hf_state, hf_prep = prepare_hf_statevector(occupation)
    scalar_energy = scalar_energy_from_mfe_metadata(mfe_metadata)

    print_header("Direct Statevector UQK S Validation")
    print(
        "This script computes exact noiseless C_k from statevectors and compares\n"
        "the resulting Toeplitz S matrix to the saved standard MFE result."
    )
    print_header("Hard-Coded Options")
    print_kv("Molecule metadata:", INPUT_MOLECULE_METADATA_JSON)
    print_kv("Grouped QPY archive:", INPUT_QPY)
    print_kv("Circuit metadata:", INPUT_CIRCUIT_METADATA_JSON)
    print_kv("Grouped Pauli archive:", INPUT_GROUPED_PAULI_JSON)
    print_kv("Saved MFE NPZ:", INPUT_MFE_NPZ)
    print_kv("Saved MFE metadata:", INPUT_MFE_METADATA_JSON)
    print_kv("Krylov dimension M:", KRYLOV_DIMENSION)
    print_kv("Max correlation power:", MAX_CORRELATION_POWER)
    print_kv("Guard dt:", KRYLOV_DT)
    print_kv("Trotter order:", TROTTER_ORDER)
    print_kv("Exact MFE probability check:", COMPUTE_EXACT_MFE_PROBABILITY_CHECK)
    print_kv("Output NPZ:", OUTPUT_COMPARISON_NPZ)
    print_kv("Output JSON:", OUTPUT_COMPARISON_JSON)

    print_header("Circuit And Reference Summary")
    print_kv("Qubits:", non_scalar_step.num_qubits)
    print_kv("HF occupation n_p:", occupation)
    print_kv("HF count key:", hf_count_key)
    print_kv("HF prep ops:", operation_counts(hf_prep))
    print_kv("Loaded group circuits:", len(group_circuits))
    print_kv("Non-scalar groups included:", len(included_groups))
    print_kv("Scalar groups skipped:", skipped_scalar_groups)
    print_kv("Scalar energy:", f"{scalar_energy:.12f}")
    print_kv("Non-scalar step depth:", non_scalar_step.depth())
    print_kv("Non-scalar step size:", non_scalar_step.size())
    print_kv("Non-scalar step ops:", operation_counts(non_scalar_step))

    physical_non_scalar_step = build_physical_non_scalar_step_matrix(
        pauli_archive,
        circuit_metadata,
    )
    direct_c, direct_non_scalar_c = direct_correlations(
        hf_state,
        physical_non_scalar_step,
        scalar_energy,
    )
    vacuum_diagnostics = vacuum_branch_diagnostics(hf_state, non_scalar_step)
    s_direct = assemble_overlap_matrix(direct_c)
    s_mfe_saved = np.asarray(mfe_npz["S"], dtype=np.complex128)
    saved_c = np.asarray(mfe_npz["correlations"], dtype=np.complex128)
    saved_c_for_m = saved_c[:KRYLOV_DIMENSION]
    saved_metrics = comparison_metrics(s_direct, s_mfe_saved)

    exact_mfe_c = None
    exact_mfe_records = []
    exact_mfe_metrics = None
    s_exact_mfe = None
    if COMPUTE_EXACT_MFE_PROBABILITY_CHECK:
        exact_mfe_c, exact_mfe_records = exact_mfe_correlations(
            non_scalar_step,
            occupation,
            hf_count_key,
            scalar_energy,
        )
        s_exact_mfe = assemble_overlap_matrix(exact_mfe_c)
        exact_mfe_metrics = comparison_metrics(s_direct, s_exact_mfe)

    print_header("Correlation Values")
    print(f"{'k':>3} {'direct C_k':>28} {'saved MFE C_k':>28}")
    print("-" * 78)
    for power in range(KRYLOV_DIMENSION):
        print(
            f"{power:>3} "
            f"{direct_c[power].real:+.10f}{direct_c[power].imag:+.10f}j "
            f"{saved_c[power].real:+.10f}{saved_c[power].imag:+.10f}j"
        )

    print_header("MFE Vacuum-Branch Assumption Check")
    print(
        "The QPY circuits may give the vacuum branch a phase because identity\n"
        "Pauli terms in non-scalar groups are not measurable as standalone\n"
        "global phases. Raw MFE is intentionally a vacuum-reference measurement:\n"
        "that phase cancels in the extracted physical grouped-Pauli amplitude.\n"
        "The HF/vacuum cross terms should still vanish."
    )
    print(
        f"{'k':>3} {'<vac|V^k|vac>':>28} {'<vac|V^k|HF>':>28} {'<HF|V^k|vac>':>28}"
    )
    print("-" * 94)
    for record in vacuum_diagnostics:
        vv = record["vacuum_vacuum"]
        vh = record["vacuum_hf"]
        hv = record["hf_vacuum"]
        print(
            f"{record['power']:>3} "
            f"{vv.real:+.8f}{vv.imag:+.8f}j "
            f"{vh.real:+.8f}{vh.imag:+.8f}j "
            f"{hv.real:+.8f}{hv.imag:+.8f}j"
        )

    print_header("Overlap Matrices")
    print("S_direct:")
    print(s_direct)
    print("\nS_MFE_saved:")
    print(s_mfe_saved)
    if s_exact_mfe is not None:
        print("\nS_MFE_exact_probability:")
        print(s_exact_mfe)

    print_header("Comparison Metrics")
    print("Physical grouped-Pauli direct statevector vs saved finite-shot raw MFE:")
    for key, value in saved_metrics.items():
        print_kv(key + ":", f"{value:.12e}")
    if exact_mfe_metrics is not None:
        print("\nPhysical grouped-Pauli direct statevector vs exact-probability raw MFE:")
        for key, value in exact_mfe_metrics.items():
            print_kv(key + ":", f"{value:.12e}")

    OUTPUT_COMPARISON_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_COMPARISON_NPZ,
        S_direct=s_direct,
        C_direct=direct_c,
        C_direct_non_scalar=direct_non_scalar_c,
        S_MFE_saved=s_mfe_saved,
        C_MFE_saved=saved_c,
        S_MFE_exact_probability=(
            s_exact_mfe if s_exact_mfe is not None else np.array([], dtype=np.complex128)
        ),
        C_MFE_exact_probability=(
            exact_mfe_c if exact_mfe_c is not None else np.array([], dtype=np.complex128)
        ),
    )

    metadata = {
        "schema_version": 1,
        "generated_at_utc": now_utc(),
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "inputs": {
            "molecule_metadata_json": str(INPUT_MOLECULE_METADATA_JSON),
            "qpy": str(INPUT_QPY),
            "circuit_metadata_json": str(INPUT_CIRCUIT_METADATA_JSON),
            "grouped_pauli_json": str(INPUT_GROUPED_PAULI_JSON),
            "mfe_npz": str(INPUT_MFE_NPZ),
            "mfe_metadata_json": str(INPUT_MFE_METADATA_JSON),
        },
        "outputs": {
            "npz": str(OUTPUT_COMPARISON_NPZ),
            "json": str(OUTPUT_COMPARISON_JSON),
        },
        "options": {
            "krylov_dimension": KRYLOV_DIMENSION,
            "krylov_dt_guard": KRYLOV_DT,
            "trotter_order": TROTTER_ORDER,
            "compute_exact_mfe_probability_check": COMPUTE_EXACT_MFE_PROBABILITY_CHECK,
        },
        "conventions": {
            "toeplitz": "S_mn=C_(n-m), C_-k=conj(C_k)",
            "standard_path": (
                "Direct validation builds the physical grouped-Pauli Trotter "
                "block including identity terms. Exact MFE uses the saved QPY "
                "circuits because those are what hardware would measure."
            ),
            "reference_branch_correction": (
                "No simulated reference-branch correction is applied. Raw MFE "
                "is a vacuum-reference measurement, and the zero-body scalar "
                "phase exp(-i E_scalar k dt) is applied analytically."
            ),
            "dt_guard": (
                "KRYLOV_DT is checked against QPY metadata; it does not rescale "
                "the already-transpiled standard circuits."
            ),
        },
        "molecule": molecule_metadata["molecule"],
        "active_space": molecule_metadata["active_space"],
        "hf_reference": molecule_metadata["hf_reference"],
        "scalar_energy": scalar_energy,
        "included_non_scalar_group_indices": included_groups,
        "skipped_scalar_group_indices": skipped_scalar_groups,
        "non_scalar_step": {
            "depth": int(non_scalar_step.depth()),
            "size": int(non_scalar_step.size()),
            "operation_counts": operation_counts(non_scalar_step),
        },
        "correlations_direct": complex_array_records(direct_c),
        "correlations_direct_non_scalar": complex_array_records(direct_non_scalar_c),
        "correlations_mfe_saved": complex_array_records(saved_c[:KRYLOV_DIMENSION]),
        "overlap_direct": complex_matrix_records(s_direct),
        "overlap_mfe_saved": complex_matrix_records(s_mfe_saved),
        "saved_mfe_comparison_metrics": saved_metrics,
        "exact_mfe_probability_records": exact_mfe_records,
        "exact_mfe_probability_comparison_metrics": exact_mfe_metrics,
        "vacuum_branch_diagnostics": [
            {
                "power": record["power"],
                "hf_hf": complex_record(record["hf_hf"]),
                "vacuum_vacuum": complex_record(record["vacuum_vacuum"]),
                "vacuum_hf": complex_record(record["vacuum_hf"]),
                "hf_vacuum": complex_record(record["hf_vacuum"]),
            }
            for record in vacuum_diagnostics
        ],
    }
    with OUTPUT_COMPARISON_JSON.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print_header("Saved Outputs")
    print_kv("NPZ:", OUTPUT_COMPARISON_NPZ)
    print_kv("JSON:", OUTPUT_COMPARISON_JSON)


if __name__ == "__main__":
    main()
