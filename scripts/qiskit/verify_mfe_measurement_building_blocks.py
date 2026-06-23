# Manual run:
#   conda activate qiskit_env_v1
#   python scripts/qiskit/verify_mfe_measurement_building_blocks.py
#
# Summary:
#   Verify the MFE measurement building blocks without constructing full Krylov
#   matrices. This checks the HF bitstring convention, builds F1/F2_plus/F2_i
#   templates for a supplied evolution circuit V, and uses local Aer counts for
#   two analytically known cases.
#
# Hard-coded options:
#   MOLECULE_METADATA_JSON = data/molecules/h4_linear_sto3g_metadata.json
#   CIRCUIT_QPY = circuits/transpiled/h4_linear_sto3g_grouped_evolution.qpy
#   CIRCUIT_METADATA_JSON = circuits/transpiled/h4_linear_sto3g_grouped_evolution_metadata.json
#   SHOTS = 20000
#   SIMULATOR_SEED = 917
#   PHASE_RADIANS = 0.4

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from qiskit import QuantumCircuit, qpy
from qiskit_aer import AerSimulator


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from mfe_measurement_building_blocks import (  # noqa: E402
    F1_LABEL,
    F2_I_LABEL,
    F2_PLUS_LABEL,
    build_mfe_templates,
    estimate_z_from_counts,
    validate_hf_metadata,
)


MOLECULE_METADATA_JSON = (
    REPO_ROOT / "data" / "molecules" / "h4_linear_sto3g_metadata.json"
)
CIRCUIT_QPY = (
    REPO_ROOT / "circuits" / "transpiled" / "h4_linear_sto3g_grouped_evolution.qpy"
)
CIRCUIT_METADATA_JSON = (
    REPO_ROOT
    / "circuits"
    / "transpiled"
    / "h4_linear_sto3g_grouped_evolution_metadata.json"
)

SHOTS = 20_000
SIMULATOR_SEED = 917
PHASE_RADIANS = 0.4
ESTIMATE_ABS_TOLERANCE = 0.035


def print_header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def print_kv(label, value):
    print(f"{label:<34} {value}")


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_qpy_circuits(path):
    with path.open("rb") as handle:
        return list(qpy.load(handle))


def run_templates_on_aer(templates):
    simulator = AerSimulator(seed_simulator=SIMULATOR_SEED)
    ordered_labels = [F1_LABEL, F2_PLUS_LABEL, F2_I_LABEL]
    circuits = [templates[label] for label in ordered_labels]
    result = simulator.run(circuits, shots=SHOTS).result()
    return {
        label: result.get_counts(index)
        for index, label in enumerate(ordered_labels)
    }


def assert_close_complex(label, estimate, expected):
    real_error = abs(estimate.real - expected.real)
    imag_error = abs(estimate.imag - expected.imag)
    if real_error > ESTIMATE_ABS_TOLERANCE or imag_error > ESTIMATE_ABS_TOLERANCE:
        raise AssertionError(
            f"{label} estimate {estimate} differs from expected {expected} by "
            f"real_error={real_error:.6f}, imag_error={imag_error:.6f}."
        )


def summarize_case(label, estimate, expected, counts_by_label, hf_count_key):
    print_header(label)
    for experiment_label in [F1_LABEL, F2_PLUS_LABEL, F2_I_LABEL]:
        counts = counts_by_label[experiment_label]
        hf_returns = counts.get(hf_count_key, 0)
        print_kv(f"{experiment_label} HF returns:", f"{hf_returns} / {SHOTS}")
    print_kv("F1:", f"{estimate.f1:.8f}")
    print_kv("F2_plus:", f"{estimate.f2_plus:.8f}")
    print_kv("F2_i:", f"{estimate.f2_i:.8f}")
    print_kv("Estimated z:", f"{estimate.z.real:+.8f}{estimate.z.imag:+.8f}j")
    print_kv("Expected z:", f"{expected.real:+.8f}{expected.imag:+.8f}j")
    print_kv("Abs error Re/Im:", f"{abs(estimate.real - expected.real):.8f} / {abs(estimate.imag - expected.imag):.8f}")


def identity_evolution(num_qubits):
    return QuantumCircuit(num_qubits, name="identity_V")


def hf_phase_evolution(num_qubits, occupied_qubits):
    """Return V with V|vac> = |vac> and V|HF> = exp(i PHASE_RADIANS)|HF>."""

    circuit = QuantumCircuit(num_qubits, name="hf_phase_V")
    circuit.p(PHASE_RADIANS, occupied_qubits[0])
    return circuit


def main():
    print_header("MFE Measurement Building Block Verification")
    print_kv("Molecule metadata JSON:", MOLECULE_METADATA_JSON)
    print_kv("Circuit QPY archive:", CIRCUIT_QPY)
    print_kv("Circuit metadata JSON:", CIRCUIT_METADATA_JSON)
    print_kv("Shots:", SHOTS)
    print_kv("Simulator seed:", SIMULATOR_SEED)
    print_kv("Phase test radians:", PHASE_RADIANS)

    molecule_metadata = load_json(MOLECULE_METADATA_JSON)
    circuit_metadata = load_json(CIRCUIT_METADATA_JSON)
    occupation, hf_count_key = validate_hf_metadata(molecule_metadata)
    occupied_qubits = [index for index, bit in enumerate(occupation) if bit]
    num_qubits = len(occupation)

    print_header("HF Convention Check")
    print_kv("Occupation n_p list:", occupation)
    print_kv("Occupied qubits:", occupied_qubits)
    print_kv("Derived Qiskit HF key:", hf_count_key)
    print_kv(
        "Circuit metadata HF key:",
        circuit_metadata["hf_reference"]["qiskit_counts_bitstring_if_measured_q_to_c_same_index"],
    )
    if hf_count_key != circuit_metadata["hf_reference"]["qiskit_counts_bitstring_if_measured_q_to_c_same_index"]:
        raise ValueError("HF count key mismatch between molecule and circuit metadata.")

    archived_circuits = load_qpy_circuits(CIRCUIT_QPY)
    representative_index = 1 if len(archived_circuits) > 1 else 0
    representative_templates = build_mfe_templates(
        archived_circuits[representative_index],
        occupation,
    )
    print_header("Archived Evolution Circuit Template Smoke Test")
    print_kv("Archived circuits loaded:", len(archived_circuits))
    print_kv("Representative QPY index:", representative_index)
    for label, circuit in representative_templates.items():
        print_kv(f"{label} template depth:", circuit.depth())
        print_kv(f"{label} template ops:", dict(circuit.count_ops()))

    identity_templates = build_mfe_templates(identity_evolution(num_qubits), occupation)
    identity_counts = run_templates_on_aer(identity_templates)
    identity_estimate = estimate_z_from_counts(identity_counts, hf_count_key)
    identity_expected = complex(1.0, 0.0)
    summarize_case(
        "Identity Evolution: Expected z = 1",
        identity_estimate,
        identity_expected,
        identity_counts,
        hf_count_key,
    )
    assert_close_complex("Identity evolution", identity_estimate.z, identity_expected)

    phase_templates = build_mfe_templates(
        hf_phase_evolution(num_qubits, occupied_qubits),
        occupation,
    )
    phase_counts = run_templates_on_aer(phase_templates)
    phase_estimate = estimate_z_from_counts(phase_counts, hf_count_key)
    phase_expected = complex(math.cos(PHASE_RADIANS), math.sin(PHASE_RADIANS))
    summarize_case(
        "HF Phase Evolution: Expected z = exp(i theta)",
        phase_estimate,
        phase_expected,
        phase_counts,
        hf_count_key,
    )
    assert_close_complex("HF phase evolution", phase_estimate.z, phase_expected)

    print_header("Result")
    print(
        "MFE measurement templates and arithmetic passed the local simulator "
        "checks. No credentials, QPU backend, or Krylov matrix construction were used."
    )


if __name__ == "__main__":
    main()
