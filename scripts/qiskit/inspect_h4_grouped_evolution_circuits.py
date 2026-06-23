# Manual run:
#   conda activate qiskit_env_v1
#   python scripts/qiskit/inspect_h4_grouped_evolution_circuits.py
#
# Summary:
#   Load the QPY grouped-evolution circuit archive and metadata sidecar, then
#   print a readable report of options, circuit counts, depth statistics, a
#   compact group table, and one representative detailed group.
#
# Hard-coded options:
#   INPUT_QPY = circuits/transpiled/h4_linear_sto3g_grouped_evolution.qpy
#   INPUT_METADATA_JSON = circuits/transpiled/h4_linear_sto3g_grouped_evolution_metadata.json
#   DETAILED_GROUP_INDEX = first non-scalar group if None
#   MAX_GROUP_TABLE_ROWS = 25

from __future__ import annotations

import json
from pathlib import Path

from qiskit import qpy


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_QPY = (
    REPO_ROOT / "circuits" / "transpiled" / "h4_linear_sto3g_grouped_evolution.qpy"
)
INPUT_METADATA_JSON = (
    REPO_ROOT
    / "circuits"
    / "transpiled"
    / "h4_linear_sto3g_grouped_evolution_metadata.json"
)
DETAILED_GROUP_INDEX = None
MAX_GROUP_TABLE_ROWS = 25


def print_header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def print_kv(label, value):
    print(f"{label:<34} {value}")


def load_qpy_circuits(path):
    with path.open("rb") as handle:
        return list(qpy.load(handle))


def choose_detail_group(metadata):
    if DETAILED_GROUP_INDEX is not None:
        return int(DETAILED_GROUP_INDEX)
    for group in metadata["groups"]:
        if group["source_classification"] != "zero_body_scalar":
            return int(group["group_index"])
    return int(metadata["groups"][0]["group_index"])


def main():
    if not INPUT_QPY.exists():
        raise FileNotFoundError(
            f"Missing QPY archive: {INPUT_QPY}. "
            "Run scripts/qiskit/build_h4_grouped_evolution_circuits.py first."
        )
    if not INPUT_METADATA_JSON.exists():
        raise FileNotFoundError(
            f"Missing circuit metadata JSON: {INPUT_METADATA_JSON}. "
            "Run scripts/qiskit/build_h4_grouped_evolution_circuits.py first."
        )

    with INPUT_METADATA_JSON.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    circuits = load_qpy_circuits(INPUT_QPY)

    print_header("Grouped Evolution Circuit Archive")
    print_kv("QPY archive:", INPUT_QPY)
    print_kv("Metadata JSON:", INPUT_METADATA_JSON)
    print_kv("Qiskit version at build:", metadata.get("qiskit_version"))
    print_kv("Circuits in QPY:", len(circuits))
    print_kv("Circuits in metadata:", metadata["circuit_archive"]["num_circuits"])
    if len(circuits) != metadata["circuit_archive"]["num_circuits"]:
        raise ValueError("QPY circuit count does not match metadata.")

    print_header("Hard-Coded Options Used To Build Archive")
    for key, value in metadata["options"].items():
        print_kv(key + ":", value)

    print_header("Molecule And Source Pauli Archive")
    molecule = metadata["molecule"]
    active = metadata["active_space"]
    source_diag = metadata["source_archive_diagnostics"]
    print_kv("Molecule label:", molecule.get("label"))
    print_kv("Basis:", molecule.get("basis"))
    print_kv("Qubits:", active.get("num_qubits"))
    print_kv("Electrons:", active.get("num_electrons"))
    print_kv("Source groups:", source_diag.get("grouped_pauli_groups"))
    print_kv("Source total Pauli terms:", source_diag.get("total_pauli_terms"))
    print_kv("Source identity terms:", source_diag.get("identity_pauli_terms"))
    print_kv("Input SHA256:", metadata.get("input_grouped_pauli_sha256"))

    print_header("Depth Statistics")
    for key, value in metadata["depth_statistics"].items():
        print_kv(key + ":", value)

    print_header("Compact Group Table")
    print(
        f"{'grp':>4} {'class':<16} {'paulis':>6} {'raw_d':>6} "
        f"{'tr_d':>6} {'tr_ops':<30}"
    )
    print("-" * 78)
    rows = metadata["groups"][:MAX_GROUP_TABLE_ROWS]
    for group in rows:
        print(
            f"{group['group_index']:>4} "
            f"{group['source_classification']:<16} "
            f"{group['num_pauli_terms']:>6} "
            f"{group['raw']['depth']:>6} "
            f"{group['transpiled']['depth']:>6} "
            f"{str(group['transpiled']['operation_counts']):<30}"
        )
    if len(metadata["groups"]) > MAX_GROUP_TABLE_ROWS:
        print(f"... {len(metadata['groups']) - MAX_GROUP_TABLE_ROWS} more groups omitted")

    detail_group_index = choose_detail_group(metadata)
    detail_record = metadata["groups"][detail_group_index]
    detail_circuit = circuits[detail_record["qpy_circuit_index"]]
    print_header(f"Detailed Representative Group {detail_group_index}")
    print_kv("Source classification:", detail_record["source_classification"])
    print_kv("Pauli terms:", detail_record["num_pauli_terms"])
    print_kv("Evolution method:", detail_record.get("evolution_method"))
    print_kv("PauliEvolution synthesis:", detail_record.get("pauli_evolution_synthesis"))
    print_kv("QPY circuit index:", detail_record["qpy_circuit_index"])
    print_kv("Raw summary:", detail_record["raw"])
    print_kv("Transpiled summary:", detail_record["transpiled"])
    print("\nCircuit drawing:")
    print(detail_circuit.draw(output="text", fold=120))

    print_header("Conventions")
    for key, value in metadata["conventions"].items():
        print_kv(key + ":", value)
    print("\nInspection complete.")


if __name__ == "__main__":
    main()
