# Manual run:
#   conda activate qfe_env_v1
#   python scripts/qforte/verify_h4_molecule_metadata.py
#
# Summary:
#   Read the H4 metadata JSON created by generate_h4_molecule_metadata.py and
#   verify that downstream scripts have the molecule, energy, FCI-root, and
#   Hartree-Fock reference fields they need.
#
# Hard-coded options:
#   METADATA_JSON = data/molecules/h4_linear_sto3g_metadata.json
#   EXPECTED_REQUESTED_FCI_ROOTS = 8

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
METADATA_JSON = REPO_ROOT / "data" / "molecules" / "h4_linear_sto3g_metadata.json"
EXPECTED_REQUESTED_FCI_ROOTS = 8


REQUIRED_TOP_LEVEL_KEYS = [
    "schema_version",
    "generated_at_utc",
    "qforte_source_root",
    "backend",
    "molecule",
    "active_space",
    "energies_hartree",
    "fci",
    "hf_reference",
]

REQUIRED_MOLECULE_KEYS = [
    "geometry",
    "geometry_units",
    "basis",
    "charge",
    "multiplicity",
    "symmetry",
]

REQUIRED_ACTIVE_SPACE_KEYS = [
    "num_electrons",
    "num_spatial_orbitals",
    "num_spin_orbitals",
    "num_qubits",
]

REQUIRED_ENERGY_KEYS = [
    "nuclear_repulsion",
    "hf",
    "fci_roots",
]

REQUIRED_HF_REFERENCE_KEYS = [
    "occupation_little_endian",
    "occupied_spin_orbitals",
    "orbital_to_qubit_order",
    "qforte_occupation_bitstring",
    "qiskit_counts_bitstring_if_measured_q_to_c_same_index",
]


def require_keys(mapping, keys, label):
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{label} is missing required keys: {missing}")


def main():
    if not METADATA_JSON.exists():
        raise FileNotFoundError(f"Missing metadata JSON: {METADATA_JSON}")

    with METADATA_JSON.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    require_keys(metadata, REQUIRED_TOP_LEVEL_KEYS, "metadata")
    require_keys(metadata["molecule"], REQUIRED_MOLECULE_KEYS, "molecule")
    require_keys(metadata["active_space"], REQUIRED_ACTIVE_SPACE_KEYS, "active_space")
    require_keys(metadata["energies_hartree"], REQUIRED_ENERGY_KEYS, "energies_hartree")
    require_keys(metadata["hf_reference"], REQUIRED_HF_REFERENCE_KEYS, "hf_reference")

    fci = metadata["fci"]
    fci_roots = metadata["energies_hartree"]["fci_roots"]
    hf_reference = metadata["hf_reference"]["occupation_little_endian"]
    num_qubits = metadata["active_space"]["num_qubits"]

    if fci["num_roots_requested"] != EXPECTED_REQUESTED_FCI_ROOTS:
        raise ValueError(
            "Metadata did not record the expected 8-root FCI request: "
            f"{fci['num_roots_requested']}"
        )
    if fci["num_roots_returned"] != len(fci_roots):
        raise ValueError("FCI root count does not match fci_roots length.")
    if fci["num_roots_returned"] != EXPECTED_REQUESTED_FCI_ROOTS and not fci["warning"]:
        raise ValueError("Missing warning for incomplete FCI root availability.")
    if len(hf_reference) != num_qubits:
        raise ValueError("HF occupation length does not match num_qubits.")
    if sum(int(bit) for bit in hf_reference) != metadata["active_space"]["num_electrons"]:
        raise ValueError("HF occupation count does not match num_electrons.")

    print(f"Verified metadata: {METADATA_JSON}")
    print(f"Basis:             {metadata['molecule']['basis']}")
    print(f"Num qubits:        {metadata['active_space']['num_qubits']}")
    print(f"HF energy:         {metadata['energies_hartree']['hf']:.12f}")
    if fci_roots:
        print(f"FCI root 0:        {fci_roots[0]:.12f}")
    print(
        "FCI roots:         "
        f"{fci['num_roots_returned']} returned / "
        f"{fci['num_roots_requested']} requested"
    )
    if fci["warning"]:
        print(f"FCI warning:       {fci['warning']}")


if __name__ == "__main__":
    main()
