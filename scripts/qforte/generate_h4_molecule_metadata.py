# Manual run:
#   conda activate qfe_env_v1
#   python scripts/qforte/generate_h4_molecule_metadata.py
#
# Summary:
#   Build a linear H4/STO-3G molecule with qforte/Psi4 and write small,
#   human-readable metadata for downstream qforte-qiskit scripts. This script
#   intentionally does not serialize Hamiltonian/operator objects.
#
# Hard-coded options:
#   QFORTE_SOURCE_ROOT = /Users/nstair/Src/my_qforte/qforte
#   GEOMETRY = linear H4 with 1.5 Angstrom spacing
#   BASIS = sto-3g
#   CHARGE = 0
#   MULTIPLICITY = 1
#   SYMMETRY = c1
#   REQUESTED_FCI_ROOTS = 8
#   OUTPUT_JSON = data/molecules/h4_linear_sto3g_metadata.json

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from traceback import format_exception_only


REPO_ROOT = Path(__file__).resolve().parents[2]
QFORTE_SOURCE_ROOT = Path("/Users/nstair/Src/my_qforte/qforte")
QFORTE_SOURCE_PATH = QFORTE_SOURCE_ROOT / "src"

OUTPUT_JSON = REPO_ROOT / "data" / "molecules" / "h4_linear_sto3g_metadata.json"
PSI4_OUTPUT_STEM = REPO_ROOT / "data" / "molecules" / "h4_linear_sto3g_psi4"

BASIS = "sto-3g"
CHARGE = 0
MULTIPLICITY = 1
SYMMETRY = "c1"
REQUESTED_FCI_ROOTS = 8
SPACING_ANGSTROM = 1.0
GEOMETRY = [
    ("H", (0.0, 0.0, 1.0)),
    ("H", (0.0, 0.0, 2.0)),
    ("H", (0.0, 0.0, 3.0)),
    ("H", (0.0, 0.0, 4.0)),
]


def now_utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_float(value):
    if value is None:
        return None
    return float(value)


def json_int(value):
    if value is None:
        return None
    return int(value)


def list_float(values):
    return [float(value) for value in values]


def build_molecule(qforte, nroots_fci):
    return qforte.system_factory(
        system_type="molecule",
        build_type="psi4",
        basis=BASIS,
        mol_geometry=GEOMETRY,
        symmetry=SYMMETRY,
        multiplicity=MULTIPLICITY,
        charge=CHARGE,
        nroots_fci=nroots_fci,
        run_mp2=False,
        run_cisd=False,
        run_ccsd=False,
        run_fci=True,
        build_qb_ham=False,
        store_mo_ints=True,
        filename=str(PSI4_OUTPUT_STEM),
    )


def fci_status(requested_roots, returned_roots, fallback_reason):
    if fallback_reason:
        return "fallback_after_requested_roots_failed"
    if len(returned_roots) == requested_roots:
        return "all_requested_roots_returned"
    if len(returned_roots) > 0:
        return "fewer_roots_returned_than_requested"
    return "no_fci_roots_returned"


def main():
    if str(QFORTE_SOURCE_PATH) not in sys.path:
        sys.path.insert(0, str(QFORTE_SOURCE_PATH))

    import qforte

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    fallback_reason = None
    nroots_used = REQUESTED_FCI_ROOTS
    try:
        mol = build_molecule(qforte, REQUESTED_FCI_ROOTS)
    except Exception as exc:
        fallback_reason = "".join(format_exception_only(type(exc), exc)).strip()
        print("Requested 8-root FCI build failed; falling back to one FCI root.")
        print(fallback_reason)
        nroots_used = 1
        mol = build_molecule(qforte, nroots_used)

    hf_reference = [int(bit) for bit in mol.hf_reference]
    occupied_spin_orbitals = [idx for idx, bit in enumerate(hf_reference) if bit]
    num_qubits = len(hf_reference)
    num_spatial_orbitals = num_qubits // 2
    num_electrons = sum(hf_reference)
    fci_roots = list_float(mol.fci_energy_list)
    status = fci_status(REQUESTED_FCI_ROOTS, fci_roots, fallback_reason)

    warning = None
    if status != "all_requested_roots_returned":
        warning = (
            f"Requested {REQUESTED_FCI_ROOTS} FCI roots, but "
            f"recorded {len(fci_roots)} root(s)."
        )
        if fallback_reason:
            warning += f" Initial 8-root failure: {fallback_reason}"

    metadata = {
        "schema_version": 1,
        "generated_at_utc": now_utc(),
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "qforte_source_root": str(QFORTE_SOURCE_ROOT),
        "qforte_source_path": str(QFORTE_SOURCE_PATH),
        "qforte_import_file": getattr(qforte, "__file__", None),
        "qforte_version": getattr(qforte, "__version__", None),
        "backend": "psi4",
        "molecule": {
            "label": "linear_h4_sto3g",
            "geometry_units": "angstrom",
            "spacing_angstrom": SPACING_ANGSTROM,
            "geometry": [
                {"atom": atom, "xyz": [float(x), float(y), float(z)]}
                for atom, (x, y, z) in GEOMETRY
            ],
            "basis": BASIS,
            "charge": CHARGE,
            "multiplicity": MULTIPLICITY,
            "symmetry": SYMMETRY,
            "point_group": list(getattr(mol, "point_group", [])),
        },
        "active_space": {
            "num_electrons": json_int(num_electrons),
            "num_spatial_orbitals": json_int(num_spatial_orbitals),
            "num_spin_orbitals": json_int(num_qubits),
            "num_qubits": json_int(num_qubits),
            "frozen_core_orbitals": json_int(getattr(mol, "frozen_core", 0)),
            "frozen_virtual_orbitals": json_int(getattr(mol, "frozen_virtual", 0)),
        },
        "energies_hartree": {
            "nuclear_repulsion": json_float(mol.nuclear_repulsion_energy),
            "frozen_core": json_float(getattr(mol, "frozen_core_energy", 0.0)),
            "hf": json_float(mol.hf_energy),
            "fci_roots": fci_roots,
        },
        "fci": {
            "num_roots_requested": REQUESTED_FCI_ROOTS,
            "num_roots_used_for_build": nroots_used,
            "num_roots_returned": len(fci_roots),
            "status": status,
            "warning": warning,
            "fallback_reason": fallback_reason,
        },
        "hf_reference": {
            "occupation_little_endian": hf_reference,
            "occupied_spin_orbitals": occupied_spin_orbitals,
            "orbital_to_qubit_order": "spin orbital p maps to qubit p",
            "qforte_occupation_bitstring": "".join(str(bit) for bit in hf_reference),
            "qiskit_counts_bitstring_if_measured_q_to_c_same_index": "".join(
                str(bit) for bit in reversed(hf_reference)
            ),
            "endianness_note": (
                "The occupation list is indexed as n_p for spin orbital/qubit p. "
                "Qiskit count strings are usually printed as c[n-1]...c[0] "
                "when qubit q[i] is measured into classical bit c[i]."
            ),
        },
        "files": {
            "metadata_json": str(OUTPUT_JSON),
            "psi4_output": str(PSI4_OUTPUT_STEM) + ".out",
        },
    }

    with OUTPUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Wrote metadata: {OUTPUT_JSON}")
    print(f"HF energy:      {metadata['energies_hartree']['hf']:.12f}")
    if fci_roots:
        print(f"FCI root 0:     {fci_roots[0]:.12f}")
    print(f"FCI root status: {status}")
    if warning:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
