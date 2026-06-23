# Manual run:
#   conda activate qfe_env_v1
#   python scripts/openfermion/convert_h4_hermitian_pairs_to_openfermion.py
#
# Summary:
#   Read the qforte Hermitian-pair JSON archive, convert each grouped
#   second-quantized term to an OpenFermion FermionOperator, apply OpenFermion's
#   Jordan-Wigner transform, and write grouped Pauli terms for Qiskit-side
#   circuit construction. This script intentionally trusts OpenFermion's
#   Jordan-Wigner implementation and does not validate against qforte JW.
#
# Hard-coded options:
#   INPUT_JSON = data/hamiltonians/h4_linear_sto3g_hermitian_pairs.json
#   OUTPUT_JSON = data/hamiltonians/h4_linear_sto3g_grouped_paulis.json
#   COEFFICIENT_TOLERANCE = 1.0e-12

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from openfermion import FermionOperator
from openfermion.transforms import jordan_wigner


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_JSON = REPO_ROOT / "data" / "hamiltonians" / "h4_linear_sto3g_hermitian_pairs.json"
OUTPUT_JSON = REPO_ROOT / "data" / "hamiltonians" / "h4_linear_sto3g_grouped_paulis.json"
COEFFICIENT_TOLERANCE = 1.0e-12


def now_utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def complex_record(value):
    zval = complex(value)
    return {
        "real": float(zval.real),
        "imag": float(zval.imag),
    }


def complex_from_record(record):
    return complex(float(record["real"]), float(record["imag"]))


def ladder_ops_from_sq_term(sq_term):
    ladder_ops = []
    for idx in sq_term["creators"]:
        ladder_ops.append((int(idx), 1))
    for idx in sq_term["annihilators"]:
        ladder_ops.append((int(idx), 0))
    return tuple(ladder_ops)


def ladder_ops_record(ladder_ops):
    return [
        {
            "mode": int(mode),
            "action": int(action),
            "action_label": "create" if int(action) == 1 else "annihilate",
        }
        for mode, action in ladder_ops
    ]


def pauli_word_record(term_tuple):
    return [
        {
            "qubit": int(qubit),
            "pauli": str(pauli),
        }
        for qubit, pauli in sorted(term_tuple, key=lambda item: int(item[0]))
    ]


def pauli_word_string(term_tuple):
    word = pauli_word_record(term_tuple)
    if not word:
        return "I"
    return " ".join(f"{item['pauli']}{item['qubit']}" for item in word)


def build_group_fermion_operator(group):
    outer_coeff = complex_from_record(group["outer_coefficient"])
    group_op = FermionOperator()
    fermion_terms = []

    for sq_term in group["inner_terms"]:
        ladder_ops = ladder_ops_from_sq_term(sq_term)
        coefficient = outer_coeff * complex_from_record(sq_term["coefficient"])
        group_op += FermionOperator(ladder_ops, coefficient)
        fermion_terms.append(
            {
                "source_term_index": int(sq_term["term_index"]),
                "coefficient": complex_record(coefficient),
                "ladder_ops": ladder_ops_record(ladder_ops),
                "is_identity": len(ladder_ops) == 0,
            }
        )

    group_op.compress(abs_tol=COEFFICIENT_TOLERANCE)
    return group_op, fermion_terms


def serialize_pauli_terms(qubit_operator):
    qubit_operator.compress(abs_tol=COEFFICIENT_TOLERANCE)
    records = []
    sorted_terms = sorted(
        qubit_operator.terms.items(),
        key=lambda item: (len(item[0]), pauli_word_string(item[0])),
    )
    for term_index, (term_tuple, coefficient) in enumerate(sorted_terms):
        records.append(
            {
                "term_index": int(term_index),
                "coefficient": complex_record(coefficient),
                "pauli_word": pauli_word_record(term_tuple),
                "pauli_word_string": pauli_word_string(term_tuple),
                "is_identity": len(term_tuple) == 0,
            }
        )
    return records


def convert_group(group, num_qubits):
    fermion_op, fermion_terms = build_group_fermion_operator(group)
    qubit_op = jordan_wigner(fermion_op)
    pauli_terms = serialize_pauli_terms(qubit_op)
    identity_count = sum(1 for term in pauli_terms if term["is_identity"])

    return {
        "group_index": int(group["group_index"]),
        "source_group_index": int(group["group_index"]),
        "source_classification": group["classification"],
        "source_num_inner_terms": int(group["num_inner_terms"]),
        "source_inner_terms": group["inner_terms"],
        "openfermion_fermion_terms": fermion_terms,
        "num_pauli_terms": len(pauli_terms),
        "num_identity_terms": identity_count,
        "num_non_identity_terms": len(pauli_terms) - identity_count,
        "qubit_count": int(num_qubits),
        "pauli_terms": pauli_terms,
    }


def main():
    if not INPUT_JSON.exists():
        raise FileNotFoundError(
            f"Missing Hermitian-pair archive: {INPUT_JSON}. "
            "Run scripts/qforte/build_h4_hermitian_pairs.py first."
        )

    with INPUT_JSON.open("r", encoding="utf-8") as handle:
        hp_archive = json.load(handle)

    num_qubits = int(hp_archive["active_space"]["num_qubits"])
    groups = [convert_group(group, num_qubits) for group in hp_archive["groups"]]

    total_pauli_terms = sum(group["num_pauli_terms"] for group in groups)
    identity_terms = sum(group["num_identity_terms"] for group in groups)
    non_identity_terms = total_pauli_terms - identity_terms
    nonempty_groups = sum(1 for group in groups if group["num_pauli_terms"] > 0)

    archive = {
        "schema_version": 1,
        "generated_at_utc": now_utc(),
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "input_hermitian_pair_archive": str(INPUT_JSON),
        "openfermion": {
            "fermion_operator": "openfermion.FermionOperator",
            "transform": "openfermion.transforms.jordan_wigner",
            "trust_note": (
                "This bridge step trusts OpenFermion's Jordan-Wigner transform "
                "and does not validate Pauli terms against qforte."
            ),
            "coefficient_tolerance": COEFFICIENT_TOLERANCE,
        },
        "indexing": {
            "orbital_to_qubit_order": "spin orbital p maps to qubit p",
            "openfermion_ladder_action": "1 means creation, 0 means annihilation",
            "pauli_word_convention": (
                "Pauli words are lists of {qubit, pauli}; an empty list and "
                "pauli_word_string='I' denote the identity."
            ),
            "endianness_note": hp_archive["hf_reference"].get("endianness_note"),
        },
        "molecule": hp_archive["molecule"],
        "active_space": hp_archive["active_space"],
        "hf_reference": hp_archive["hf_reference"],
        "diagnostics": {
            "source_hermitian_pair_groups": len(hp_archive["groups"]),
            "grouped_pauli_groups": len(groups),
            "nonempty_pauli_groups": nonempty_groups,
            "total_pauli_terms": total_pauli_terms,
            "identity_pauli_terms": identity_terms,
            "non_identity_pauli_terms": non_identity_terms,
            "max_pauli_terms_in_group": max(
                (group["num_pauli_terms"] for group in groups),
                default=0,
            ),
        },
        "groups": groups,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(archive, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Wrote grouped Pauli archive: {OUTPUT_JSON}")
    print(f"Hermitian-pair groups:       {len(hp_archive['groups'])}")
    print(f"Grouped Pauli groups:        {len(groups)}")
    print(f"Total Pauli terms:           {total_pauli_terms}")
    print(f"Identity Pauli terms:        {identity_terms}")
    print(f"Non-identity Pauli terms:    {non_identity_terms}")


if __name__ == "__main__":
    main()
