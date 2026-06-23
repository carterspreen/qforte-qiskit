# Manual run:
#   conda activate qfe_env_v1
#   python scripts/openfermion/verify_h4_grouped_paulis.py
#
# Summary:
#   Read the grouped OpenFermion/Jordan-Wigner Pauli archive and verify that
#   each group preserves source indices, has parseable Pauli terms, represents
#   identity terms explicitly, and contains nonzero term counts.
#
# Hard-coded options:
#   INPUT_JSON = data/hamiltonians/h4_linear_sto3g_grouped_paulis.json

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_JSON = REPO_ROOT / "data" / "hamiltonians" / "h4_linear_sto3g_grouped_paulis.json"
VALID_PAULIS = {"X", "Y", "Z"}

REQUIRED_TOP_LEVEL_KEYS = [
    "schema_version",
    "generated_at_utc",
    "openfermion",
    "indexing",
    "active_space",
    "diagnostics",
    "groups",
]

REQUIRED_GROUP_KEYS = [
    "group_index",
    "source_group_index",
    "source_classification",
    "source_num_inner_terms",
    "source_inner_terms",
    "openfermion_fermion_terms",
    "num_pauli_terms",
    "num_identity_terms",
    "num_non_identity_terms",
    "qubit_count",
    "pauli_terms",
]

REQUIRED_PAULI_TERM_KEYS = [
    "term_index",
    "coefficient",
    "pauli_word",
    "pauli_word_string",
    "is_identity",
]


def require_keys(mapping, keys, label):
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{label} is missing required keys: {missing}")


def complex_from_record(record):
    require_keys(record, ["real", "imag"], "complex coefficient")
    return complex(float(record["real"]), float(record["imag"]))


def expected_word_string(pauli_word):
    if not pauli_word:
        return "I"
    return " ".join(f"{item['pauli']}{item['qubit']}" for item in pauli_word)


def verify_pauli_word(pauli_word, qubit_count, label):
    if not isinstance(pauli_word, list):
        raise ValueError(f"{label} pauli_word is not a list.")
    seen_qubits = set()
    for item in pauli_word:
        require_keys(item, ["qubit", "pauli"], label)
        qubit = item["qubit"]
        pauli = item["pauli"]
        if not isinstance(qubit, int):
            raise ValueError(f"{label} qubit index is not an integer: {qubit!r}")
        if qubit < 0 or qubit >= qubit_count:
            raise ValueError(f"{label} qubit index {qubit} is out of range.")
        if pauli not in VALID_PAULIS:
            raise ValueError(f"{label} has invalid Pauli label: {pauli!r}")
        if qubit in seen_qubits:
            raise ValueError(f"{label} repeats qubit index {qubit}.")
        seen_qubits.add(qubit)


def main():
    if not INPUT_JSON.exists():
        raise FileNotFoundError(
            f"Missing grouped Pauli archive: {INPUT_JSON}. "
            "Run scripts/openfermion/convert_h4_hermitian_pairs_to_openfermion.py first."
        )

    with INPUT_JSON.open("r", encoding="utf-8") as handle:
        archive = json.load(handle)

    require_keys(archive, REQUIRED_TOP_LEVEL_KEYS, "archive")

    groups = archive["groups"]
    diagnostics = archive["diagnostics"]
    active_space = archive["active_space"]
    num_qubits = int(active_space["num_qubits"])

    if not groups:
        raise ValueError("Grouped Pauli archive contains no groups.")
    if diagnostics["grouped_pauli_groups"] != len(groups):
        raise ValueError("Grouped Pauli group count diagnostic mismatch.")
    if diagnostics["source_hermitian_pair_groups"] != len(groups):
        raise ValueError("Source group count was not preserved.")

    total_pauli_terms = 0
    identity_terms = 0
    non_identity_terms = 0
    seen_identity_term = False

    for expected_group_index, group in enumerate(groups):
        require_keys(group, REQUIRED_GROUP_KEYS, f"group {expected_group_index}")
        if group["group_index"] != expected_group_index:
            raise ValueError(f"Unexpected group_index at group {expected_group_index}.")
        if group["source_group_index"] != expected_group_index:
            raise ValueError(
                f"Source group index was not preserved for group {expected_group_index}."
            )
        if group["qubit_count"] != num_qubits:
            raise ValueError(f"Qubit count mismatch in group {expected_group_index}.")

        pauli_terms = group["pauli_terms"]
        if not pauli_terms:
            raise ValueError(f"Group {expected_group_index} has no Pauli terms.")
        if group["num_pauli_terms"] != len(pauli_terms):
            raise ValueError(f"Pauli term count mismatch in group {expected_group_index}.")
        if group["source_num_inner_terms"] != len(group["source_inner_terms"]):
            raise ValueError(f"Source term count mismatch in group {expected_group_index}.")

        group_identity_count = 0
        group_non_identity_count = 0

        for expected_term_index, term in enumerate(pauli_terms):
            require_keys(term, REQUIRED_PAULI_TERM_KEYS, "Pauli term")
            if term["term_index"] != expected_term_index:
                raise ValueError(
                    f"Unexpected Pauli term index in group {expected_group_index}."
                )

            coeff = complex_from_record(term["coefficient"])
            if coeff.real != coeff.real or coeff.imag != coeff.imag:
                raise ValueError("NaN coefficient found in Pauli term.")

            pauli_word = term["pauli_word"]
            verify_pauli_word(pauli_word, num_qubits, "Pauli term")
            if term["pauli_word_string"] != expected_word_string(pauli_word):
                raise ValueError(
                    f"Pauli word string mismatch in group {expected_group_index}."
                )

            is_identity = len(pauli_word) == 0
            if term["is_identity"] != is_identity:
                raise ValueError(
                    f"Identity flag mismatch in group {expected_group_index}."
                )
            if is_identity and term["pauli_word_string"] != "I":
                raise ValueError("Identity Pauli term is not represented as 'I'.")

            group_identity_count += int(is_identity)
            group_non_identity_count += int(not is_identity)
            seen_identity_term = seen_identity_term or is_identity

        if group["num_identity_terms"] != group_identity_count:
            raise ValueError(f"Identity count mismatch in group {expected_group_index}.")
        if group["num_non_identity_terms"] != group_non_identity_count:
            raise ValueError(
                f"Non-identity count mismatch in group {expected_group_index}."
            )

        total_pauli_terms += len(pauli_terms)
        identity_terms += group_identity_count
        non_identity_terms += group_non_identity_count

    if total_pauli_terms == 0:
        raise ValueError("No Pauli terms were found.")
    if not seen_identity_term:
        raise ValueError("No explicit identity/scalar Pauli term was found.")
    if diagnostics["total_pauli_terms"] != total_pauli_terms:
        raise ValueError("Total Pauli term diagnostic mismatch.")
    if diagnostics["identity_pauli_terms"] != identity_terms:
        raise ValueError("Identity Pauli term diagnostic mismatch.")
    if diagnostics["non_identity_pauli_terms"] != non_identity_terms:
        raise ValueError("Non-identity Pauli term diagnostic mismatch.")

    print(f"Verified grouped Pauli archive: {INPUT_JSON}")
    print(f"Groups:              {len(groups)}")
    print(f"Total Pauli terms:   {total_pauli_terms}")
    print(f"Identity terms:      {identity_terms}")
    print(f"Non-identity terms:  {non_identity_terms}")
    print(f"Qubits:              {num_qubits}")


if __name__ == "__main__":
    main()
