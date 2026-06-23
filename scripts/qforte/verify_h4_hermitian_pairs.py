# Manual run:
#   conda activate qfe_env_v1
#   python scripts/qforte/verify_h4_hermitian_pairs.py
#
# Summary:
#   Read the H4 Hermitian-pair JSON archive, verify the grouped term schema,
#   reconstruct a qforte.SQOpPool from the saved coefficients, and check that
#   complex coefficients and integer index lists round-trip cleanly.
#
# Hard-coded options:
#   INPUT_JSON = data/hamiltonians/h4_linear_sto3g_hermitian_pairs.json
#   QFORTE_SOURCE_ROOT = /Users/nstair/Src/my_qforte/qforte

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_JSON = REPO_ROOT / "data" / "hamiltonians" / "h4_linear_sto3g_hermitian_pairs.json"
QFORTE_SOURCE_ROOT = Path("/Users/nstair/Src/my_qforte/qforte")
QFORTE_SOURCE_PATH = QFORTE_SOURCE_ROOT / "src"

REQUIRED_TOP_LEVEL_KEYS = [
    "schema_version",
    "generated_at_utc",
    "construction",
    "molecule",
    "active_space",
    "separation",
    "diagnostics",
    "groups",
]

REQUIRED_GROUP_KEYS = [
    "group_index",
    "outer_coefficient",
    "classification",
    "num_inner_terms",
    "max_rank",
    "inner_terms",
    "zero_body_contribution",
]

REQUIRED_TERM_KEYS = [
    "term_index",
    "coefficient",
    "creators",
    "annihilators",
    "rank",
    "is_zero_body",
]


def require_keys(mapping, keys, label):
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{label} is missing required keys: {missing}")


def complex_from_record(record):
    require_keys(record, ["real", "imag"], "complex coefficient")
    return complex(float(record["real"]), float(record["imag"]))


def assert_int_list(values, label):
    if not isinstance(values, list):
        raise ValueError(f"{label} is not a list.")
    for value in values:
        if not isinstance(value, int):
            raise ValueError(f"{label} contains a non-integer value: {value!r}")


def main():
    if not INPUT_JSON.exists():
        raise FileNotFoundError(
            f"Missing Hermitian-pair archive: {INPUT_JSON}. "
            "Run scripts/qforte/build_h4_hermitian_pairs.py first."
        )

    with INPUT_JSON.open("r", encoding="utf-8") as handle:
        archive = json.load(handle)

    require_keys(archive, REQUIRED_TOP_LEVEL_KEYS, "archive")

    qforte_source_path = Path(archive.get("qforte_source_path", QFORTE_SOURCE_PATH))
    if str(qforte_source_path) not in sys.path:
        sys.path.insert(0, str(qforte_source_path))

    import qforte

    groups = archive["groups"]
    if not groups:
        raise ValueError("Hermitian-pair archive contains no groups.")

    scalar_indices = set(archive["separation"]["scalar_group_indices"])
    nontrivial_indices = set(archive["separation"]["nontrivial_group_indices"])
    seen_indices = set()

    reconstructed_pool = qforte.SQOpPool()
    total_inner_terms = 0
    scalar_energy = 0.0 + 0.0j

    for expected_index, group in enumerate(groups):
        require_keys(group, REQUIRED_GROUP_KEYS, f"group {expected_index}")
        if group["group_index"] != expected_index:
            raise ValueError(f"Unexpected group index at position {expected_index}.")
        seen_indices.add(group["group_index"])

        outer_coeff = complex_from_record(group["outer_coefficient"])
        inner_terms = group["inner_terms"]
        if group["num_inner_terms"] != len(inner_terms):
            raise ValueError(f"Group {expected_index} inner term count mismatch.")
        if not inner_terms:
            raise ValueError(f"Group {expected_index} has no inner terms.")

        sq_operator = qforte.SQOperator()
        all_zero_body = True
        max_rank = 0

        for expected_term_index, term in enumerate(inner_terms):
            require_keys(term, REQUIRED_TERM_KEYS, f"group {expected_index} term")
            if term["term_index"] != expected_term_index:
                raise ValueError(
                    f"Unexpected term index in group {expected_index}: "
                    f"{term['term_index']}"
                )

            creators = term["creators"]
            annihilators = term["annihilators"]
            assert_int_list(creators, "creators")
            assert_int_list(annihilators, "annihilators")
            if term["rank"] != len(creators):
                raise ValueError(f"Rank mismatch in group {expected_index}.")
            if len(creators) != len(annihilators):
                raise ValueError(
                    f"Creation/annihilation rank mismatch in group {expected_index}."
                )

            coeff = complex_from_record(term["coefficient"])
            roundtrip = {"real": coeff.real, "imag": coeff.imag}
            if complex_from_record(roundtrip) != coeff:
                raise ValueError("Complex coefficient failed round-trip conversion.")

            is_zero_body = len(creators) == 0 and len(annihilators) == 0
            if term["is_zero_body"] != is_zero_body:
                raise ValueError(f"Zero-body flag mismatch in group {expected_index}.")

            all_zero_body = all_zero_body and is_zero_body
            max_rank = max(max_rank, len(creators))
            total_inner_terms += 1
            if is_zero_body:
                scalar_energy += outer_coeff * coeff

            sq_operator.add_term(coeff, creators, annihilators)

        expected_classification = "zero_body_scalar" if all_zero_body else "operator"
        if group["classification"] != expected_classification:
            raise ValueError(f"Classification mismatch in group {expected_index}.")
        if group["max_rank"] != max_rank:
            raise ValueError(f"Max-rank mismatch in group {expected_index}.")

        reconstructed_pool.add_term(outer_coeff, sq_operator)

    if scalar_indices & nontrivial_indices:
        raise ValueError("Scalar and nontrivial group index sets overlap.")
    if scalar_indices | nontrivial_indices != seen_indices:
        raise ValueError("Separated group index sets do not cover all groups.")
    if len(reconstructed_pool) != len(groups):
        raise ValueError("Reconstructed qforte.SQOpPool has the wrong length.")
    if archive["diagnostics"]["total_inner_terms"] != total_inner_terms:
        raise ValueError("Total inner term diagnostic mismatch.")

    archived_scalar = complex_from_record(
        archive["diagnostics"]["scalar_energy_from_zero_body_groups"]
    )
    if abs(archived_scalar - scalar_energy) > 1.0e-12:
        raise ValueError("Scalar zero-body energy did not round-trip.")

    print(f"Verified Hermitian-pair archive: {INPUT_JSON}")
    print(f"Groups:             {len(groups)}")
    print(f"Scalar groups:      {len(scalar_indices)}")
    print(f"Nontrivial groups:  {len(nontrivial_indices)}")
    print(f"Total inner terms:  {total_inner_terms}")
    print(
        "Scalar energy:      "
        f"{archived_scalar.real:.12f} + {archived_scalar.imag:.12f}j"
    )


if __name__ == "__main__":
    main()
