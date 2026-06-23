# Manual run:
#   conda activate qfe_env_v1
#   python scripts/qforte/build_h4_hermitian_pairs.py
#
# Summary:
#   Rebuild the H4/STO-3G qforte molecule from metadata, construct the
#   Hermitian-pair Hamiltonian as a qforte.SQOpPool, and write an explicit JSON
#   archive of grouped second-quantized terms. This script does not serialize
#   qforte operator objects directly.
#
# Hard-coded options:
#   INPUT_METADATA_JSON = data/molecules/h4_linear_sto3g_metadata.json
#   OUTPUT_JSON = data/hamiltonians/h4_linear_sto3g_hermitian_pairs.json
#   QFORTE_SOURCE_ROOT = /Users/nstair/Src/my_qforte/qforte
#   OUTER_POOL_COEFFICIENT = 1.0

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_METADATA_JSON = REPO_ROOT / "data" / "molecules" / "h4_linear_sto3g_metadata.json"
OUTPUT_JSON = REPO_ROOT / "data" / "hamiltonians" / "h4_linear_sto3g_hermitian_pairs.json"
PSI4_OUTPUT_STEM = REPO_ROOT / "data" / "hamiltonians" / "h4_linear_sto3g_hp_psi4"

QFORTE_SOURCE_ROOT = Path("/Users/nstair/Src/my_qforte/qforte")
QFORTE_SOURCE_PATH = QFORTE_SOURCE_ROOT / "src"
OUTER_POOL_COEFFICIENT = 1.0


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


def metadata_geometry_to_qforte(metadata):
    geom = []
    for item in metadata["molecule"]["geometry"]:
        x, y, z = item["xyz"]
        geom.append((item["atom"], (float(x), float(y), float(z))))
    return geom


def build_molecule(qforte, metadata):
    molecule = metadata["molecule"]
    fci = metadata.get("fci", {})
    return qforte.system_factory(
        system_type="molecule",
        build_type="psi4",
        basis=molecule["basis"],
        mol_geometry=metadata_geometry_to_qforte(metadata),
        symmetry=molecule["symmetry"],
        multiplicity=int(molecule["multiplicity"]),
        charge=int(molecule["charge"]),
        nroots_fci=int(fci.get("num_roots_used_for_build", 1)),
        run_mp2=False,
        run_cisd=False,
        run_ccsd=False,
        run_fci=False,
        build_qb_ham=False,
        store_mo_ints=True,
        filename=str(PSI4_OUTPUT_STEM),
    )


def serialize_sq_term(term_index, term):
    coeff, creators, annihilators = term
    creators = [int(idx) for idx in creators]
    annihilators = [int(idx) for idx in annihilators]
    return {
        "term_index": int(term_index),
        "coefficient": complex_record(coeff),
        "creators": creators,
        "annihilators": annihilators,
        "rank": int(len(creators)),
        "is_zero_body": len(creators) == 0 and len(annihilators) == 0,
    }


def group_classification(inner_terms):
    if all(term["is_zero_body"] for term in inner_terms):
        return "zero_body_scalar"
    return "operator"


def serialize_group(group_index, outer_coeff, sq_operator):
    inner_terms = [
        serialize_sq_term(term_index, term)
        for term_index, term in enumerate(sq_operator.terms())
    ]
    outer = complex_record(outer_coeff)
    combined_zero_body = 0.0 + 0.0j
    for term in inner_terms:
        if term["is_zero_body"]:
            combined_zero_body += complex_from_record(outer) * complex_from_record(term["coefficient"])

    return {
        "group_index": int(group_index),
        "outer_coefficient": outer,
        "classification": group_classification(inner_terms),
        "num_inner_terms": len(inner_terms),
        "max_rank": max((term["rank"] for term in inner_terms), default=0),
        "inner_terms": inner_terms,
        "zero_body_contribution": complex_record(combined_zero_body),
        "reconstruction_hint": (
            "Create qforte.SQOperator, add each inner term with add_term(coeff, "
            "creators, annihilators), then add it to qforte.SQOpPool with "
            "pool.add_term(outer_coefficient, sq_operator)."
        ),
    }


def main():
    if not INPUT_METADATA_JSON.exists():
        raise FileNotFoundError(
            f"Missing molecule metadata: {INPUT_METADATA_JSON}. "
            "Run scripts/qforte/generate_h4_molecule_metadata.py first."
        )

    with INPUT_METADATA_JSON.open("r", encoding="utf-8") as handle:
        molecule_metadata = json.load(handle)

    qforte_source_path = Path(molecule_metadata.get("qforte_source_path", QFORTE_SOURCE_PATH))
    if str(qforte_source_path) not in sys.path:
        sys.path.insert(0, str(qforte_source_path))

    import qforte

    mol = build_molecule(qforte, molecule_metadata)

    hermitian_pairs = qforte.SQOpPool()
    hermitian_pairs.add_hermitian_pairs(OUTER_POOL_COEFFICIENT, mol.sq_hamiltonian)

    groups = [
        serialize_group(group_index, outer_coeff, sq_operator)
        for group_index, (outer_coeff, sq_operator) in enumerate(hermitian_pairs)
    ]

    scalar_group_indices = [
        group["group_index"]
        for group in groups
        if group["classification"] == "zero_body_scalar"
    ]
    nontrivial_group_indices = [
        group["group_index"]
        for group in groups
        if group["classification"] != "zero_body_scalar"
    ]
    scalar_energy = sum(
        complex_from_record(groups[idx]["zero_body_contribution"])
        for idx in scalar_group_indices
    )

    source_terms = list(mol.sq_hamiltonian.terms())
    diagnostics = {
        "source_sq_hamiltonian_terms": len(source_terms),
        "hermitian_pair_groups": len(groups),
        "scalar_groups": len(scalar_group_indices),
        "nontrivial_groups": len(nontrivial_group_indices),
        "total_inner_terms": sum(group["num_inner_terms"] for group in groups),
        "max_group_rank": max((group["max_rank"] for group in groups), default=0),
        "scalar_energy_from_zero_body_groups": complex_record(scalar_energy),
    }

    archive = {
        "schema_version": 1,
        "generated_at_utc": now_utc(),
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "input_molecule_metadata": str(INPUT_METADATA_JSON),
        "qforte_source_root": str(QFORTE_SOURCE_ROOT),
        "qforte_source_path": str(qforte_source_path),
        "qforte_import_file": getattr(qforte, "__file__", None),
        "qforte_version": getattr(qforte, "__version__", None),
        "construction": {
            "pool_class": "qforte.SQOpPool",
            "source_operator": "mol.sq_hamiltonian",
            "method": "hermitian_pairs.add_hermitian_pairs(1.0, mol.sq_hamiltonian)",
            "outer_pool_coefficient": complex_record(OUTER_POOL_COEFFICIENT),
            "qforte_grouping_note": (
                "qforte.SQOpPool.add_hermitian_pairs groups each second-quantized "
                "monomial with its Hermitian adjoint after sorting creation and "
                "annihilation index lists."
            ),
        },
        "molecule": molecule_metadata["molecule"],
        "active_space": molecule_metadata["active_space"],
        "hf_reference": molecule_metadata["hf_reference"],
        "separation": {
            "scalar_group_indices": scalar_group_indices,
            "nontrivial_group_indices": nontrivial_group_indices,
        },
        "diagnostics": diagnostics,
        "groups": groups,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(archive, handle, indent=2, sort_keys=True)
        handle.write("\n")

    scalar_real = diagnostics["scalar_energy_from_zero_body_groups"]["real"]
    scalar_imag = diagnostics["scalar_energy_from_zero_body_groups"]["imag"]
    print(f"Wrote Hermitian-pair archive: {OUTPUT_JSON}")
    print(f"Source SQ Hamiltonian terms: {diagnostics['source_sq_hamiltonian_terms']}")
    print(f"Hermitian-pair groups:       {diagnostics['hermitian_pair_groups']}")
    print(f"Scalar groups:              {diagnostics['scalar_groups']}")
    print(f"Nontrivial groups:          {diagnostics['nontrivial_groups']}")
    print(f"Total inner terms:          {diagnostics['total_inner_terms']}")
    print(f"Scalar zero-body energy:    {scalar_real:.12f} + {scalar_imag:.12f}j")


if __name__ == "__main__":
    main()
