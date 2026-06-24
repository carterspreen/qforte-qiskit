"""Helpers for notebook 1: qforte molecule data to grouped JW Pauli JSON.

These functions are intentionally plain and chatty. They do the bookkeeping
that is annoying in a notebook cell, while leaving all research-facing options
in the notebook itself.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from traceback import format_exception_only


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

    print(f"{label:<38} {value}")


def complex_record(value):
    """Serialize a complex number as a little JSON object."""

    zval = complex(value)
    return {"real": float(zval.real), "imag": float(zval.imag)}


def complex_from_record(record):
    """Undo complex_record."""

    return complex(float(record["real"]), float(record["imag"]))


def clean_token(text):
    """Make a short lowercase filesystem-safe token."""

    token = re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip().lower())
    return re.sub(r"_+", "_", token).strip("_")


def geometry_formula(geometry):
    """Return a compact formula such as h4 from [('H', ...), ...]."""

    counts = Counter(atom for atom, _xyz in geometry)
    parts = []
    for atom in sorted(counts):
        count = counts[atom]
        parts.append(atom.lower() if count == 1 else f"{atom.lower()}{count}")
    return "".join(parts)


def geometry_shape_label(geometry, tolerance=1.0e-8):
    """Guess a friendly geometry-shape label for the molecule name."""

    coords = [tuple(float(v) for v in xyz) for _atom, xyz in geometry]
    if len(coords) <= 2:
        return "diatomic" if len(coords) == 2 else "atom"
    axes_with_motion = []
    for axis in range(3):
        values = [coord[axis] for coord in coords]
        if max(values) - min(values) > tolerance:
            axes_with_motion.append(axis)
    if len(axes_with_motion) == 1:
        return "linear"
    return "molecule"


def automatic_molecule_name(geometry, basis, charge=0, multiplicity=1, label=None):
    """Build the notebook-local molecule folder name from visible options.

    If label is supplied, it wins. Otherwise we combine a simple geometry-shape
    guess, the stoichiometric formula, basis, charge, and multiplicity. This
    keeps the notebook from hard-coding names like h4 while still producing
    stable, readable directory names.
    """

    if label:
        return clean_token(label)
    pieces = [
        geometry_shape_label(geometry),
        geometry_formula(geometry),
        clean_token(basis),
    ]
    if int(charge) != 0:
        pieces.append(f"q{int(charge):+d}".replace("+", "p").replace("-", "m"))
    if int(multiplicity) != 1:
        pieces.append(f"mult{int(multiplicity)}")
    return "_".join(piece for piece in pieces if piece)


def notebook_workflow_paths(notebooks_root, molecule_name):
    """Create and return the notebook-local directory layout."""

    root = Path(notebooks_root).resolve() / molecule_name
    paths = {
        "root": root,
        "metadata_dir": root / "metadata",
        "hamiltonian_blocks_dir": root / "hamiltonian_blocks",
        "circuits_dir": root / "circuits",
        "results_dir": root / "results",
        "summaries_dir": root / "summaries",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    paths["manifest_json"] = paths["metadata_dir"] / "workflow_manifest.json"
    paths["molecule_metadata_json"] = (
        paths["metadata_dir"] / f"{molecule_name}_metadata.json"
    )
    paths["hermitian_pairs_json"] = (
        paths["hamiltonian_blocks_dir"] / f"{molecule_name}_hermitian_pairs.json"
    )
    paths["fermion_blocks_json"] = (
        paths["hamiltonian_blocks_dir"] / f"{molecule_name}_fermion_blocks.json"
    )
    paths["grouped_paulis_json"] = (
        paths["hamiltonian_blocks_dir"] / f"{molecule_name}_grouped_paulis.json"
    )
    return paths


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
    """Merge updates into workflow_manifest.json.

    The manifest is the handoff contract between the qforte and Qiskit
    notebooks. Each cell adds the files it produced so later cells can discover
    them without memorizing paths.
    """

    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        manifest = load_json(manifest_path)
    else:
        manifest = {"schema_version": 1, "created_at_utc": now_utc()}
    manifest["updated_at_utc"] = now_utc()
    manifest.update(updates)
    save_json(manifest_path, manifest)
    return manifest


def add_qforte_to_path(qforte_source_root):
    """Put the local qforte source checkout on sys.path and import qforte."""

    source_root = Path(qforte_source_root).expanduser().resolve()
    source_path = source_root / "src"
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))
    import qforte

    return qforte, source_root, source_path


def json_float(value):
    """Convert possible numpy-ish numbers to plain JSON floats."""

    if value is None:
        return None
    return float(value)


def json_int(value):
    """Convert possible numpy-ish numbers to plain JSON ints."""

    if value is None:
        return None
    return int(value)


def metadata_geometry_to_qforte(metadata):
    """Convert saved molecule metadata geometry back to qforte's tuple form."""

    geom = []
    for item in metadata["molecule"]["geometry"]:
        x, y, z = item["xyz"]
        geom.append((item["atom"], (float(x), float(y), float(z))))
    return geom


def build_qforte_molecule_and_metadata(
    *,
    notebooks_root,
    qforte_source_root,
    geometry,
    basis,
    charge,
    multiplicity,
    symmetry,
    requested_fci_roots,
    molecule_label=None,
    geometry_units="angstrom",
    run_fci=True,
):
    """Build the qforte molecule and write the first metadata/manifest files.

    This is Cell 1's workhorse. It calls qforte.system_factory, records HF/FCI
    references and bitstring conventions, creates notebooks/<molecule_name>/...,
    and returns the live qforte molecule object for the next cell.
    """

    qforte, source_root, source_path = add_qforte_to_path(qforte_source_root)
    molecule_name = automatic_molecule_name(
        geometry,
        basis,
        charge=charge,
        multiplicity=multiplicity,
        label=molecule_label,
    )
    paths = notebook_workflow_paths(notebooks_root, molecule_name)
    psi4_output_stem = paths["metadata_dir"] / f"{molecule_name}_psi4"

    fallback_reason = None
    nroots_used = int(requested_fci_roots)
    try:
        mol = qforte.system_factory(
            system_type="molecule",
            build_type="psi4",
            basis=basis,
            mol_geometry=geometry,
            symmetry=symmetry,
            multiplicity=int(multiplicity),
            charge=int(charge),
            nroots_fci=nroots_used,
            run_mp2=False,
            run_cisd=False,
            run_ccsd=False,
            run_fci=bool(run_fci),
            build_qb_ham=False,
            store_mo_ints=True,
            filename=str(psi4_output_stem),
        )
    except Exception as exc:
        fallback_reason = "".join(format_exception_only(type(exc), exc)).strip()
        print("Requested FCI build failed; falling back to one FCI root.")
        print(fallback_reason)
        nroots_used = 1
        mol = qforte.system_factory(
            system_type="molecule",
            build_type="psi4",
            basis=basis,
            mol_geometry=geometry,
            symmetry=symmetry,
            multiplicity=int(multiplicity),
            charge=int(charge),
            nroots_fci=nroots_used,
            run_mp2=False,
            run_cisd=False,
            run_ccsd=False,
            run_fci=bool(run_fci),
            build_qb_ham=False,
            store_mo_ints=True,
            filename=str(psi4_output_stem),
        )

    hf_reference = [int(bit) for bit in mol.hf_reference]
    occupied_spin_orbitals = [idx for idx, bit in enumerate(hf_reference) if bit]
    num_qubits = len(hf_reference)
    fci_roots = [float(value) for value in getattr(mol, "fci_energy_list", [])]
    status = (
        "all_requested_roots_returned"
        if len(fci_roots) == int(requested_fci_roots)
        else "fewer_roots_returned_than_requested"
    )
    if fallback_reason:
        status = "fallback_after_requested_roots_failed"

    metadata = {
        "schema_version": 1,
        "generated_at_utc": now_utc(),
        "qforte_source_root": str(source_root),
        "qforte_source_path": str(source_path),
        "qforte_import_file": getattr(qforte, "__file__", None),
        "qforte_version": getattr(qforte, "__version__", None),
        "backend": "psi4",
        "molecule": {
            "label": molecule_name,
            "geometry_units": geometry_units,
            "geometry": [
                {"atom": atom, "xyz": [float(x), float(y), float(z)]}
                for atom, (x, y, z) in geometry
            ],
            "basis": basis,
            "charge": int(charge),
            "multiplicity": int(multiplicity),
            "symmetry": symmetry,
            "point_group": list(getattr(mol, "point_group", [])),
        },
        "active_space": {
            "num_electrons": json_int(sum(hf_reference)),
            "num_spatial_orbitals": json_int(num_qubits // 2),
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
            "num_roots_requested": int(requested_fci_roots),
            "num_roots_used_for_build": int(nroots_used),
            "num_roots_returned": len(fci_roots),
            "status": status,
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
                "Qiskit count strings are printed as c[n-1]...c[0] when q[i] "
                "is measured into c[i]."
            ),
        },
        "files": {
            "metadata_json": str(paths["molecule_metadata_json"]),
            "psi4_output": str(psi4_output_stem) + ".out",
        },
    }

    save_json(paths["molecule_metadata_json"], metadata)
    manifest = update_manifest(
        paths["manifest_json"],
        {
            "molecule_name": molecule_name,
            "workflow_root": str(paths["root"]),
            "directories": {key: str(value) for key, value in paths.items() if key.endswith("_dir")},
            "files": {
                "manifest_json": str(paths["manifest_json"]),
                "molecule_metadata_json": str(paths["molecule_metadata_json"]),
            },
            "molecule": metadata["molecule"],
            "active_space": metadata["active_space"],
            "energies_hartree": metadata["energies_hartree"],
            "hf_reference": metadata["hf_reference"],
        },
    )

    print_header("Cell 1 Output: qforte Molecule")
    print_kv("Molecule name:", molecule_name)
    print_kv("Workflow root:", paths["root"])
    print_kv("Metadata JSON:", paths["molecule_metadata_json"])
    print_kv("Manifest JSON:", paths["manifest_json"])
    print_kv("Qubits:", metadata["active_space"]["num_qubits"])
    print_kv("Electrons:", metadata["active_space"]["num_electrons"])
    print_kv("HF energy:", f"{metadata['energies_hartree']['hf']:+.12f}")
    if fci_roots:
        print_kv("FCI root 0:", f"{fci_roots[0]:+.12f}")
    print_kv("FCI status:", status)

    return {
        "qforte": qforte,
        "molecule": mol,
        "molecule_name": molecule_name,
        "paths": paths,
        "metadata": metadata,
        "manifest": manifest,
    }


def serialize_sq_term(term_index, term):
    """Serialize one qforte second-quantized term."""

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


def serialize_hp_group(group_index, outer_coeff, sq_operator):
    """Serialize one qforte SQOpPool Hermitian-pair group."""

    inner_terms = [
        serialize_sq_term(term_index, term)
        for term_index, term in enumerate(sq_operator.terms())
    ]
    outer = complex_record(outer_coeff)
    combined_zero_body = 0.0 + 0.0j
    for term in inner_terms:
        if term["is_zero_body"]:
            combined_zero_body += complex_from_record(outer) * complex_from_record(
                term["coefficient"]
            )
    classification = (
        "zero_body_scalar"
        if all(term["is_zero_body"] for term in inner_terms)
        else "operator"
    )
    return {
        "group_index": int(group_index),
        "outer_coefficient": outer,
        "classification": classification,
        "num_inner_terms": len(inner_terms),
        "max_rank": max((term["rank"] for term in inner_terms), default=0),
        "inner_terms": inner_terms,
        "zero_body_contribution": complex_record(combined_zero_body),
    }


def build_and_save_hermitian_pairs(
    *,
    qforte,
    molecule,
    molecule_metadata,
    output_json,
    manifest_json,
    outer_pool_coefficient=1.0,
):
    """Build qforte Hermitian-pair groups and save them as readable JSON."""

    hermitian_pairs = qforte.SQOpPool()
    hermitian_pairs.add_hermitian_pairs(outer_pool_coefficient, molecule.sq_hamiltonian)
    groups = [
        serialize_hp_group(group_index, outer_coeff, sq_operator)
        for group_index, (outer_coeff, sq_operator) in enumerate(hermitian_pairs)
    ]
    scalar_group_indices = [
        group["group_index"] for group in groups if group["classification"] == "zero_body_scalar"
    ]
    scalar_energy = sum(
        complex_from_record(groups[idx]["zero_body_contribution"])
        for idx in scalar_group_indices
    )
    diagnostics = {
        "source_sq_hamiltonian_terms": len(list(molecule.sq_hamiltonian.terms())),
        "hermitian_pair_groups": len(groups),
        "scalar_groups": len(scalar_group_indices),
        "nontrivial_groups": sum(
            1 for group in groups if group["classification"] != "zero_body_scalar"
        ),
        "total_inner_terms": sum(group["num_inner_terms"] for group in groups),
        "max_group_rank": max((group["max_rank"] for group in groups), default=0),
        "scalar_energy_from_zero_body_groups": complex_record(scalar_energy),
    }
    archive = {
        "schema_version": 1,
        "generated_at_utc": now_utc(),
        "input_molecule_metadata": molecule_metadata["files"]["metadata_json"],
        "construction": {
            "pool_class": "qforte.SQOpPool",
            "source_operator": "mol.sq_hamiltonian",
            "method": "hermitian_pairs.add_hermitian_pairs",
            "outer_pool_coefficient": complex_record(outer_pool_coefficient),
        },
        "molecule": molecule_metadata["molecule"],
        "active_space": molecule_metadata["active_space"],
        "hf_reference": molecule_metadata["hf_reference"],
        "diagnostics": diagnostics,
        "groups": groups,
    }
    save_json(output_json, archive)
    update_manifest(
        manifest_json,
        {
            "files": {
                **load_json(manifest_json).get("files", {}),
                "hermitian_pairs_json": str(output_json),
            },
            "hermitian_pair_diagnostics": diagnostics,
        },
    )

    print_header("Cell 2 Output: Hermitian Pairs")
    print_kv("Hermitian-pair JSON:", output_json)
    print_kv("Groups:", diagnostics["hermitian_pair_groups"])
    print_kv("Scalar groups:", diagnostics["scalar_groups"])
    print_kv("Nontrivial groups:", diagnostics["nontrivial_groups"])
    scalar = diagnostics["scalar_energy_from_zero_body_groups"]
    print_kv("Scalar energy:", f"{scalar['real']:+.12f}{scalar['imag']:+.12f}j")
    return archive


def ladder_ops_from_sq_term(sq_term):
    """Convert serialized qforte creators/annihilators into OpenFermion tuples."""

    ladder_ops = []
    for idx in sq_term["creators"]:
        ladder_ops.append((int(idx), 1))
    for idx in sq_term["annihilators"]:
        ladder_ops.append((int(idx), 0))
    return tuple(ladder_ops)


def ladder_ops_record(ladder_ops):
    """Serialize OpenFermion ladder tuples."""

    return [
        {
            "mode": int(mode),
            "action": int(action),
            "action_label": "create" if int(action) == 1 else "annihilate",
        }
        for mode, action in ladder_ops
    ]


def build_and_save_openfermion_blocks(
    *,
    hermitian_pair_archive,
    output_json,
    manifest_json,
    coefficient_tolerance=1.0e-12,
):
    """Port each serialized qforte group into an OpenFermion FermionOperator.

    JSON cannot store the live FermionOperator objects, so this writes their
    term lists in OpenFermion's ladder-operator convention. The next cell reads
    these records back and applies OpenFermion's Jordan-Wigner transform.
    """

    from openfermion import FermionOperator

    groups = []
    total_terms = 0
    for group in hermitian_pair_archive["groups"]:
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
        group_op.compress(abs_tol=coefficient_tolerance)
        total_terms += len(group_op.terms)
        groups.append(
            {
                "group_index": int(group["group_index"]),
                "source_group_index": int(group["group_index"]),
                "source_classification": group["classification"],
                "source_num_inner_terms": int(group["num_inner_terms"]),
                "source_inner_terms": group["inner_terms"],
                "openfermion_fermion_terms": fermion_terms,
                "compressed_fermion_terms": [
                    {
                        "coefficient": complex_record(coeff),
                        "ladder_ops": ladder_ops_record(term),
                        "is_identity": len(term) == 0,
                    }
                    for term, coeff in sorted(group_op.terms.items(), key=lambda item: str(item[0]))
                ],
                "num_compressed_fermion_terms": len(group_op.terms),
            }
        )

    manifest_files = load_json(manifest_json).get("files", {})
    archive = {
        "schema_version": 1,
        "generated_at_utc": now_utc(),
        "input_hermitian_pair_archive": manifest_files.get("hermitian_pairs_json"),
        "openfermion": {
            "fermion_operator": "openfermion.FermionOperator",
            "coefficient_tolerance": coefficient_tolerance,
        },
        "molecule": hermitian_pair_archive["molecule"],
        "active_space": hermitian_pair_archive["active_space"],
        "hf_reference": hermitian_pair_archive["hf_reference"],
        "diagnostics": {
            "fermion_operator_groups": len(groups),
            "total_compressed_fermion_terms": total_terms,
        },
        "groups": groups,
    }
    save_json(output_json, archive)
    update_manifest(
        manifest_json,
        {
            "files": {
                **load_json(manifest_json).get("files", {}),
                "fermion_blocks_json": str(output_json),
            },
            "fermion_block_diagnostics": archive["diagnostics"],
        },
    )

    print_header("Cell 3 Output: OpenFermion Blocks")
    print_kv("Fermion block JSON:", output_json)
    print_kv("Groups:", len(groups))
    print_kv("Compressed terms:", total_terms)
    return archive


def pauli_word_record(term_tuple):
    """Serialize one OpenFermion qubit term tuple into qubit/Pauli records."""

    return [
        {"qubit": int(qubit), "pauli": str(pauli)}
        for qubit, pauli in sorted(term_tuple, key=lambda item: int(item[0]))
    ]


def pauli_word_string(term_tuple):
    """Render a Pauli word as something readable like X0 Z3."""

    word = pauli_word_record(term_tuple)
    if not word:
        return "I"
    return " ".join(f"{item['pauli']}{item['qubit']}" for item in word)


def fermion_operator_from_block(block, coefficient_tolerance):
    """Rehydrate a serialized OpenFermion block for Jordan-Wigner mapping."""

    from openfermion import FermionOperator

    group_op = FermionOperator()
    for term in block["openfermion_fermion_terms"]:
        ladder_ops = tuple(
            (int(item["mode"]), int(item["action"]))
            for item in term["ladder_ops"]
        )
        group_op += FermionOperator(ladder_ops, complex_from_record(term["coefficient"]))
    group_op.compress(abs_tol=coefficient_tolerance)
    return group_op


def serialize_pauli_terms(qubit_operator, coefficient_tolerance):
    """Serialize OpenFermion QubitOperator terms into the existing Pauli schema."""

    qubit_operator.compress(abs_tol=coefficient_tolerance)
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


def build_and_save_grouped_paulis(
    *,
    fermion_block_archive,
    output_json,
    manifest_json,
    coefficient_tolerance=1.0e-12,
):
    """Apply OpenFermion Jordan-Wigner blockwise and save Qiskit-ready JSON."""

    from openfermion.transforms import jordan_wigner

    num_qubits = int(fermion_block_archive["active_space"]["num_qubits"])
    groups = []
    for block in fermion_block_archive["groups"]:
        fermion_op = fermion_operator_from_block(block, coefficient_tolerance)
        qubit_op = jordan_wigner(fermion_op)
        pauli_terms = serialize_pauli_terms(qubit_op, coefficient_tolerance)
        identity_count = sum(1 for term in pauli_terms if term["is_identity"])
        groups.append(
            {
                "group_index": int(block["group_index"]),
                "source_group_index": int(block["source_group_index"]),
                "source_classification": block["source_classification"],
                "source_num_inner_terms": int(block["source_num_inner_terms"]),
                "source_inner_terms": block["source_inner_terms"],
                "openfermion_fermion_terms": block["openfermion_fermion_terms"],
                "num_pauli_terms": len(pauli_terms),
                "num_identity_terms": identity_count,
                "num_non_identity_terms": len(pauli_terms) - identity_count,
                "qubit_count": num_qubits,
                "pauli_terms": pauli_terms,
            }
        )

    total_pauli_terms = sum(group["num_pauli_terms"] for group in groups)
    identity_terms = sum(group["num_identity_terms"] for group in groups)
    manifest_files = load_json(manifest_json).get("files", {})
    diagnostics = {
        "source_hermitian_pair_groups": len(fermion_block_archive["groups"]),
        "source_fermion_operator_groups": len(fermion_block_archive["groups"]),
        "grouped_pauli_groups": len(groups),
        "nonempty_pauli_groups": sum(1 for group in groups if group["num_pauli_terms"] > 0),
        "total_pauli_terms": total_pauli_terms,
        "identity_pauli_terms": identity_terms,
        "non_identity_pauli_terms": total_pauli_terms - identity_terms,
        "max_pauli_terms_in_group": max(
            (group["num_pauli_terms"] for group in groups),
            default=0,
        ),
    }
    archive = {
        "schema_version": 1,
        "generated_at_utc": now_utc(),
        "input_fermion_block_archive": manifest_files.get("fermion_blocks_json"),
        "input_hermitian_pair_archive": manifest_files.get("hermitian_pairs_json"),
        "openfermion": {
            "fermion_operator": "openfermion.FermionOperator",
            "transform": "openfermion.transforms.jordan_wigner",
            "coefficient_tolerance": coefficient_tolerance,
            "trust_note": (
                "This notebook bridge trusts OpenFermion's Jordan-Wigner transform "
                "and does not validate Pauli terms against qforte."
            ),
        },
        "indexing": {
            "orbital_to_qubit_order": "spin orbital p maps to qubit p",
            "openfermion_ladder_action": "1 means creation, 0 means annihilation",
            "pauli_word_convention": (
                "Pauli words are lists of {qubit, pauli}; an empty list and "
                "pauli_word_string='I' denote the identity."
            ),
            "endianness_note": fermion_block_archive["hf_reference"].get(
                "endianness_note"
            ),
        },
        "molecule": fermion_block_archive["molecule"],
        "active_space": fermion_block_archive["active_space"],
        "hf_reference": fermion_block_archive["hf_reference"],
        "diagnostics": diagnostics,
        "groups": groups,
    }
    save_json(output_json, archive)
    update_manifest(
        manifest_json,
        {
            "files": {
                **load_json(manifest_json).get("files", {}),
                "grouped_paulis_json": str(output_json),
            },
            "grouped_pauli_diagnostics": diagnostics,
        },
    )

    print_header("Cell 4 Output: Jordan-Wigner Grouped Pauli Blocks")
    print_kv("Grouped Pauli JSON:", output_json)
    print_kv("Groups:", diagnostics["grouped_pauli_groups"])
    print_kv("Total Pauli terms:", diagnostics["total_pauli_terms"])
    print_kv("Identity Pauli terms:", diagnostics["identity_pauli_terms"])
    print_kv("Non-identity Pauli terms:", diagnostics["non_identity_pauli_terms"])
    print_kv("Manifest JSON:", manifest_json)
    return archive
