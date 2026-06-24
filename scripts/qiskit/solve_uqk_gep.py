# Manual run:
#   conda activate qiskit_env_v1
#   python scripts/qiskit/solve_uqk_gep.py
#
# Summary:
#   Post-process a saved UQK correlation sequence into the projected unitary
#   generalized eigenvalue problem
#
#       U c = lambda S c,
#
#   where S_mn = C_(n-m) and U_mn = C_(n+1-m). The script solves this GEV with
#   qforte-style canonical orthogonalization of S, converts eigenphases to
#   energies E = -Arg(lambda)/dt, and saves a JSON summary for inspection.
#
# Hard-coded options:
#   INPUT_CORRELATION_NPZ = results/krylov/h4_standard_uqk_overlap_matrix.npz
#   INPUT_CORRELATION_METADATA_JSON =
#       results/krylov/h4_standard_uqk_overlap_matrix_metadata.json
#   INPUT_MOLECULE_METADATA_JSON = data/molecules/h4_linear_sto3g_metadata.json
#   KRYLOV_DIMENSION_TO_USE = 3
#   KRYLOV_DT = 0.1
#   OVERLAP_EIGENVALUE_THRESHOLD = None, meaning qforte default 1e-15
#   SORT_ROOTS_BY = "energy_real_ascending"
#   USE_DIRECT_VALIDATION_WHEN_AVAILABLE = True
#   OUTPUT_SUMMARY_JSON =
#       results/summaries/h4_standard_uqk_projected_unitary_summary.json

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

INPUT_CORRELATION_NPZ = (
    REPO_ROOT / "results" / "krylov" / "h4_standard_uqk_overlap_matrix.npz"
)
INPUT_CORRELATION_METADATA_JSON = (
    REPO_ROOT
    / "results"
    / "krylov"
    / "h4_standard_uqk_overlap_matrix_metadata.json"
)
INPUT_MOLECULE_METADATA_JSON = (
    REPO_ROOT / "data" / "molecules" / "h4_linear_sto3g_metadata.json"
)
INPUT_DIRECT_VALIDATION_NPZ = (
    REPO_ROOT
    / "results"
    / "validation"
    / "h4_standard_uqk_statevector_comparison.npz"
)
OUTPUT_SUMMARY_JSON = (
    REPO_ROOT
    / "results"
    / "summaries"
    / "h4_standard_uqk_projected_unitary_summary.json"
)

# KRYLOV_DIMENSION_TO_USE is M in the notes. The basis states are
# |phi_n> = U^n |HF>, n=0,...,M-1, giving an M x M projected problem.
KRYLOV_DIMENSION_TO_USE = 3

# KRYLOV_DT is the UQK time step used to interpret eigenphases:
#   lambda_j ~= exp(-i E_j dt), so E_j = -Arg(lambda_j)/dt.
# For standard mode, dt is sealed into the saved QPY circuits upstream; this
# script treats KRYLOV_DT as a guard and checks it against the saved NPZ.
KRYLOV_DT = 0.1

# qforte canonical_geig_solve uses 1e-15 when no stabilization threshold is
# provided. Keep None to mirror that default. Set a larger value such as 1e-10
# only when intentionally reducing the retained overlap rank.
OVERLAP_EIGENVALUE_THRESHOLD = None

# Valid option currently implemented:
#   "energy_real_ascending"
#       Sort by the real phase-derived energy E=-Arg(lambda)/dt from lowest to
#       highest. This is natural for comparing the lowest roots to HF/FCI.
SORT_ROOTS_BY = "energy_real_ascending"

# If the direct statevector validation NPZ contains C_direct through k=M, solve
# the same projected unitary problem from those direct correlations and print it
# next to the MFE/noisy result. Older validation files may only contain C_0
# through C_(M-1), which is enough for S but not enough for U.
USE_DIRECT_VALIDATION_WHEN_AVAILABLE = True


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


def complex_record(value):
    value = complex(value)
    return {"real": float(value.real), "imag": float(value.imag)}


def complex_vector_records(values):
    return [
        {"index": int(index), **complex_record(value)}
        for index, value in enumerate(values)
    ]


def complex_matrix_records(matrix):
    return [
        [complex_record(matrix[row, col]) for col in range(matrix.shape[1])]
        for row in range(matrix.shape[0])
    ]


def load_correlation_sequence(npz_path):
    data = np.load(npz_path)
    if "correlations" not in data.files:
        raise KeyError(f"{npz_path} does not contain a 'correlations' array.")
    correlations = np.asarray(data["correlations"], dtype=np.complex128)
    if "correlation_powers" in data.files:
        powers = np.asarray(data["correlation_powers"], dtype=int)
    else:
        powers = np.arange(len(correlations), dtype=int)
    if len(correlations) != len(powers):
        raise ValueError("correlations and correlation_powers have inconsistent lengths.")
    return {int(power): complex(value) for power, value in zip(powers, correlations)}


def correlation_at(correlation_by_power, power):
    if power >= 0:
        if power not in correlation_by_power:
            raise KeyError(f"Missing C_{power}; projected U needs correlations through k=M.")
        return correlation_by_power[power]
    positive_power = -power
    if positive_power not in correlation_by_power:
        raise KeyError(
            f"Missing C_{positive_power}; negative powers use C_-k=conj(C_k)."
        )
    return np.conjugate(correlation_by_power[positive_power])


def assemble_s_and_u(correlation_by_power, dimension):
    s_matrix = np.empty((dimension, dimension), dtype=np.complex128)
    u_matrix = np.empty((dimension, dimension), dtype=np.complex128)
    for m in range(dimension):
        for n in range(dimension):
            s_matrix[m, n] = correlation_at(correlation_by_power, n - m)
            u_matrix[m, n] = correlation_at(correlation_by_power, n + 1 - m)
    return s_matrix, u_matrix


def effective_threshold():
    return 1.0e-15 if OVERLAP_EIGENVALUE_THRESHOLD is None else float(
        OVERLAP_EIGENVALUE_THRESHOLD
    )


def canonical_unitary_gep_solve(s_matrix, u_matrix):
    """Solve U c = lambda S c with qforte-style canonical orthogonalization."""

    threshold = effective_threshold()

    # S is explicitly Hermitian by Toeplitz construction. qforte's canonical
    # solver uses scipy.linalg.eig; using eigh here is the same canonical
    # orthogonalization idea but takes advantage of the Hermitian S structure.
    s_evals, s_evecs = np.linalg.eigh(s_matrix)
    retained = np.where(s_evals > threshold)[0]
    if len(retained) == 0:
        raise ValueError(
            f"No overlap eigenvalues survived threshold {threshold:.3e}."
        )

    x_prime = s_evecs[:, retained] / np.sqrt(s_evals[retained])[None, :]
    u_prime = x_prime.conj().T @ u_matrix @ x_prime
    lambdas, reduced_vectors = np.linalg.eig(u_prime)
    full_vectors = x_prime @ reduced_vectors

    # Normalize generalized eigenvectors so c^\dagger S c = 1 when possible.
    for col in range(full_vectors.shape[1]):
        norm = full_vectors[:, col].conj().T @ s_matrix @ full_vectors[:, col]
        if abs(norm) > 1.0e-14:
            full_vectors[:, col] /= np.sqrt(norm)

    phases = np.angle(lambdas)
    energies = -phases / KRYLOV_DT

    if SORT_ROOTS_BY != "energy_real_ascending":
        raise ValueError(
            "Only SORT_ROOTS_BY='energy_real_ascending' is currently implemented."
        )
    order = np.argsort(np.real(energies))

    lambdas = lambdas[order]
    phases = phases[order]
    energies = energies[order]
    full_vectors = full_vectors[:, order]
    reduced_vectors = reduced_vectors[:, order]

    residuals = []
    for root, (lam, vec) in enumerate(zip(lambdas, full_vectors.T)):
        residual = u_matrix @ vec - lam * (s_matrix @ vec)
        denom = np.linalg.norm(u_matrix @ vec) + abs(lam) * np.linalg.norm(
            s_matrix @ vec
        )
        rel = np.linalg.norm(residual) / max(denom, 1.0e-15)
        s_norm = vec.conj().T @ s_matrix @ vec
        residuals.append(
            {
                "root": int(root),
                "absolute_residual_norm": float(np.linalg.norm(residual)),
                "relative_residual_norm": float(rel),
                "s_norm": complex_record(s_norm),
            }
        )

    return {
        "threshold": threshold,
        "overlap_eigenvalues": s_evals,
        "retained_overlap_indices": retained,
        "retained_rank": int(len(retained)),
        "X_prime": x_prime,
        "U_prime": u_prime,
        "lambda_values": lambdas,
        "phases": phases,
        "energies": energies,
        "eigenvectors": full_vectors,
        "residuals": residuals,
    }


def solve_from_correlations(correlation_by_power):
    s_matrix, u_matrix = assemble_s_and_u(
        correlation_by_power,
        KRYLOV_DIMENSION_TO_USE,
    )
    solution = canonical_unitary_gep_solve(s_matrix, u_matrix)
    solution["S"] = s_matrix
    solution["U"] = u_matrix
    return solution


def result_to_json_ready(result):
    return {
        "threshold": float(result["threshold"]),
        "overlap_eigenvalues": [
            float(value) for value in np.real(result["overlap_eigenvalues"])
        ],
        "retained_overlap_indices": [
            int(value) for value in result["retained_overlap_indices"]
        ],
        "retained_rank": int(result["retained_rank"]),
        "condition_number": float(np.linalg.cond(result["S"])),
        "lambda_values": complex_vector_records(result["lambda_values"]),
        "lambda_magnitudes": [
            float(abs(value)) for value in result["lambda_values"]
        ],
        "phases": [
            {"root": int(index), "arg_lambda": float(value)}
            for index, value in enumerate(result["phases"])
        ],
        "energies_hartree": [
            {"root": int(index), "energy": float(value)}
            for index, value in enumerate(result["energies"])
        ],
        "residuals": result["residuals"],
        "S": complex_matrix_records(result["S"]),
        "U": complex_matrix_records(result["U"]),
    }


def molecule_energy_summary(molecule_metadata):
    energies = molecule_metadata.get("energies_hartree", {})
    return {
        "hf": energies.get("hf"),
        "fci_roots": energies.get("fci_roots", []),
        "nuclear_repulsion": energies.get("nuclear_repulsion"),
    }


def maybe_solve_direct_validation():
    if not USE_DIRECT_VALIDATION_WHEN_AVAILABLE:
        return None, "disabled by hard-coded option"
    if not INPUT_DIRECT_VALIDATION_NPZ.exists():
        return None, f"not found: {INPUT_DIRECT_VALIDATION_NPZ}"

    data = np.load(INPUT_DIRECT_VALIDATION_NPZ)
    if "C_direct" not in data.files:
        return None, "C_direct array not present"
    direct_c = np.asarray(data["C_direct"], dtype=np.complex128)
    if len(direct_c) <= KRYLOV_DIMENSION_TO_USE:
        return (
            None,
            "C_direct does not include shifted C_M; rerun the direct validation "
            "script after updating it to save correlations through k=M",
        )

    correlation_by_power = {
        int(power): complex(value) for power, value in enumerate(direct_c)
    }
    return solve_from_correlations(correlation_by_power), None


def print_root_table(label, result, hf_energy, fci_roots):
    print_header(label)
    print(
        f"{'root':>4} {'lambda':>28} {'|lambda|':>12} "
        f"{'Arg(lambda)':>14} {'E=-Arg/dt':>16} {'FCI ref':>16}"
    )
    print("-" * 100)
    for root, lam in enumerate(result["lambda_values"]):
        fci = fci_roots[root] if root < len(fci_roots) else None
        fci_str = f"{fci:+.10f}" if fci is not None else "--"
        print(
            f"{root:>4} "
            f"{lam.real:+.9f}{lam.imag:+.9f}j "
            f"{abs(lam):>12.8f} "
            f"{result['phases'][root]:>+14.9f} "
            f"{result['energies'][root]:>+16.9f} "
            f"{fci_str:>16}"
        )
    if hf_energy is not None:
        print(f"\nHF energy: {hf_energy:+.12f} hartree")


def main():
    molecule_metadata = load_json(INPUT_MOLECULE_METADATA_JSON)
    correlation_metadata = load_json(INPUT_CORRELATION_METADATA_JSON)
    npz_data = np.load(INPUT_CORRELATION_NPZ)

    saved_m = int(npz_data["krylov_dimension"])
    saved_dt = float(npz_data["dt"])
    if KRYLOV_DIMENSION_TO_USE > saved_m:
        raise ValueError(
            f"KRYLOV_DIMENSION_TO_USE={KRYLOV_DIMENSION_TO_USE} exceeds saved "
            f"krylov_dimension={saved_m}."
        )
    if not np.isclose(saved_dt, KRYLOV_DT, atol=0.0, rtol=1.0e-12):
        raise ValueError(
            f"KRYLOV_DT={KRYLOV_DT} does not match saved NPZ dt={saved_dt}."
        )
    if not np.isclose(
        float(correlation_metadata["options"]["dt"]),
        KRYLOV_DT,
        atol=0.0,
        rtol=1.0e-12,
    ):
        raise ValueError("KRYLOV_DT does not match correlation metadata dt.")

    correlation_by_power = load_correlation_sequence(INPUT_CORRELATION_NPZ)
    if KRYLOV_DIMENSION_TO_USE not in correlation_by_power:
        raise ValueError(
            "The shifted projected U matrix requires C_M, but the input "
            f"correlation file does not contain C_{KRYLOV_DIMENSION_TO_USE}."
        )

    energy_summary = molecule_energy_summary(molecule_metadata)
    hf_energy = energy_summary["hf"]
    fci_roots = energy_summary["fci_roots"]

    print_header("Projected Unitary UQK GEV Solver")
    print(
        "This script reads a saved UQK correlation sequence C_k, assembles\n"
        "S_mn=C_(n-m) and U_mn=C_(n+1-m), solves U c=lambda S c by canonical\n"
        "orthogonalization of S, then converts eigenphases to energies."
    )
    print_header("Hard-Coded Options")
    print_kv("Input correlation NPZ:", INPUT_CORRELATION_NPZ)
    print_kv("Input correlation metadata:", INPUT_CORRELATION_METADATA_JSON)
    print_kv("Input molecule metadata:", INPUT_MOLECULE_METADATA_JSON)
    print_kv("Direct validation NPZ:", INPUT_DIRECT_VALIDATION_NPZ)
    print_kv("Krylov dimension M:", KRYLOV_DIMENSION_TO_USE)
    print_kv("dt:", KRYLOV_DT)
    print_kv("Threshold option:", OVERLAP_EIGENVALUE_THRESHOLD)
    print_kv("Effective threshold:", f"{effective_threshold():.3e}")
    print_kv("Sort roots by:", SORT_ROOTS_BY)
    print_kv("Output summary JSON:", OUTPUT_SUMMARY_JSON)

    print_header("Loaded Molecular References")
    print_kv("Molecule:", molecule_metadata["molecule"]["label"])
    print_kv("Basis:", molecule_metadata["molecule"]["basis"])
    print_kv("HF energy:", hf_energy)
    print_kv("FCI roots available:", len(fci_roots))
    if fci_roots:
        print_kv("FCI root 0:", f"{fci_roots[0]:+.12f}")

    result = solve_from_correlations(correlation_by_power)
    direct_result, direct_note = maybe_solve_direct_validation()

    print_header("Projected Matrices")
    print("S:")
    print(result["S"])
    print("\nU:")
    print(result["U"])
    print_kv("\nS Hermiticity error:", f"{np.linalg.norm(result['S'] - result['S'].conj().T):.12e}")
    print_kv("S condition number:", f"{np.linalg.cond(result['S']):.12e}")
    print_kv("Retained rank:", result["retained_rank"])
    print_kv("Energy modulo 2*pi/dt:", f"{2.0 * np.pi / KRYLOV_DT:.12f}")

    print_root_table(
        "Roots From Input Correlations",
        result,
        hf_energy,
        fci_roots,
    )

    if direct_result is not None:
        print_root_table(
            "Verification Roots From Direct Statevector Correlations",
            direct_result,
            hf_energy,
            fci_roots,
        )
    else:
        print_header("Direct Verification")
        print(f"Direct verification not used: {direct_note}")

    summary = {
        "schema_version": 1,
        "generated_at_utc": now_utc(),
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "inputs": {
            "correlation_npz": str(INPUT_CORRELATION_NPZ),
            "correlation_metadata_json": str(INPUT_CORRELATION_METADATA_JSON),
            "molecule_metadata_json": str(INPUT_MOLECULE_METADATA_JSON),
            "direct_validation_npz": str(INPUT_DIRECT_VALIDATION_NPZ),
        },
        "outputs": {
            "summary_json": str(OUTPUT_SUMMARY_JSON),
        },
        "options": {
            "krylov_dimension_to_use": KRYLOV_DIMENSION_TO_USE,
            "dt": KRYLOV_DT,
            "overlap_eigenvalue_threshold_option": OVERLAP_EIGENVALUE_THRESHOLD,
            "effective_overlap_eigenvalue_threshold": effective_threshold(),
            "sort_roots_by": SORT_ROOTS_BY,
            "use_direct_validation_when_available": USE_DIRECT_VALIDATION_WHEN_AVAILABLE,
        },
        "conventions": {
            "S": "S_mn = C_(n-m), with C_-k=conj(C_k)",
            "U": "U_mn = C_(n+1-m)",
            "phase_to_energy": "E = -Arg(lambda)/dt",
            "energy_modulo_ambiguity": "2*pi/dt",
            "canonical_orthogonalization": (
                "Diagonalize Hermitian S, discard eigenvalues below the qforte "
                "threshold, form X'=U_s s^(-1/2), solve X'^dagger U X', and "
                "back-transform eigenvectors."
            ),
        },
        "molecule": molecule_metadata["molecule"],
        "active_space": molecule_metadata["active_space"],
        "hf_reference": molecule_metadata["hf_reference"],
        "reference_energies_hartree": energy_summary,
        "correlation_metadata_options": correlation_metadata["options"],
        "correlation_values_used": [
            {"power": int(power), **complex_record(correlation_by_power[power])}
            for power in range(KRYLOV_DIMENSION_TO_USE + 1)
        ],
        "input_correlation_solution": result_to_json_ready(result),
        "direct_validation": {
            "used": direct_result is not None,
            "note": direct_note,
            "solution": (
                result_to_json_ready(direct_result)
                if direct_result is not None
                else None
            ),
        },
        "energy_modulo_2pi_over_dt": float(2.0 * np.pi / KRYLOV_DT),
    }

    OUTPUT_SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print_header("Saved Output")
    print_kv("Summary JSON:", OUTPUT_SUMMARY_JSON)


if __name__ == "__main__":
    main()
