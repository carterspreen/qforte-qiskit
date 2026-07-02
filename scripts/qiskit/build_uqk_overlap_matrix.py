# Manual run:
#   conda activate qiskit_env_v1
#   python scripts/qiskit/build_uqk_overlap_matrix.py
#
# Summary:
#   Build a unitary quantum Krylov overlap matrix S. In standard mode, each
#   Krylov time step uses the deterministic non-scalar Trotter block plus an
#   analytic scalar phase and finite-shot MFE. In exact_trotter mode, the same
#   Trotter circuits are evaluated with exact MFE return probabilities. In
#   stochastic mode, each correlation value is estimated from qDRIFT-sampled
#   grouped chunks plus the same analytic scalar phase. In exact_stochastic
#   mode, the qDRIFT sampled chunks are kept but each MFE experiment uses exact
#   return probabilities. All modes use the MFE templates to estimate
#   C_k = <HF|U^k|HF>, then assemble S_mn = C_(n-m).
#
# Hard-coded options:
#   INPUT_QPY = circuits/transpiled/h4_linear_sto3g_grouped_evolution.qpy
#   INPUT_CIRCUIT_METADATA_JSON = circuits/transpiled/h4_linear_sto3g_grouped_evolution_metadata.json
#   INPUT_MOLECULE_METADATA_JSON = data/molecules/h4_linear_sto3g_metadata.json
#   INPUT_HERMITIAN_PAIR_JSON = data/hamiltonians/h4_linear_sto3g_hermitian_pairs.json
#   INPUT_GROUPED_PAULI_JSON = data/hamiltonians/h4_linear_sto3g_grouped_paulis.json
#   UQK_MODE = standard
#   KRYLOV_DIMENSION = 3
#   MAX_CORRELATION_POWER = KRYLOV_DIMENSION
#   DT = read from circuit metadata
#   TROTTER_ORDER = read from circuit metadata, expected first order here
#   SHOTS_PER_MFE_EXPERIMENT = 200_000
#   BACKEND_MODE = local_noiseless_statevector, local_noisy_simple, or
#                  local_noisy_ibm_model
#   QDRIFT_SEGMENT_COUNT_ND = 50
#   STOCHASTIC_INSTANCES_PER_CORRELATION = 200
#   STOCHASTIC_WEIGHT_CONVENTION = group_pauli_l1_norm
#   RANDOM_SEED = 230623
#   output path is results/krylov/{prefix}_{output_label}_uqk_overlap_matrix.*
#
# Scalar convention:
#   The zero-body scalar energy is not included in qDRIFT lambda and is not
#   placed inside the MFE circuits. Identity Pauli terms that live inside
#   non-scalar normal-ordered groups are also not directly measurable as global
#   phases in the Qiskit circuits. The MFE vacuum-reference construction
#   cancels the corresponding reference-branch phase in the sampled counts or
#   exact return probabilities, so this script stores the raw MFE estimate as
#   the non-scalar physical correlation. The true zero-body scalar phase
#   exp(-i E_scalar k dt) is then applied analytically.

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, qpy, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import LieTrotter
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    depolarizing_error,
    thermal_relaxation_error,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from mfe_measurement_building_blocks import (  # noqa: E402
    F1_LABEL,
    F2_I_LABEL,
    F2_PLUS_LABEL,
    MFEFidelities,
    build_mfe_templates,
    estimate_z_from_counts,
    estimate_z_from_fidelities,
    occupied_qubits_from_occupation,
    validate_hf_metadata,
)


INPUT_QPY = (
    REPO_ROOT / "circuits" / "transpiled" / "h4_linear_sto3g_grouped_evolution.qpy"
)
INPUT_CIRCUIT_METADATA_JSON = (
    REPO_ROOT
    / "circuits"
    / "transpiled"
    / "h4_linear_sto3g_grouped_evolution_metadata.json"
)
INPUT_MOLECULE_METADATA_JSON = (
    REPO_ROOT / "data" / "molecules" / "h4_linear_sto3g_metadata.json"
)
INPUT_HERMITIAN_PAIR_JSON = (
    REPO_ROOT / "data" / "hamiltonians" / "h4_linear_sto3g_hermitian_pairs.json"
)
INPUT_GROUPED_PAULI_JSON = (
    REPO_ROOT / "data" / "hamiltonians" / "h4_linear_sto3g_grouped_paulis.json"
)
OUTPUT_DIR = REPO_ROOT / "results" / "krylov"

# UQK_MODE chooses the physical workflow branch.
#
#   "standard"
#       Deterministic UQK. For each k, build V_k = U(dt)^k by composing the
#       saved non-scalar first-order Trotter step k times. This is the most direct
#       implementation of C_k = <HF|U^k|HF>, but circuit depth grows roughly
#       linearly with k times the non-scalar Trotter-step depth.
#
#   "exact_trotter"
#       Deterministic UQK with the same full Trotter circuits as standard, but
#       the MFE return probabilities are computed exactly from statevectors
#       rather than estimated with finite measurement shots.
#
#   "stochastic"
#       Stochastic UQK. For each nonzero k, estimate C_k by averaging several
#       qDRIFT-sampled chunks for total time t = k*dt. Each chunk samples grouped
#       non-scalar indices rather than applying every grouped factor.
#
#   "exact_stochastic"
#       Stochastic UQK with the same qDRIFT-sampled chunks as stochastic mode,
#       but each sampled MFE circuit is evaluated with exact return
#       probabilities instead of finite measurement shots.
# UQK_MODE = "standard"
UQK_MODE = "stochastic"

# KRYLOV_DIMENSION is the matrix dimension M. The overlap matrix uses basis
# states |phi_n> = U^n |HF> for n = 0, ..., M-1, so S has shape (M, M).
#
# Reasonable first values:
#   2-4 for fast debugging.
#   5-10 once the simulator/runtime path is stable.
#   Larger M can make S ill-conditioned and requires more C_k estimates.
KRYLOV_DIMENSION = 3

# MAX_CORRELATION_POWER is the largest nonnegative k for C_k=<HF|U^k|HF>.
# To assemble S alone, it must be at least M-1. Keeping MAX_CORRELATION_POWER=M
# also gives C_M, which is useful later for the projected unitary matrix
# U_mn = C_(n+1-m).
MAX_CORRELATION_POWER = KRYLOV_DIMENSION

# SHOTS_PER_MFE_EXPERIMENT is used for each of the three finite-shot MFE
# circuits: F1, F2_plus, and F2_i. In stochastic mode this is per stochastic
# instance, so total circuit shots scale like
#   3 * SHOTS_PER_MFE_EXPERIMENT * STOCHASTIC_INSTANCES_PER_CORRELATION
# for each nonzero k.
# In exact_trotter and exact_stochastic modes this option is recorded for
# provenance but not used.
#
# Reasonable ranges:
#   100-1000 for smoke tests.
#   2000-10000 for less jumpy simulator studies.
#   Hardware runs should be chosen with queue time and budget in mind.
SHOTS_PER_MFE_EXPERIMENT = 2_000

# BACKEND_MODE selects where the MFE circuits are executed.
#
# Valid options:
#   "local_noiseless_statevector"
#       AerSimulator(method="statevector") with shot sampling. No credentials,
#       noise model, or hardware access.
#   "local_noisy_simple"
#       AerSimulator with a compact, hand-controlled depolarizing/readout noise
#       model. This is useful for reproducible sensitivity tests and does not
#       require IBM credentials.
#   "local_noisy_ibm_model"
#       AerSimulator with an IBM-style model. By default this uses calibration
#       information from a fake IBM backend and compresses the average error
#       rates to the active-space qubit count, so small molecule simulations do
#       not accidentally become 100+ qubit density-matrix jobs.
BACKEND_MODE = "local_noiseless_statevector"

# Output filenames historically used only UQK_MODE, for example
# h4_standard_uqk_overlap_matrix.npz. Keep that default for compatibility with
# downstream scripts. For sweeps, set OUTPUT_LABEL_OVERRIDE or
# OUTPUT_LABEL_SUFFIX so backend variants do not overwrite each other.
OUTPUT_FILE_STEM_PREFIX = "h4"
OUTPUT_LABEL_OVERRIDE = None
OUTPUT_LABEL_SUFFIX = ""

# Noisy simulator options shared by local_noisy_simple and local_noisy_ibm_model.
#
# For the current H4 active space, density_matrix is the most transparent noisy
# simulator method: it evolves a mixed state under the noise channels and then
# samples measurement counts. For larger active spaces, "automatic" or
# "statevector" may be more practical.
NOISY_SIMULATION_METHOD = "density_matrix"
TRANSPILE_MFE_CIRCUITS_FOR_NOISY_BACKEND = True
NOISY_TRANSPILE_OPTIMIZATION_LEVEL = 1

# local_noisy_simple parameters. The basis is IBM-like but intentionally small:
# RZ is virtual and receives no explicit error; X/SX receive one-qubit error;
# CX receives two-qubit error; measurement receives symmetric readout error.
SIMPLE_NOISE_BASIS_GATES = ["id", "rz", "sx", "x", "cx"]
SIMPLE_NOISE_ONE_QUBIT_DEPOLARIZING_PROBABILITY = 1.0e-3
SIMPLE_NOISE_TWO_QUBIT_DEPOLARIZING_PROBABILITY = 1.0e-2
SIMPLE_NOISE_READOUT_ERROR_PROBABILITY = 2.0e-2
SIMPLE_NOISE_INCLUDE_THERMAL_RELAXATION = True
SIMPLE_NOISE_T1_SECONDS = 100.0e-6
SIMPLE_NOISE_T2_SECONDS = 80.0e-6
SIMPLE_NOISE_ONE_QUBIT_GATE_TIME_SECONDS = 50.0e-9
SIMPLE_NOISE_TWO_QUBIT_GATE_TIME_SECONDS = 300.0e-9

# local_noisy_ibm_model parameters.
#
# "fake_backend" avoids credentials and is the default for local development.
# "runtime_backend" uses QiskitRuntimeService and requires the user's IBM
# account to be configured outside this repository.
IBM_MODEL_SOURCE = "fake_backend"
IBM_MODEL_FAKE_BACKEND_CLASS = "FakeBrisbane"
IBM_MODEL_RUNTIME_BACKEND_NAME = "ibm_brisbane"
IBM_MODEL_RUNTIME_INSTANCE = None

# Keep True for molecule-scale local simulation. False builds Aer directly from
# the full IBM backend, which can be useful for target checks but may be far too
# large for density-matrix simulation if transpilation widens the circuit to the
# complete device.
IBM_MODEL_COMPRESS_TO_ACTIVE_SPACE = True
IBM_MODEL_BASIS_GATES = ["id", "rz", "sx", "x", "ecr"]
IBM_MODEL_TWO_QUBIT_GATE = "ecr"

# QDRIFT_SEGMENT_COUNT_ND is N_d in the qDRIFT formula. Larger N_d generally
# gives a better stochastic approximation to exp(-i H t), but each sampled
# instance gets deeper because it contains more grouped factors:
#
#   sampled chunk = prod_{s=1}^{N_d} exp[-i (lambda*t/N_d) G_{mu_s}].
#
# The zero-body scalar term is excluded from lambda and applied analytically to
# C_k as exp(-i E_scalar*k*dt).
#
# Reasonable first values:
#   2-8 for quick debugging.
#   10-100 for serious stochastic convergence experiments.
QDRIFT_SEGMENT_COUNT_ND = 5

# STOCHASTIC_INSTANCES_PER_CORRELATION is the number of independent sampled
# qDRIFT chunks averaged for each C_k. Larger values reduce stochastic sampling
# noise but increase total circuit executions linearly.
#
# Reasonable first values:
#   1-5 for plumbing checks.
#   10-100 for convergence studies.
STOCHASTIC_INSTANCES_PER_CORRELATION = 2000

# STOCHASTIC_WEIGHT_CONVENTION defines the qDRIFT weights w_mu.
#
# Valid option currently implemented:
#   "group_pauli_l1_norm"
#       Define K_mu as the full saved grouped Pauli Hamiltonian contribution,
#       K_mu = sum_rho alpha_{mu rho} P_{mu rho}. Use
#       h_mu = w_mu = sum_rho |alpha_{mu rho}| and
#       G_mu = K_mu / h_mu. Then lambda = sum_mu w_mu and
#       p_mu = w_mu / lambda.
#
#       The sum over mu excludes the zero-body scalar group. Identity Pauli
#       terms that live inside a non-scalar grouped operator remain part of
#       K_mu because they are needed for the correct fermion-to-qubit action.
#
# Important convention note:
#   The grouped Pauli archive stores full K_mu coefficients. A stochastic
#   sampled factor is implemented as
#
#       exp[-i theta_k G_mu],
#       theta_k = lambda * (k*dt) / N_d.
#
#   The sign information is retained in G_mu because G_mu is the signed grouped
#   operator K_mu divided by the positive weight h_mu.
STOCHASTIC_WEIGHT_CONVENTION = "group_pauli_l1_norm"

# RANDOM_SEED controls stochastic group sampling and the Aer simulator shot
# sampler. Change it to generate an independent stochastic run while keeping all
# other hard-coded options fixed.
RANDOM_SEED = 230623

# ENFORCE_C0_EXACT stores C_0=1 in the final Toeplitz matrix. The script still
# measures the identity case and records that diagnostic separately because it
# is useful for seeing finite-shot MFE behavior.
ENFORCE_C0_EXACT = True

# MFE_VERBOSE_FOR_FIRST_NONZERO_POWER prints the detailed F1/F2_plus/F2_i
# template explanation for k=1. Keep it True while teaching/debugging; set it
# False for large sweeps.
MFE_VERBOSE_FOR_FIRST_NONZERO_POWER = True

# PRINT_CORRELATION_TABLE controls the final C_k summary printed before S.
PRINT_CORRELATION_TABLE = True

VALID_UQK_MODES = {
    "standard",
    "exact_trotter",
    "stochastic",
    "exact_stochastic",
}
EXACT_MFE_UQK_MODES = {"exact_trotter", "exact_stochastic"}
QDRIFT_UQK_MODES = {"stochastic", "exact_stochastic"}
VALID_BACKEND_MODES = {
    "local_noiseless_statevector",
    "local_noisy_simple",
    "local_noisy_ibm_model",
}


def now_utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def print_header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def print_kv(label, value):
    print(f"{label:<36} {value}")


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_qpy_circuits(path):
    with path.open("rb") as handle:
        return list(qpy.load(handle))


def operation_counts(circuit):
    return {name: int(count) for name, count in circuit.count_ops().items()}


def output_label(mode):
    if OUTPUT_LABEL_OVERRIDE:
        label = str(OUTPUT_LABEL_OVERRIDE)
    else:
        label = str(mode)
        if mode in QDRIFT_UQK_MODES:
            label = (
                f"{label}_Nd_{QDRIFT_SEGMENT_COUNT_ND}"
                f"_sipc_{STOCHASTIC_INSTANCES_PER_CORRELATION}"
            )
    if OUTPUT_LABEL_SUFFIX:
        label = f"{label}_{OUTPUT_LABEL_SUFFIX}"
    return label


def output_file_stem(mode):
    label = output_label(mode)
    if OUTPUT_FILE_STEM_PREFIX:
        return f"{OUTPUT_FILE_STEM_PREFIX}_{label}"
    return label


def output_npz_path(mode):
    return OUTPUT_DIR / f"{output_file_stem(mode)}_uqk_overlap_matrix.npz"


def output_metadata_path(mode):
    return OUTPUT_DIR / f"{output_file_stem(mode)}_uqk_overlap_matrix_metadata.json"


def complex_from_record(record):
    return complex(float(record["real"]), float(record["imag"]))


def validate_options(circuit_metadata):
    if UQK_MODE not in VALID_UQK_MODES:
        raise ValueError(
            f"UQK_MODE must be one of {sorted(VALID_UQK_MODES)}, not {UQK_MODE!r}."
        )
    if KRYLOV_DIMENSION < 1:
        raise ValueError("KRYLOV_DIMENSION must be at least 1.")
    if MAX_CORRELATION_POWER < KRYLOV_DIMENSION - 1:
        raise ValueError(
            "MAX_CORRELATION_POWER must be at least KRYLOV_DIMENSION - 1 "
            "to assemble S."
        )
    if UQK_MODE not in EXACT_MFE_UQK_MODES and SHOTS_PER_MFE_EXPERIMENT <= 0:
        raise ValueError("SHOTS_PER_MFE_EXPERIMENT must be positive.")
    if UQK_MODE not in EXACT_MFE_UQK_MODES and BACKEND_MODE not in VALID_BACKEND_MODES:
        raise ValueError(
            f"BACKEND_MODE must be one of {sorted(VALID_BACKEND_MODES)}, "
            f"not {BACKEND_MODE!r}."
        )
    if NOISY_TRANSPILE_OPTIMIZATION_LEVEL not in {0, 1, 2, 3}:
        raise ValueError("NOISY_TRANSPILE_OPTIMIZATION_LEVEL must be 0, 1, 2, or 3.")
    if SIMPLE_NOISE_ONE_QUBIT_DEPOLARIZING_PROBABILITY < 0.0:
        raise ValueError("Simple one-qubit depolarizing probability must be nonnegative.")
    if SIMPLE_NOISE_TWO_QUBIT_DEPOLARIZING_PROBABILITY < 0.0:
        raise ValueError("Simple two-qubit depolarizing probability must be nonnegative.")
    if not 0.0 <= SIMPLE_NOISE_READOUT_ERROR_PROBABILITY <= 0.5:
        raise ValueError("Simple readout error probability must be in [0, 0.5].")
    if IBM_MODEL_SOURCE not in {"fake_backend", "runtime_backend"}:
        raise ValueError("IBM_MODEL_SOURCE must be 'fake_backend' or 'runtime_backend'.")
    if QDRIFT_SEGMENT_COUNT_ND <= 0:
        raise ValueError("QDRIFT_SEGMENT_COUNT_ND must be positive.")
    if STOCHASTIC_INSTANCES_PER_CORRELATION <= 0:
        raise ValueError("STOCHASTIC_INSTANCES_PER_CORRELATION must be positive.")
    if (
        STOCHASTIC_WEIGHT_CONVENTION
        != "group_pauli_l1_norm"
    ):
        raise ValueError(
            "Only STOCHASTIC_WEIGHT_CONVENTION="
            "'group_pauli_l1_norm' is implemented."
        )

    encoded_order = int(circuit_metadata["options"]["trotter_sequence_order"])
    if encoded_order != 1:
        raise ValueError(
            "This deterministic UQK implementation expects first-order full-dt "
            f"group circuits. Found trotter_sequence_order={encoded_order}."
        )


def print_startup(circuit_metadata, molecule_metadata):
    print_header("UQK Overlap Matrix Builder")
    print(
        "This script has four deliberately separate branches:\n"
        "  standard:      deterministic UQK, saved non-scalar Trotter blocks,\n"
        "                 finite-shot MFE.\n"
        "  exact_trotter: deterministic UQK, saved non-scalar Trotter blocks,\n"
        "                 exact MFE return probabilities.\n"
        "  stochastic:    stochastic UQK, qDRIFT-sampled grouped blocks,\n"
        "                 finite-shot MFE.\n"
        "  exact_stochastic: stochastic UQK, qDRIFT-sampled grouped blocks,\n"
        "                    exact MFE return probabilities.\n"
        "All branches estimate C_k = <HF|U^k|HF> with the MFE circuits and\n"
        "assemble the Toeplitz overlap matrix S_mn = C_(n-m)."
    )

    print_header("Hard-Coded Options")
    print_kv("UQK mode:", UQK_MODE)
    print_kv("Valid UQK modes:", sorted(VALID_UQK_MODES))
    print_kv("Input QPY:", INPUT_QPY)
    print_kv("Input circuit metadata:", INPUT_CIRCUIT_METADATA_JSON)
    print_kv("Input molecule metadata:", INPUT_MOLECULE_METADATA_JSON)
    print_kv("Input Hermitian-pair JSON:", INPUT_HERMITIAN_PAIR_JSON)
    print_kv("Input grouped Pauli JSON:", INPUT_GROUPED_PAULI_JSON)
    print_kv("Krylov dimension M:", KRYLOV_DIMENSION)
    print_kv("Max correlation power:", MAX_CORRELATION_POWER)
    print_kv("dt from metadata:", circuit_metadata["options"]["dt"])
    print_kv("Trotter order from metadata:", circuit_metadata["options"]["trotter_sequence_order"])
    print_kv("Shots per MFE experiment:", SHOTS_PER_MFE_EXPERIMENT)
    print_kv("Backend mode:", BACKEND_MODE)
    print_kv("Valid backend modes:", sorted(VALID_BACKEND_MODES))
    print_kv("Noisy simulator method:", NOISY_SIMULATION_METHOD)
    print_kv(
        "Transpile noisy MFE circuits:",
        TRANSPILE_MFE_CIRCUITS_FOR_NOISY_BACKEND,
    )
    print_kv("Noisy transpile opt level:", NOISY_TRANSPILE_OPTIMIZATION_LEVEL)
    print_kv("Simple 1q depol prob:", SIMPLE_NOISE_ONE_QUBIT_DEPOLARIZING_PROBABILITY)
    print_kv("Simple 2q depol prob:", SIMPLE_NOISE_TWO_QUBIT_DEPOLARIZING_PROBABILITY)
    print_kv("Simple readout error prob:", SIMPLE_NOISE_READOUT_ERROR_PROBABILITY)
    print_kv("Simple thermal relaxation:", SIMPLE_NOISE_INCLUDE_THERMAL_RELAXATION)
    print_kv("IBM model source:", IBM_MODEL_SOURCE)
    print_kv("IBM fake backend class:", IBM_MODEL_FAKE_BACKEND_CLASS)
    print_kv("IBM runtime backend:", IBM_MODEL_RUNTIME_BACKEND_NAME)
    print_kv("IBM compact active-space model:", IBM_MODEL_COMPRESS_TO_ACTIVE_SPACE)
    print_kv("IBM model basis gates:", IBM_MODEL_BASIS_GATES)
    print_kv("qDRIFT segment count N_d:", QDRIFT_SEGMENT_COUNT_ND)
    print_kv("Stochastic instances per C_k:", STOCHASTIC_INSTANCES_PER_CORRELATION)
    print_kv("Stochastic weight convention:", STOCHASTIC_WEIGHT_CONVENTION)
    print_kv("Random seed:", RANDOM_SEED)
    print_kv("Enforce C0 exact:", ENFORCE_C0_EXACT)
    print_kv("Output label override:", OUTPUT_LABEL_OVERRIDE)
    print_kv("Output label suffix:", OUTPUT_LABEL_SUFFIX)
    print_kv("Output NPZ:", output_npz_path(UQK_MODE))
    print_kv("Output metadata JSON:", output_metadata_path(UQK_MODE))

    print_header("Loaded Molecule And Reference")
    print_kv("Molecule label:", molecule_metadata["molecule"]["label"])
    print_kv("Basis:", molecule_metadata["molecule"]["basis"])
    print_kv("Qubits:", molecule_metadata["active_space"]["num_qubits"])
    print_kv("Electrons:", molecule_metadata["active_space"]["num_electrons"])
    print_kv("HF occupation n_p:", molecule_metadata["hf_reference"]["occupation_little_endian"])
    print_kv(
        "Qiskit HF count key:",
        molecule_metadata["hf_reference"]["qiskit_counts_bitstring_if_measured_q_to_c_same_index"],
    )

    print_header("MFE And Toeplitz Convention")
    print(
        "For each k, V_k is the composed circuit U^k. The MFE templates estimate\n"
        "the HF branch relative to the vacuum reference branch using F1,\n"
        "F2_plus, and F2_i return probabilities or counts. The raw MFE estimate\n"
        "is kept as the\n"
        "non-scalar correlation. This is deliberate: identity Pauli terms inside\n"
        "normal-ordered non-scalar groups appear as unmeasurable circuit global\n"
        "phases, and the vacuum-reference MFE construction cancels the matching\n"
        "reference phase. The script only applies the separate zero-body scalar\n"
        "phase exp(-i E_scalar*k*dt) analytically.\n"
        "C_0 is physically exactly 1. This script still evaluates the identity\n"
        "case for diagnostics, then stores C_0 = 1 when ENFORCE_C0_EXACT=True.\n"
        "Negative powers are not measured: S_mn uses C_-k = conj(C_k)."
    )
    print_header("Mode-Specific Meaning")
    if UQK_MODE == "standard":
        print(
            "STANDARD UQK: every time step uses every non-scalar saved grouped "
            "circuit in the first-order Trotter sequence. The separate scalar "
            "phase is applied analytically to C_k, and MFE probabilities are "
            "estimated with finite shots."
        )
    elif UQK_MODE == "exact_trotter":
        print(
            "EXACT_TROTTER UQK: every time step uses the same full non-scalar "
            "Trotter circuit as standard mode. The MFE return probabilities "
            "are computed exactly from statevectors, so C_k has no finite-shot "
            "sampling noise."
        )
    elif UQK_MODE == "stochastic":
        print(
            "STOCHASTIC UQK: each C_k uses several independently sampled qDRIFT "
            "instances. A sampled instance chooses non-scalar grouped indices "
            "with p_mu=w_mu/lambda, rebuilds normalized grouped Pauli circuits "
            "at theta_k=lambda*k*dt/N_d, runs MFE, and averages the raw "
            "counts-derived MFE statistics. It does not apply a per-sample "
            "reference-branch correction."
        )
    else:
        print(
            "EXACT_STOCHASTIC UQK: each C_k uses the same independently "
            "sampled qDRIFT instances as stochastic mode, but each sampled "
            "MFE circuit is evaluated with exact Statevector return "
            "probabilities. The exact F1/F2 statistics are averaged across "
            "instances before applying the MFE formula."
        )


def print_user_option_guide():
    print_header("User-Facing Knob Guide")
    print(
        "Core equations:\n"
        "  C_k = <HF|U^k|HF>\n"
        "  S_mn = C_(n-m), with C_-k = conj(C_k)\n"
        "  stochastic total time T_k = k * dt\n"
    )
    print(
        "UQK_MODE:\n"
        "  standard      -> saved non-scalar Trotter step repeated k times, finite-shot MFE.\n"
        "  exact_trotter -> same Trotter step powers, exact statevector MFE probabilities.\n"
        "  stochastic    -> V_k is estimated by averaging qDRIFT-sampled chunks.\n"
        "  exact_stochastic -> same qDRIFT samples, exact statevector MFE probabilities.\n"
    )
    print(
        "KRYLOV_DIMENSION M:\n"
        "  Builds an M x M S matrix from |phi_n> = U^n|HF>, n=0..M-1.\n"
        "  Use M=2..4 for first checks; larger M needs more C_k values and can\n"
        "  make S ill-conditioned.\n"
    )
    print(
        "MAX_CORRELATION_POWER:\n"
        "  Must be at least M-1 for S. Setting it to M also prepares C_M for the\n"
        "  projected unitary matrix U_mn = C_(n+1-m) in a later workflow.\n"
    )
    print(
        "SHOTS_PER_MFE_EXPERIMENT:\n"
        "  Shots for each F1/F2_plus/F2_i circuit. Stochastic mode multiplies this\n"
        "  by STOCHASTIC_INSTANCES_PER_CORRELATION for each nonzero C_k.\n"
        "  exact_trotter and exact_stochastic ignore this knob because they\n"
        "  compute exact MFE probabilities instead of sampling counts.\n"
    )
    print(
        "QDRIFT_SEGMENT_COUNT_ND and STOCHASTIC_INSTANCES_PER_CORRELATION:\n"
        "  Larger N_d makes each sampled chunk deeper but closer to the qDRIFT\n"
        "  channel. More instances reduce stochastic sampling noise by averaging\n"
        "  independent chunks. Useful starter ranges are N_d=2..8 and instances=1..10;\n"
        "  convergence studies may need much larger values.\n"
    )
    print(
        "BACKEND_MODE:\n"
        "  local_noiseless_statevector -> Aer shot sampling without noise.\n"
        "  local_noisy_simple         -> compact hand-tuned depolarizing/readout noise.\n"
        "  local_noisy_ibm_model      -> IBM-style fake/runtime backend noise model.\n"
        "  exact_trotter and exact_stochastic ignore BACKEND_MODE and use local Statevector probabilities.\n"
    )
    print(
        "Noisy backend notes:\n"
        "  local_noisy_simple is deliberately reproducible and easy to sweep.\n"
        "  local_noisy_ibm_model defaults to a FakeBrisbane-derived compact model\n"
        "  on the active-space qubits, avoiding accidental full-device density\n"
        "  matrix simulation. Set IBM_MODEL_COMPRESS_TO_ACTIVE_SPACE=False only\n"
        "  when you intentionally want AerSimulator.from_backend(target_backend).\n"
    )


def print_qdrift_circuit_structure(sampling_model, dt):
    print_header("qDRIFT Circuit Structure")
    print(
        "The grouped Hamiltonian is treated as\n"
        "  H = E_scalar I + sum_mu K_mu\n"
        "  K_mu = sum_rho alpha_{mu,rho} P_{mu,rho}\n"
        "For stochastic modes this script defines\n"
        "  h_mu = w_mu = sum_rho |alpha_{mu,rho}|\n"
        "  G_mu = K_mu / h_mu\n"
        "  lambda = sum_mu w_mu,       p_mu = w_mu / lambda\n"
        "where the sum over mu excludes the zero-body scalar group."
    )
    print(
        "For total time T_k = k*dt, each stochastic sample builds\n"
        "  prod_{s=1}^{N_d} exp[-i theta_k G_{mu_s}],\n"
        "  theta_k = lambda*T_k/N_d.\n"
        "The scalar phase exp(-i E_scalar*T_k) is not part of the MFE circuit;\n"
        "it is multiplied into C_k analytically after MFE estimation."
    )
    print(
        "The deterministic Trotter modes still use the saved fixed-dt QPY group\n"
        "circuits, but skip the separate scalar group for the same MFE reason\n"
        "and apply the same analytic scalar phase to the final C_k."
    )
    first_entry = sampling_model["entries"][0]
    example_theta = sampling_model["weight_sum_lambda"] * dt / QDRIFT_SEGMENT_COUNT_ND
    print_kv("qDRIFT norm lambda:", f"{sampling_model['weight_sum_lambda']:.12f}")
    print_kv("Scalar excluded from lambda:", f"{sampling_model['scalar_energy']:.12f}")
    print_kv("Example k=1 total time T_1:", f"{dt:.12f}")
    print_kv("Example group index:", first_entry["group_index"])
    print_kv("Example h_mu=w_mu:", f"{first_entry['h_mu']:.12f}")
    print_kv("Example p_mu:", f"{first_entry['probability']:.12f}")
    print_kv("Example theta_k for k=1:", f"{example_theta:.12f}")


def build_one_trotter_step(group_circuits, circuit_metadata):
    num_qubits = int(circuit_metadata["active_space"]["num_qubits"])
    step = QuantumCircuit(num_qubits, name="non_scalar_first_order_trotter_step")
    sequence = circuit_metadata["trotter_step_sequence"]
    group_metadata = {
        int(group["group_index"]): group
        for group in circuit_metadata["groups"]
    }
    included_group_indices = []
    skipped_scalar_group_indices = []
    for item in sequence:
        if float(item["time_multiplier"]) != 1.0:
            raise ValueError(
                "This first UQK script expects full-dt group circuits with "
                f"time_multiplier=1.0. Found {item['time_multiplier']}."
            )
        group_index = int(item["group_index"])
        metadata = group_metadata[group_index]
        if metadata["source_classification"] == "zero_body_scalar":
            skipped_scalar_group_indices.append(group_index)
            continue
        qpy_index = int(metadata["qpy_circuit_index"])
        step.compose(group_circuits[qpy_index], inplace=True)
        included_group_indices.append(group_index)
    return step, included_group_indices, skipped_scalar_group_indices


def build_trotter_power(full_step, power):
    circuit = QuantumCircuit(full_step.num_qubits, name=f"uqk_V_power_{power}")
    for _ in range(power):
        circuit.compose(full_step, inplace=True)
    return circuit


def pauli_label(pauli_word, num_qubits):
    letters = ["I"] * num_qubits
    for item in pauli_word:
        letters[int(item["qubit"])] = item["pauli"]
    return "".join(reversed(letters))


def real_pauli_coefficient(term, group_index, term_index):
    coefficient = complex_from_record(term["coefficient"])
    if abs(coefficient.imag) > 1.0e-10:
        raise ValueError(
            f"Group {group_index} Pauli term {term_index} has imaginary "
            f"coefficient {coefficient.imag}, but real-time evolution expects "
            "real coefficients."
        )
    return float(coefficient.real)


def sparse_pauli_op_from_group(pauli_group, num_qubits, coefficient_scale=1.0):
    labels = []
    coefficients = []
    group_index = int(pauli_group["group_index"])
    for term in pauli_group["pauli_terms"]:
        term_index = int(term["term_index"])
        labels.append(pauli_label(term["pauli_word"], num_qubits))
        coefficients.append(
            float(coefficient_scale)
            * real_pauli_coefficient(term, group_index, term_index)
        )
    return SparsePauliOp(labels, coeffs=coefficients)


def build_group_evolution_circuit(
    pauli_group,
    num_qubits,
    evolution_time,
    basis_gates,
    coefficient_scale=1.0,
):
    """Build one grouped factor at an arbitrary time.

    The saved QPY group circuits are already fixed at dt, which is perfect for
    standard UQK. Stochastic qDRIFT needs exp[-i theta_k G_mu] with
    G_mu=K_mu/h_mu, so this helper rebuilds the selected grouped factor from the
    grouped Pauli archive while preserving the same group boundary.
    """

    group_index = int(pauli_group["group_index"])
    operator = sparse_pauli_op_from_group(
        pauli_group,
        num_qubits,
        coefficient_scale=coefficient_scale,
    )
    circuit = QuantumCircuit(num_qubits, name=f"qdrift_group_{group_index:04d}")
    gate = PauliEvolutionGate(
        operator,
        time=float(evolution_time),
        synthesis=LieTrotter(reps=1),
    )
    circuit.append(gate, range(num_qubits))
    return transpile(
        circuit,
        basis_gates=basis_gates,
        optimization_level=1,
    )


def group_pauli_l1_norm(pauli_group):
    """Return h_mu=w_mu=sum_rho |alpha_mu,rho| for one grouped Pauli K_mu."""

    return float(
        sum(
            abs(complex_from_record(term["coefficient"]))
            for term in pauli_group["pauli_terms"]
        )
    )


def scalar_energy_from_grouped_paulis(pauli_archive):
    total = 0.0
    for group in pauli_archive["groups"]:
        if group["source_classification"] != "zero_body_scalar":
            continue
        for term in group["pauli_terms"]:
            if term["is_identity"]:
                total += real_pauli_coefficient(
                    term,
                    int(group["group_index"]),
                    int(term["term_index"]),
                )
    return float(total)


def build_qdrift_sampling_model(hp_archive, pauli_archive):
    hp_by_index = {
        int(group["group_index"]): group
        for group in hp_archive["groups"]
    }
    pauli_by_index = {
        int(group["group_index"]): group
        for group in pauli_archive["groups"]
    }

    entries = []
    for group_index, pauli_group in sorted(pauli_by_index.items()):
        hp_group = hp_by_index.get(group_index, {})
        source_classification = pauli_group.get(
            "source_classification",
            hp_group.get("classification", "unknown"),
        )
        if source_classification == "zero_body_scalar":
            continue
        h_mu = group_pauli_l1_norm(pauli_group)
        weight = h_mu
        if weight <= 0.0:
            continue
        entries.append(
            {
                "group_index": group_index,
                "h_mu": float(h_mu),
                "w_mu": float(weight),
                "weight": float(weight),
                "coefficient_scale_for_G_mu": float(1.0 / h_mu),
                "source_classification": source_classification,
                "num_pauli_terms": int(pauli_group["num_pauli_terms"]),
                "num_identity_terms": int(pauli_group["num_identity_terms"]),
                "num_non_identity_terms": int(pauli_group["num_non_identity_terms"]),
            }
        )

    if not entries:
        raise ValueError("No nontrivial positive-weight groups available for qDRIFT.")

    weight_sum = float(sum(entry["weight"] for entry in entries))
    for entry in entries:
        entry["probability"] = float(entry["weight"] / weight_sum)

    return {
        "entries": entries,
        "group_indices": np.array([entry["group_index"] for entry in entries], dtype=int),
        "probabilities": np.array([entry["probability"] for entry in entries], dtype=float),
        "weight_sum_lambda": weight_sum,
        "pauli_by_index": pauli_by_index,
        "scalar_energy": scalar_energy_from_grouped_paulis(pauli_archive),
    }


def build_stochastic_qdrift_instance(
    sampling_model,
    total_time,
    rng,
    num_qubits,
    basis_gates,
    power,
    instance_index,
):
    circuit = QuantumCircuit(
        num_qubits,
        name=f"suqk_qdrift_k{power}_sample{instance_index}",
    )
    segment_angle = (
        sampling_model["weight_sum_lambda"]
        * total_time
        / QDRIFT_SEGMENT_COUNT_ND
    )

    sampled_indices = rng.choice(
        sampling_model["group_indices"],
        size=QDRIFT_SEGMENT_COUNT_ND,
        replace=True,
        p=sampling_model["probabilities"],
    )
    entries_by_index = {
        int(entry["group_index"]): entry
        for entry in sampling_model["entries"]
    }
    history = []
    for segment_index, sampled_index in enumerate(sampled_indices):
        sampled_index = int(sampled_index)
        entry = entries_by_index[sampled_index]
        group_circuit = build_group_evolution_circuit(
            sampling_model["pauli_by_index"][sampled_index],
            num_qubits,
            segment_angle,
            basis_gates,
            coefficient_scale=entry["coefficient_scale_for_G_mu"],
        )
        circuit.compose(group_circuit, inplace=True)
        history.append(
            {
                "segment": int(segment_index),
                "group_index": sampled_index,
                "h_mu": float(entry["h_mu"]),
                "w_mu": float(entry["w_mu"]),
                "weight": float(entry["weight"]),
                "probability": float(entry["probability"]),
                "segment_angle_theta": float(segment_angle),
                "coefficient_scale_for_G_mu": float(
                    entry["coefficient_scale_for_G_mu"]
                ),
                "group_depth": int(group_circuit.depth()),
                "group_operation_counts": operation_counts(group_circuit),
            }
        )

    return circuit, history


def mean_or_default(values, default):
    values = [float(value) for value in values if value is not None]
    if not values:
        return float(default)
    return float(np.mean(values))


def backend_name(backend):
    name = getattr(backend, "name", None)
    if callable(name):
        return str(name())
    if name is None:
        return str(type(backend).__name__)
    return str(name)


def target_property_values(target_backend, operation_name, attribute_name):
    """Collect Target InstructionProperties values when available."""

    target = getattr(target_backend, "target", None)
    if target is None or operation_name not in getattr(target, "operation_names", []):
        return []
    values = []
    for properties in target[operation_name].values():
        if properties is None:
            continue
        value = getattr(properties, attribute_name, None)
        if value is not None:
            values.append(float(value))
    return values


def mean_qubit_property(target_backend, attribute_name, default):
    if not hasattr(target_backend, "qubit_properties"):
        return float(default)
    num_qubits = int(getattr(target_backend, "num_qubits", 0) or 0)
    values = []
    for qubit in range(num_qubits):
        try:
            properties = target_backend.qubit_properties(qubit)
        except Exception:
            continue
        value = getattr(properties, attribute_name, None)
        if value is not None:
            values.append(float(value))
    return mean_or_default(values, default)


def build_depolarizing_thermal_error(
    depolarizing_probability,
    num_qubits,
    t1_seconds,
    t2_seconds,
    gate_time_seconds,
    include_thermal_relaxation,
):
    """Build a compact gate error used by the local noisy simulator modes.

    The depolarizing term controls stochastic Pauli-like gate error. The thermal
    term gives the model a rough T1/T2 time scale. The implementation is
    intentionally all-qubit averaged because these local simulation modes target
    active-space studies, not full-device calibration replay.
    """

    error = depolarizing_error(float(depolarizing_probability), int(num_qubits))
    if not include_thermal_relaxation:
        return error

    # Qiskit's thermal channel requires T2 <= 2*T1. Real backend loaders handle
    # this internally; for our compact averaged model we enforce the same
    # physical constraint before constructing the channel.
    t1 = float(t1_seconds)
    t2 = min(float(t2_seconds), 2.0 * t1)
    gate_time = float(gate_time_seconds)
    one_qubit_thermal = thermal_relaxation_error(t1, t2, gate_time)
    thermal = one_qubit_thermal
    for _ in range(1, int(num_qubits)):
        thermal = thermal.tensor(one_qubit_thermal)
    return error.compose(thermal)


def add_symmetric_readout_error(noise_model, probability):
    probability = float(probability)
    if probability <= 0.0:
        return
    readout_error = ReadoutError(
        [
            [1.0 - probability, probability],
            [probability, 1.0 - probability],
        ]
    )
    noise_model.add_all_qubit_readout_error(readout_error)


def build_simple_noise_model():
    noise_model = NoiseModel(basis_gates=SIMPLE_NOISE_BASIS_GATES)
    one_qubit_error = build_depolarizing_thermal_error(
        SIMPLE_NOISE_ONE_QUBIT_DEPOLARIZING_PROBABILITY,
        1,
        SIMPLE_NOISE_T1_SECONDS,
        SIMPLE_NOISE_T2_SECONDS,
        SIMPLE_NOISE_ONE_QUBIT_GATE_TIME_SECONDS,
        SIMPLE_NOISE_INCLUDE_THERMAL_RELAXATION,
    )
    two_qubit_error = build_depolarizing_thermal_error(
        SIMPLE_NOISE_TWO_QUBIT_DEPOLARIZING_PROBABILITY,
        2,
        SIMPLE_NOISE_T1_SECONDS,
        SIMPLE_NOISE_T2_SECONDS,
        SIMPLE_NOISE_TWO_QUBIT_GATE_TIME_SECONDS,
        SIMPLE_NOISE_INCLUDE_THERMAL_RELAXATION,
    )

    # RZ is virtual in this basis, so it intentionally receives no explicit
    # quantum error. Measurement noise is added separately below.
    noise_model.add_all_qubit_quantum_error(one_qubit_error, ["id", "sx", "x"])
    noise_model.add_all_qubit_quantum_error(two_qubit_error, ["cx"])
    add_symmetric_readout_error(
        noise_model,
        SIMPLE_NOISE_READOUT_ERROR_PROBABILITY,
    )
    metadata = {
        "model_kind": "simple_all_qubit_noise",
        "basis_gates": list(SIMPLE_NOISE_BASIS_GATES),
        "one_qubit_depolarizing_probability": (
            SIMPLE_NOISE_ONE_QUBIT_DEPOLARIZING_PROBABILITY
        ),
        "two_qubit_depolarizing_probability": (
            SIMPLE_NOISE_TWO_QUBIT_DEPOLARIZING_PROBABILITY
        ),
        "readout_error_probability": SIMPLE_NOISE_READOUT_ERROR_PROBABILITY,
        "include_thermal_relaxation": SIMPLE_NOISE_INCLUDE_THERMAL_RELAXATION,
        "t1_seconds": SIMPLE_NOISE_T1_SECONDS,
        "t2_seconds": SIMPLE_NOISE_T2_SECONDS,
        "one_qubit_gate_time_seconds": SIMPLE_NOISE_ONE_QUBIT_GATE_TIME_SECONDS,
        "two_qubit_gate_time_seconds": SIMPLE_NOISE_TWO_QUBIT_GATE_TIME_SECONDS,
    }
    return noise_model, metadata


def load_ibm_model_target_backend():
    if IBM_MODEL_SOURCE == "fake_backend":
        import qiskit_ibm_runtime.fake_provider as fake_provider

        backend_class = getattr(fake_provider, IBM_MODEL_FAKE_BACKEND_CLASS)
        return backend_class()

    from qiskit_ibm_runtime import QiskitRuntimeService

    if IBM_MODEL_RUNTIME_INSTANCE:
        service = QiskitRuntimeService(instance=IBM_MODEL_RUNTIME_INSTANCE)
    else:
        service = QiskitRuntimeService()
    return service.backend(IBM_MODEL_RUNTIME_BACKEND_NAME)


def build_compact_ibm_noise_model(target_backend):
    """Build an active-space-sized IBM-style model from target averages."""

    noise_model = NoiseModel(basis_gates=IBM_MODEL_BASIS_GATES)
    target_name = backend_name(target_backend)
    t1 = mean_qubit_property(target_backend, "t1", SIMPLE_NOISE_T1_SECONDS)
    t2 = mean_qubit_property(target_backend, "t2", SIMPLE_NOISE_T2_SECONDS)
    readout_probability = mean_or_default(
        target_property_values(target_backend, "measure", "error"),
        SIMPLE_NOISE_READOUT_ERROR_PROBABILITY,
    )

    one_qubit_error_by_gate = {}
    one_qubit_time_by_gate = {}
    for gate_name in ["id", "sx", "x"]:
        one_qubit_error_by_gate[gate_name] = mean_or_default(
            target_property_values(target_backend, gate_name, "error"),
            SIMPLE_NOISE_ONE_QUBIT_DEPOLARIZING_PROBABILITY,
        )
        one_qubit_time_by_gate[gate_name] = mean_or_default(
            target_property_values(target_backend, gate_name, "duration"),
            SIMPLE_NOISE_ONE_QUBIT_GATE_TIME_SECONDS,
        )

    two_qubit_probability = mean_or_default(
        target_property_values(target_backend, IBM_MODEL_TWO_QUBIT_GATE, "error"),
        SIMPLE_NOISE_TWO_QUBIT_DEPOLARIZING_PROBABILITY,
    )
    two_qubit_gate_time = mean_or_default(
        target_property_values(target_backend, IBM_MODEL_TWO_QUBIT_GATE, "duration"),
        SIMPLE_NOISE_TWO_QUBIT_GATE_TIME_SECONDS,
    )

    for gate_name in ["id", "sx", "x"]:
        gate_error = build_depolarizing_thermal_error(
            one_qubit_error_by_gate[gate_name],
            1,
            t1,
            t2,
            one_qubit_time_by_gate[gate_name],
            include_thermal_relaxation=True,
        )
        noise_model.add_all_qubit_quantum_error(gate_error, [gate_name])

    two_qubit_error = build_depolarizing_thermal_error(
        two_qubit_probability,
        2,
        t1,
        t2,
        two_qubit_gate_time,
        include_thermal_relaxation=True,
    )
    noise_model.add_all_qubit_quantum_error(
        two_qubit_error,
        [IBM_MODEL_TWO_QUBIT_GATE],
    )
    add_symmetric_readout_error(noise_model, readout_probability)

    metadata = {
        "model_kind": "ibm_target_averaged_active_space_noise",
        "target_backend_name": target_name,
        "target_backend_num_qubits": int(getattr(target_backend, "num_qubits", 0) or 0),
        "target_source": IBM_MODEL_SOURCE,
        "fake_backend_class": IBM_MODEL_FAKE_BACKEND_CLASS,
        "runtime_backend_name": IBM_MODEL_RUNTIME_BACKEND_NAME,
        "runtime_instance": IBM_MODEL_RUNTIME_INSTANCE,
        "compressed_to_active_space": True,
        "basis_gates": list(IBM_MODEL_BASIS_GATES),
        "two_qubit_gate": IBM_MODEL_TWO_QUBIT_GATE,
        "mean_t1_seconds": t1,
        "mean_t2_seconds": t2,
        "mean_readout_error_probability": readout_probability,
        "mean_one_qubit_error_by_gate": one_qubit_error_by_gate,
        "mean_one_qubit_gate_time_seconds_by_gate": one_qubit_time_by_gate,
        "mean_two_qubit_error_probability": two_qubit_probability,
        "mean_two_qubit_gate_time_seconds": two_qubit_gate_time,
        "note": (
            "This compact model derives average rates from the selected IBM "
            "target and applies them to all active-space qubits. It is intended "
            "for molecule-scale local simulation, not full-device calibration "
            "replay with a hardware coupling map."
        ),
    }
    return noise_model, metadata


def build_backend(num_qubits):
    if BACKEND_MODE == "local_noiseless_statevector":
        backend = AerSimulator(method="statevector", seed_simulator=RANDOM_SEED)
        return backend, {
            "backend_mode": BACKEND_MODE,
            "simulator_method": "statevector",
            "noise_model": None,
            "final_mfe_transpilation": False,
        }

    if BACKEND_MODE == "local_noisy_simple":
        noise_model, noise_metadata = build_simple_noise_model()
        backend = AerSimulator(
            method=NOISY_SIMULATION_METHOD,
            noise_model=noise_model,
            basis_gates=SIMPLE_NOISE_BASIS_GATES,
            seed_simulator=RANDOM_SEED,
        )
        return backend, {
            "backend_mode": BACKEND_MODE,
            "simulator_method": NOISY_SIMULATION_METHOD,
            "noise_model": noise_metadata,
            "active_space_num_qubits": int(num_qubits),
            "final_mfe_transpilation": TRANSPILE_MFE_CIRCUITS_FOR_NOISY_BACKEND,
            "final_mfe_transpile_optimization_level": (
                NOISY_TRANSPILE_OPTIMIZATION_LEVEL
            ),
        }

    if BACKEND_MODE == "local_noisy_ibm_model":
        target_backend = load_ibm_model_target_backend()
        if IBM_MODEL_COMPRESS_TO_ACTIVE_SPACE:
            noise_model, noise_metadata = build_compact_ibm_noise_model(
                target_backend
            )
            backend = AerSimulator(
                method=NOISY_SIMULATION_METHOD,
                noise_model=noise_model,
                basis_gates=IBM_MODEL_BASIS_GATES,
                seed_simulator=RANDOM_SEED,
            )
            return backend, {
                "backend_mode": BACKEND_MODE,
                "simulator_method": NOISY_SIMULATION_METHOD,
                "noise_model": noise_metadata,
                "active_space_num_qubits": int(num_qubits),
                "final_mfe_transpilation": TRANSPILE_MFE_CIRCUITS_FOR_NOISY_BACKEND,
                "final_mfe_transpile_optimization_level": (
                    NOISY_TRANSPILE_OPTIMIZATION_LEVEL
                ),
            }

        backend = AerSimulator.from_backend(
            target_backend,
            method=NOISY_SIMULATION_METHOD,
            seed_simulator=RANDOM_SEED,
        )
        return backend, {
            "backend_mode": BACKEND_MODE,
            "simulator_method": NOISY_SIMULATION_METHOD,
            "noise_model": {
                "model_kind": "aer_simulator_from_backend",
                "target_backend_name": backend_name(target_backend),
                "target_backend_num_qubits": int(
                    getattr(target_backend, "num_qubits", 0) or 0
                ),
                "target_source": IBM_MODEL_SOURCE,
                "fake_backend_class": IBM_MODEL_FAKE_BACKEND_CLASS,
                "runtime_backend_name": IBM_MODEL_RUNTIME_BACKEND_NAME,
                "runtime_instance": IBM_MODEL_RUNTIME_INSTANCE,
                "compressed_to_active_space": False,
                "note": (
                    "AerSimulator.from_backend was used directly. Be careful "
                    "with density_matrix simulations because final transpilation "
                    "to a full device target can widen a small active-space "
                    "circuit to the full hardware qubit count."
                ),
            },
            "active_space_num_qubits": int(num_qubits),
            "final_mfe_transpilation": TRANSPILE_MFE_CIRCUITS_FOR_NOISY_BACKEND,
            "final_mfe_transpile_optimization_level": (
                NOISY_TRANSPILE_OPTIMIZATION_LEVEL
            ),
        }
    raise ValueError(f"Unsupported BACKEND_MODE={BACKEND_MODE!r}.")


def maybe_transpile_mfe_circuits(backend, circuits):
    if BACKEND_MODE == "local_noiseless_statevector":
        return circuits
    if not TRANSPILE_MFE_CIRCUITS_FOR_NOISY_BACKEND:
        return circuits
    return transpile(
        circuits,
        backend=backend,
        optimization_level=NOISY_TRANSPILE_OPTIMIZATION_LEVEL,
        seed_transpiler=RANDOM_SEED,
    )


def run_mfe_for_power(backend, power_circuit, occupation, hf_count_key, verbose):
    templates = build_mfe_templates(power_circuit, occupation, verbose=verbose)
    labels = [F1_LABEL, F2_PLUS_LABEL, F2_I_LABEL]
    circuits = [templates[label] for label in labels]
    execution_circuits = maybe_transpile_mfe_circuits(backend, circuits)
    result = backend.run(
        execution_circuits,
        shots=SHOTS_PER_MFE_EXPERIMENT,
    ).result()
    counts_by_label = {
        label: {key: int(value) for key, value in result.get_counts(index).items()}
        for index, label in enumerate(labels)
    }
    estimate = estimate_z_from_counts(counts_by_label, hf_count_key, verbose=verbose)
    fidelities = {
        "F1": estimate.f1,
        "F2_plus": estimate.f2_plus,
        "F2_i": estimate.f2_i,
    }
    depths = {label: int(circuit.depth()) for label, circuit in templates.items()}
    ops = {label: operation_counts(circuit) for label, circuit in templates.items()}
    execution_depths = {
        label: int(circuit.depth())
        for label, circuit in zip(labels, execution_circuits)
    }
    execution_ops = {
        label: operation_counts(circuit)
        for label, circuit in zip(labels, execution_circuits)
    }
    return {
        "counts": counts_by_label,
        "fidelities": fidelities,
        "estimate": {
            "real": estimate.real,
            "imag": estimate.imag,
            "abs": abs(estimate.z),
        },
        "template_depths": depths,
        "template_operation_counts": ops,
        "execution_template_depths": execution_depths,
        "execution_template_operation_counts": execution_ops,
        "final_mfe_transpiled_for_backend": bool(
            execution_circuits is not circuits
        ),
    }


def hf_return_probability_exact(circuit, hf_count_key):
    no_measurements = circuit.remove_final_measurements(inplace=False)
    state = Statevector.from_instruction(no_measurements)
    probability = float(state.probabilities_dict().get(hf_count_key, 0.0))
    if abs(probability) < 1.0e-14:
        probability = 0.0
    if abs(probability - 1.0) < 1.0e-14:
        probability = 1.0
    return probability


def run_exact_mfe_for_power(power_circuit, occupation, hf_count_key, verbose):
    templates = build_mfe_templates(power_circuit, occupation, verbose=verbose)
    labels = [F1_LABEL, F2_PLUS_LABEL, F2_I_LABEL]
    probabilities_by_label = {
        label: hf_return_probability_exact(templates[label], hf_count_key)
        for label in labels
    }
    fidelities = MFEFidelities(
        f1=probabilities_by_label[F1_LABEL],
        f2_plus=probabilities_by_label[F2_PLUS_LABEL],
        f2_i=probabilities_by_label[F2_I_LABEL],
    )
    estimate = estimate_z_from_fidelities(fidelities, verbose=verbose)
    fidelity_record = {
        "F1": estimate.f1,
        "F2_plus": estimate.f2_plus,
        "F2_i": estimate.f2_i,
    }
    depths = {label: int(circuit.depth()) for label, circuit in templates.items()}
    ops = {label: operation_counts(circuit) for label, circuit in templates.items()}
    return {
        "counts": None,
        "exact_return_probabilities": {
            "F1": probabilities_by_label[F1_LABEL],
            "F2_plus": probabilities_by_label[F2_PLUS_LABEL],
            "F2_i": probabilities_by_label[F2_I_LABEL],
        },
        "fidelities": fidelity_record,
        "estimate": {
            "real": estimate.real,
            "imag": estimate.imag,
            "abs": abs(estimate.z),
        },
        "template_depths": depths,
        "template_operation_counts": ops,
        "execution_template_depths": depths,
        "execution_template_operation_counts": ops,
        "final_mfe_transpiled_for_backend": False,
        "mfe_measurement_mode": "exact_statevector_probabilities",
        "shots_per_mfe_experiment_used": None,
    }


def count_totals_by_label(instance_records):
    totals = {}
    for label in [F1_LABEL, F2_PLUS_LABEL, F2_I_LABEL]:
        label_totals = {}
        for record in instance_records:
            for bitstring, count in record["counts"][label].items():
                label_totals[bitstring] = label_totals.get(bitstring, 0) + int(count)
        totals[label] = label_totals
    return totals


def average_fidelities(instance_records):
    return {
        key: float(np.mean([record["fidelities"][key] for record in instance_records]))
        for key in ["F1", "F2_plus", "F2_i"]
    }


def average_estimate(instance_records):
    values = np.array(
        [
            complex(record["estimate"]["real"], record["estimate"]["imag"])
            for record in instance_records
        ],
        dtype=np.complex128,
    )
    return complex(np.mean(values))


def standard_record_for_power(
    backend,
    full_step,
    power,
    occupation,
    hf_count_key,
):
    power_circuit = build_trotter_power(full_step, power)
    verbose = MFE_VERBOSE_FOR_FIRST_NONZERO_POWER and power == 1
    print(
        f"\nC_{power}: STANDARD V_{power}=U^{power}, "
        f"depth {power_circuit.depth()}, size {power_circuit.size()}"
    )
    record = run_mfe_for_power(
        backend,
        power_circuit,
        occupation,
        hf_count_key,
        verbose=verbose,
    )
    # Keep the raw MFE estimate as the physical non-scalar correlation. The
    # saved Qiskit circuits may omit identity Pauli terms as unobservable global
    # phases, but the MFE superposition experiment measures the HF branch
    # relative to the vacuum branch. For normal-ordered non-scalar molecular
    # terms, that raw reference-frame estimate restores the identity/Z vacuum
    # cancellation that the bare circuit cannot reveal by measurement. The
    # zero-body scalar h_o is the only phase handled separately in main().
    raw_mfe_value = complex(record["estimate"]["real"], record["estimate"]["imag"])
    measured_value = raw_mfe_value
    record.update(
        {
            "mode": "standard",
            "power": power,
            "num_instances": 1,
            "raw_mfe_relative_correlation": {
                "real": float(raw_mfe_value.real),
                "imag": float(raw_mfe_value.imag),
            },
            "mfe_reference_policy": (
                "Use raw MFE y as the non-scalar correlation; do not divide by "
                "<vac|V|vac> from a simulator or full-Trotter reference circuit."
            ),
            "raw_mfe_non_scalar_correlation": {
                "real": float(measured_value.real),
                "imag": float(measured_value.imag),
            },
            "power_circuit_depth": int(power_circuit.depth()),
            "power_circuit_size": int(power_circuit.size()),
            "power_circuit_operation_counts": operation_counts(power_circuit),
            "sampled_group_histories": [],
            "mean_fidelities": record["fidelities"],
            "summed_counts": record["counts"],
        }
    )
    return measured_value, record


def exact_trotter_record_for_power(
    full_step,
    power,
    occupation,
    hf_count_key,
):
    power_circuit = build_trotter_power(full_step, power)
    verbose = MFE_VERBOSE_FOR_FIRST_NONZERO_POWER and power == 1
    print(
        f"\nC_{power}: EXACT_TROTTER V_{power}=U^{power}, "
        f"depth {power_circuit.depth()}, size {power_circuit.size()}"
    )
    record = run_exact_mfe_for_power(
        power_circuit,
        occupation,
        hf_count_key,
        verbose=verbose,
    )
    raw_mfe_value = complex(record["estimate"]["real"], record["estimate"]["imag"])
    measured_value = raw_mfe_value
    record.update(
        {
            "mode": "exact_trotter",
            "power": power,
            "num_instances": 1,
            "raw_mfe_relative_correlation": {
                "real": float(raw_mfe_value.real),
                "imag": float(raw_mfe_value.imag),
            },
            "mfe_reference_policy": (
                "Use exact MFE return probabilities as the non-scalar "
                "correlation; do not divide by <vac|V|vac> from a simulator "
                "or full-Trotter reference circuit."
            ),
            "raw_mfe_non_scalar_correlation": {
                "real": float(measured_value.real),
                "imag": float(measured_value.imag),
            },
            "power_circuit_depth": int(power_circuit.depth()),
            "power_circuit_size": int(power_circuit.size()),
            "power_circuit_operation_counts": operation_counts(power_circuit),
            "sampled_group_histories": [],
            "mean_fidelities": record["fidelities"],
            "summed_counts": None,
        }
    )
    return measured_value, record


def stochastic_record_for_power(
    backend,
    sampling_model,
    power,
    occupation,
    hf_count_key,
    num_qubits,
    basis_gates,
    rng,
    dt,
):
    total_time = power * dt
    if power == 0:
        identity = QuantumCircuit(num_qubits, name="suqk_identity_power_0")
        print("\nC_0: STOCHASTIC mode still measures identity once for diagnostics")
        record = run_mfe_for_power(
            backend,
            identity,
            occupation,
            hf_count_key,
            verbose=False,
        )
        raw_mfe_value = complex(record["estimate"]["real"], record["estimate"]["imag"])
        measured_value = raw_mfe_value
        record.update(
            {
                "mode": "stochastic",
                "power": power,
                "num_instances": 1,
                "total_time": float(total_time),
                "qdrift_segment_count": 0,
                "raw_mfe_relative_correlation": {
                    "real": float(raw_mfe_value.real),
                    "imag": float(raw_mfe_value.imag),
                },
                "mfe_reference_policy": (
                    "Use raw MFE y as the non-scalar qDRIFT correlation; no "
                    "per-instance reference-branch correction is applied."
                ),
                "raw_mfe_non_scalar_correlation": {
                    "real": float(measured_value.real),
                    "imag": float(measured_value.imag),
                },
                "power_circuit_depth": 0,
                "power_circuit_size": 0,
                "power_circuit_operation_counts": {},
                "sampled_group_histories": [],
                "instance_records": [],
                "mean_fidelities": record["fidelities"],
                "summed_counts": record["counts"],
            }
        )
        return measured_value, record

    print(
        f"\nC_{power}: STOCHASTIC qDRIFT total_time={total_time:.8f}, "
        f"N_d={QDRIFT_SEGMENT_COUNT_ND}, "
        f"instances={STOCHASTIC_INSTANCES_PER_CORRELATION}"
    )
    instance_records = []
    sampled_group_histories = []
    for instance_index in range(STOCHASTIC_INSTANCES_PER_CORRELATION):
        chunk, history = build_stochastic_qdrift_instance(
            sampling_model,
            total_time,
            rng,
            num_qubits,
            basis_gates,
            power,
            instance_index,
        )
        verbose = (
            MFE_VERBOSE_FOR_FIRST_NONZERO_POWER
            and power == 1
            and instance_index == 0
        )
        print(
            f"  sample {instance_index}: groups "
            f"{[item['group_index'] for item in history]}, "
            f"theta {[round(item['segment_angle_theta'], 8) for item in history]}, "
            f"depth {chunk.depth()}, size {chunk.size()}"
        )
        instance_record = run_mfe_for_power(
            backend,
            chunk,
            occupation,
            hf_count_key,
            verbose=verbose,
        )
        # qDRIFT is a randomized channel: in the hardware workflow, each shot
        # would use a sampled circuit and then we forget the random label. The
        # MFE estimator must therefore average the measured F1/F2 statistics (or
        # equivalently the raw y_omega values for equal shot counts). Do not
        # divide each sample by its own <vac|V_omega|vac>; that would reinsert a
        # random circuit/global phase and would estimate a different coherent
        # object, not the qDRIFT mixed-channel measurement.
        raw_mfe_value = complex(
            instance_record["estimate"]["real"],
            instance_record["estimate"]["imag"],
        )
        instance_record.update(
            {
                "instance_index": int(instance_index),
                "chunk_depth": int(chunk.depth()),
                "chunk_size": int(chunk.size()),
                "chunk_operation_counts": operation_counts(chunk),
                "sampled_group_history": history,
                "raw_mfe_relative_correlation": {
                    "real": float(raw_mfe_value.real),
                    "imag": float(raw_mfe_value.imag),
                },
                "mfe_reference_policy": "raw qDRIFT MFE sample; no per-sample reference correction",
            }
        )
        instance_records.append(instance_record)
        sampled_group_histories.append(history)

    raw_instance_mean_value = average_estimate(instance_records)
    summed_counts = count_totals_by_label(instance_records)
    aggregate_estimate = estimate_z_from_counts(
        summed_counts,
        hf_count_key,
        verbose=False,
    )
    measured_value = complex(aggregate_estimate.real, aggregate_estimate.imag)
    max_depth = max(record["chunk_depth"] for record in instance_records)
    mean_depth = float(np.mean([record["chunk_depth"] for record in instance_records]))
    record = {
        "mode": "stochastic",
        "power": power,
        "num_instances": STOCHASTIC_INSTANCES_PER_CORRELATION,
        "total_time": float(total_time),
        "qdrift_segment_count": QDRIFT_SEGMENT_COUNT_ND,
        "power_circuit_depth": int(max_depth),
        "mean_power_circuit_depth": mean_depth,
        "power_circuit_size": int(max(record["chunk_size"] for record in instance_records)),
        "power_circuit_operation_counts": {
            "note": "See per-instance chunk_operation_counts for stochastic mode."
        },
        "counts": summed_counts,
        "fidelities": {
            "F1": aggregate_estimate.f1,
            "F2_plus": aggregate_estimate.f2_plus,
            "F2_i": aggregate_estimate.f2_i,
        },
        "estimate": {
            "real": float(measured_value.real),
            "imag": float(measured_value.imag),
            "abs": float(abs(measured_value)),
        },
        "raw_mfe_relative_correlation_mean": {
            "real": float(measured_value.real),
            "imag": float(measured_value.imag),
            "abs": float(abs(measured_value)),
        },
        "raw_mfe_relative_correlation_instance_mean": {
            "real": float(raw_instance_mean_value.real),
            "imag": float(raw_instance_mean_value.imag),
            "abs": float(abs(raw_instance_mean_value)),
        },
        "raw_mfe_non_scalar_correlation": {
            "real": float(measured_value.real),
            "imag": float(measured_value.imag),
            "abs": float(abs(measured_value)),
        },
        "mfe_reference_policy": (
            "Aggregate raw qDRIFT MFE counts/statistics across sampled circuits; "
            "do not apply per-instance <vac|V_omega|vac> correction."
        ),
        "template_depths": {
            "note": "See per-instance template_depths for stochastic mode."
        },
        "template_operation_counts": {
            "note": "See per-instance template_operation_counts for stochastic mode."
        },
        "sampled_group_histories": sampled_group_histories,
        "instance_records": instance_records,
        "mean_fidelities": average_fidelities(instance_records),
        "summed_counts": summed_counts,
    }
    return measured_value, record


def exact_stochastic_record_for_power(
    sampling_model,
    power,
    occupation,
    hf_count_key,
    num_qubits,
    basis_gates,
    rng,
    dt,
):
    total_time = power * dt
    if power == 0:
        identity = QuantumCircuit(num_qubits, name="exact_suqk_identity_power_0")
        print("\nC_0: EXACT_STOCHASTIC mode evaluates identity once for diagnostics")
        record = run_exact_mfe_for_power(
            identity,
            occupation,
            hf_count_key,
            verbose=False,
        )
        raw_mfe_value = complex(record["estimate"]["real"], record["estimate"]["imag"])
        measured_value = raw_mfe_value
        record.update(
            {
                "mode": "exact_stochastic",
                "power": power,
                "num_instances": 1,
                "total_time": float(total_time),
                "qdrift_segment_count": 0,
                "raw_mfe_relative_correlation": {
                    "real": float(raw_mfe_value.real),
                    "imag": float(raw_mfe_value.imag),
                },
                "mfe_reference_policy": (
                    "Use exact MFE probabilities as the non-scalar qDRIFT "
                    "correlation; no per-instance reference-branch correction "
                    "is applied."
                ),
                "raw_mfe_non_scalar_correlation": {
                    "real": float(measured_value.real),
                    "imag": float(measured_value.imag),
                },
                "aggregate_exact_return_probabilities": record[
                    "exact_return_probabilities"
                ],
                "power_circuit_depth": 0,
                "power_circuit_size": 0,
                "power_circuit_operation_counts": {},
                "sampled_group_histories": [],
                "instance_records": [],
                "mean_fidelities": record["fidelities"],
                "summed_counts": None,
            }
        )
        return measured_value, record

    print(
        f"\nC_{power}: EXACT_STOCHASTIC qDRIFT total_time={total_time:.8f}, "
        f"N_d={QDRIFT_SEGMENT_COUNT_ND}, "
        f"instances={STOCHASTIC_INSTANCES_PER_CORRELATION}"
    )
    instance_records = []
    sampled_group_histories = []
    for instance_index in range(STOCHASTIC_INSTANCES_PER_CORRELATION):
        chunk, history = build_stochastic_qdrift_instance(
            sampling_model,
            total_time,
            rng,
            num_qubits,
            basis_gates,
            power,
            instance_index,
        )
        verbose = (
            MFE_VERBOSE_FOR_FIRST_NONZERO_POWER
            and power == 1
            and instance_index == 0
        )
        print(
            f"  exact sample {instance_index}: groups "
            f"{[item['group_index'] for item in history]}, "
            f"theta {[round(item['segment_angle_theta'], 8) for item in history]}, "
            f"depth {chunk.depth()}, size {chunk.size()}"
        )
        instance_record = run_exact_mfe_for_power(
            chunk,
            occupation,
            hf_count_key,
            verbose=verbose,
        )
        # qDRIFT is a randomized channel. Exact-stochastic mode removes only
        # finite measurement noise: it still forgets the sampled circuit label by
        # averaging the exact F1/F2 probabilities before applying MFE arithmetic.
        raw_mfe_value = complex(
            instance_record["estimate"]["real"],
            instance_record["estimate"]["imag"],
        )
        instance_record.update(
            {
                "instance_index": int(instance_index),
                "chunk_depth": int(chunk.depth()),
                "chunk_size": int(chunk.size()),
                "chunk_operation_counts": operation_counts(chunk),
                "sampled_group_history": history,
                "raw_mfe_relative_correlation": {
                    "real": float(raw_mfe_value.real),
                    "imag": float(raw_mfe_value.imag),
                },
                "mfe_reference_policy": (
                    "exact qDRIFT MFE sample; no per-sample reference correction"
                ),
            }
        )
        instance_records.append(instance_record)
        sampled_group_histories.append(history)

    aggregate_fidelities = average_fidelities(instance_records)
    aggregate_estimate = estimate_z_from_fidelities(
        MFEFidelities(
            f1=aggregate_fidelities["F1"],
            f2_plus=aggregate_fidelities["F2_plus"],
            f2_i=aggregate_fidelities["F2_i"],
        ),
        verbose=False,
    )
    measured_value = complex(aggregate_estimate.real, aggregate_estimate.imag)
    raw_instance_mean_value = average_estimate(instance_records)
    max_depth = max(record["chunk_depth"] for record in instance_records)
    mean_depth = float(np.mean([record["chunk_depth"] for record in instance_records]))
    record = {
        "mode": "exact_stochastic",
        "power": power,
        "num_instances": STOCHASTIC_INSTANCES_PER_CORRELATION,
        "total_time": float(total_time),
        "qdrift_segment_count": QDRIFT_SEGMENT_COUNT_ND,
        "power_circuit_depth": int(max_depth),
        "mean_power_circuit_depth": mean_depth,
        "power_circuit_size": int(max(record["chunk_size"] for record in instance_records)),
        "power_circuit_operation_counts": {
            "note": "See per-instance chunk_operation_counts for exact_stochastic mode."
        },
        "counts": None,
        "summed_counts": None,
        "exact_return_probabilities": aggregate_fidelities,
        "aggregate_exact_return_probabilities": aggregate_fidelities,
        "fidelities": {
            "F1": aggregate_estimate.f1,
            "F2_plus": aggregate_estimate.f2_plus,
            "F2_i": aggregate_estimate.f2_i,
        },
        "estimate": {
            "real": float(measured_value.real),
            "imag": float(measured_value.imag),
            "abs": float(abs(measured_value)),
        },
        "raw_mfe_relative_correlation_mean": {
            "real": float(measured_value.real),
            "imag": float(measured_value.imag),
            "abs": float(abs(measured_value)),
        },
        "raw_mfe_relative_correlation_instance_mean": {
            "real": float(raw_instance_mean_value.real),
            "imag": float(raw_instance_mean_value.imag),
            "abs": float(abs(raw_instance_mean_value)),
        },
        "raw_mfe_non_scalar_correlation": {
            "real": float(measured_value.real),
            "imag": float(measured_value.imag),
            "abs": float(abs(measured_value)),
        },
        "mfe_reference_policy": (
            "Aggregate exact qDRIFT MFE probabilities across sampled circuits; "
            "do not apply per-instance <vac|V_omega|vac> correction."
        ),
        "mfe_measurement_mode": (
            "exact_statevector_probabilities_averaged_over_qdrift_samples"
        ),
        "template_depths": {
            "note": "See per-instance template_depths for exact_stochastic mode."
        },
        "template_operation_counts": {
            "note": "See per-instance template_operation_counts for exact_stochastic mode."
        },
        "sampled_group_histories": sampled_group_histories,
        "instance_records": instance_records,
        "mean_fidelities": aggregate_fidelities,
    }
    return measured_value, record


def assemble_overlap_matrix(correlations):
    matrix = np.empty((KRYLOV_DIMENSION, KRYLOV_DIMENSION), dtype=np.complex128)
    for m in range(KRYLOV_DIMENSION):
        for n in range(KRYLOV_DIMENSION):
            diff = n - m
            if diff >= 0:
                matrix[m, n] = correlations[diff]
            else:
                matrix[m, n] = np.conjugate(correlations[-diff])
    return matrix


def complex_array_to_records(values):
    return [
        {"index": int(index), "real": float(value.real), "imag": float(value.imag)}
        for index, value in enumerate(values)
    ]


def complex_matrix_to_nested_records(matrix):
    return [
        [
            {"real": float(matrix[row, col].real), "imag": float(matrix[row, col].imag)}
            for col in range(matrix.shape[1])
        ]
        for row in range(matrix.shape[0])
    ]


def print_correlation_table(measured, enforced):
    if not PRINT_CORRELATION_TABLE:
        return
    print_header("Correlation Sequence")
    print(f"{'k':>3} {'measured C_k':>28} {'stored C_k':>28}")
    print("-" * 78)
    for k, measured_value in enumerate(measured):
        stored_value = enforced[k]
        print(
            f"{k:>3} "
            f"{measured_value.real:+.8f}{measured_value.imag:+.8f}j "
            f"{stored_value.real:+.8f}{stored_value.imag:+.8f}j"
        )


def main():
    molecule_metadata = load_json(INPUT_MOLECULE_METADATA_JSON)
    circuit_metadata = load_json(INPUT_CIRCUIT_METADATA_JSON)
    hp_archive = load_json(INPUT_HERMITIAN_PAIR_JSON)
    pauli_archive = load_json(INPUT_GROUPED_PAULI_JSON)
    validate_options(circuit_metadata)
    print_startup(circuit_metadata, molecule_metadata)
    print_user_option_guide()

    occupation, hf_count_key = validate_hf_metadata(molecule_metadata)
    circuit_hf_key = circuit_metadata["hf_reference"][
        "qiskit_counts_bitstring_if_measured_q_to_c_same_index"
    ]
    if hf_count_key != circuit_hf_key:
        raise ValueError(
            f"HF count key mismatch: molecule metadata has {hf_count_key}, "
            f"circuit metadata has {circuit_hf_key}."
        )

    group_circuits = load_qpy_circuits(INPUT_QPY)
    full_step, full_step_group_indices, skipped_scalar_group_indices = (
        build_one_trotter_step(group_circuits, circuit_metadata)
    )
    basis_gates = circuit_metadata["options"]["generic_basis_gates"]
    num_qubits = int(circuit_metadata["active_space"]["num_qubits"])
    dt = float(circuit_metadata["options"]["dt"])
    sampling_model = build_qdrift_sampling_model(hp_archive, pauli_archive)

    print_header("Full Trotter Step Summary")
    print_kv("Grouped circuits loaded:", len(group_circuits))
    print_kv("Non-scalar groups included:", len(full_step_group_indices))
    print_kv("Scalar groups skipped:", skipped_scalar_group_indices)
    print_kv("Non-scalar full-step depth:", full_step.depth())
    print_kv("Non-scalar full-step size:", full_step.size())
    print_kv("Non-scalar full-step op counts:", operation_counts(full_step))

    print_header("qDRIFT Sampling Model")
    print_kv("Sampled non-scalar groups:", len(sampling_model["entries"]))
    print_kv(
        "qDRIFT lambda=sum non-scalar w_mu:",
        f"{sampling_model['weight_sum_lambda']:.12f}",
    )
    print_kv(
        "Scalar energy excluded from lambda:",
        f"{sampling_model['scalar_energy']:.12f}",
    )
    print_kv("Scalar phase application:", "analytic after MFE")
    print_kv("Weight convention:", STOCHASTIC_WEIGHT_CONVENTION)
    print_kv(
        "First five (mu,h_mu,p_mu):",
        [
            (
                entry["group_index"],
                round(entry["h_mu"], 8),
                round(entry["probability"], 8),
            )
            for entry in sampling_model["entries"][:5]
        ],
    )
    if UQK_MODE in QDRIFT_UQK_MODES:
        print_qdrift_circuit_structure(sampling_model, dt)
    else:
        print_header("Deterministic Trotter Circuit Structure")
        print(
            "This mode uses the saved fixed-dt QPY group circuits, skips the "
            "separate scalar group in the MFE circuit, and applies the scalar "
            "phase analytically to the final C_k."
        )

    if UQK_MODE in EXACT_MFE_UQK_MODES:
        backend = None
        backend_metadata = {
            "backend_mode": BACKEND_MODE,
            "simulator_method": "statevector_exact_probabilities",
            "noise_model": None,
            "final_mfe_transpilation": False,
            "backend_ignored_by_exact_mfe": True,
            "backend_ignored_by_exact_trotter": UQK_MODE == "exact_trotter",
            "backend_ignored_by_exact_stochastic": UQK_MODE == "exact_stochastic",
            "note": (
                f"{UQK_MODE} computes MFE return probabilities exactly with "
                "qiskit.quantum_info.Statevector and does not run a "
                "finite-shot backend."
            ),
        }
    else:
        backend, backend_metadata = build_backend(num_qubits)
    print_header("Backend Summary")
    print_kv("Backend mode:", BACKEND_MODE)
    print_kv(
        "Backend object:",
        "not used for exact MFE modes" if backend is None else backend_name(backend),
    )
    print_kv("Simulator method:", backend_metadata["simulator_method"])
    print_kv("Noise model:", backend_metadata["noise_model"])
    print_kv("Final MFE transpilation:", backend_metadata["final_mfe_transpilation"])

    rng = np.random.default_rng(RANDOM_SEED)
    measured_correlations = np.zeros(MAX_CORRELATION_POWER + 1, dtype=np.complex128)
    stored_correlations = np.zeros(MAX_CORRELATION_POWER + 1, dtype=np.complex128)
    per_power_metadata = []

    print_header("Estimating C_k With MFE")
    for power in range(MAX_CORRELATION_POWER + 1):
        if UQK_MODE == "standard":
            non_scalar_measured_value, record = standard_record_for_power(
                backend,
                full_step,
                power,
                occupation,
                hf_count_key,
            )
        elif UQK_MODE == "exact_trotter":
            non_scalar_measured_value, record = exact_trotter_record_for_power(
                full_step,
                power,
                occupation,
                hf_count_key,
            )
        elif UQK_MODE == "stochastic":
            non_scalar_measured_value, record = stochastic_record_for_power(
                backend,
                sampling_model,
                power,
                occupation,
                hf_count_key,
                num_qubits,
                basis_gates,
                rng,
                dt,
            )
        elif UQK_MODE == "exact_stochastic":
            non_scalar_measured_value, record = exact_stochastic_record_for_power(
                sampling_model,
                power,
                occupation,
                hf_count_key,
                num_qubits,
                basis_gates,
                rng,
                dt,
            )
        else:
            raise ValueError(f"Unsupported UQK_MODE={UQK_MODE!r}.")

        total_time = power * dt
        scalar_phase = np.exp(-1j * sampling_model["scalar_energy"] * total_time)
        measured_value = scalar_phase * non_scalar_measured_value
        stored_value = measured_value
        if power == 0 and ENFORCE_C0_EXACT:
            stored_value = complex(1.0, 0.0)

        measured_correlations[power] = measured_value
        stored_correlations[power] = stored_value
        reference_policy = (
            "raw_mfe_vacuum_reference_no_per_circuit_correction"
        )
        record.update(
            {
                "total_time": float(total_time),
                "scalar_energy": float(sampling_model["scalar_energy"]),
                "reference_branch_correction_applied": False,
                "mfe_reference_policy": reference_policy,
                "mfe_reference_formula": (
                    "Store raw MFE y_k as the non-scalar correlation measured "
                    "relative to the vacuum branch. Do not divide by a simulated "
                    "<vac|V_k|vac> or by per-sampled qDRIFT reference phases. "
                    "Apply only the zero-body scalar phase analytically."
                ),
                "scalar_phase_applied_analytically": {
                    "real": float(scalar_phase.real),
                    "imag": float(scalar_phase.imag),
                },
                "non_scalar_measured_correlation": {
                    "real": float(non_scalar_measured_value.real),
                    "imag": float(non_scalar_measured_value.imag),
                },
                "measured_correlation": {
                    "real": float(measured_value.real),
                    "imag": float(measured_value.imag),
                },
                "stored_correlation": {
                    "real": float(stored_value.real),
                    "imag": float(stored_value.imag),
                },
                "c0_enforced": bool(power == 0 and ENFORCE_C0_EXACT),
            }
        )
        per_power_metadata.append(record)
        if "raw_mfe_relative_correlation" in record:
            raw_for_print = complex(
                record["raw_mfe_relative_correlation"]["real"],
                record["raw_mfe_relative_correlation"]["imag"],
            )
        else:
            raw_for_print = complex(
                record["raw_mfe_relative_correlation_mean"]["real"],
                record["raw_mfe_relative_correlation_mean"]["imag"],
            )
        print(
            f"C_{power} raw MFE y = {raw_for_print.real:+.8f}"
            f"{raw_for_print.imag:+.8f}j; "
            "reference correction = not applied; "
            f"non-scalar estimator = {non_scalar_measured_value.real:+.8f}"
            f"{non_scalar_measured_value.imag:+.8f}j; "
            f"scalar phase = {scalar_phase.real:+.8f}"
            f"{scalar_phase.imag:+.8f}j; "
            f"measured = {measured_value.real:+.8f}"
            f"{measured_value.imag:+.8f}j; stored = {stored_value.real:+.8f}"
            f"{stored_value.imag:+.8f}j"
        )

    overlap_matrix = assemble_overlap_matrix(stored_correlations)
    hermiticity_error = float(np.linalg.norm(overlap_matrix - overlap_matrix.conj().T))
    condition_number = float(np.linalg.cond(overlap_matrix))

    print_correlation_table(measured_correlations, stored_correlations)

    print_header("Overlap Matrix S")
    print(overlap_matrix)
    print_kv("Hermiticity error ||S-S^dag||:", f"{hermiticity_error:.12e}")
    print_kv("Condition number:", f"{condition_number:.12e}")

    if UQK_MODE == "standard":
        mode_note = (
            "This is the deterministic finite-shot UQK branch: every time step "
            "uses the saved non-scalar Trotter block, estimates MFE return "
            "probabilities with shots, and applies the scalar phase analytically."
        )
    elif UQK_MODE == "exact_trotter":
        mode_note = (
            "This is the deterministic exact-trotter UQK branch: every time "
            "step uses the saved non-scalar Trotter block, computes MFE return "
            "probabilities exactly with Statevector, and applies the scalar "
            "phase analytically."
        )
    elif UQK_MODE == "stochastic":
        mode_note = (
            "This is the stochastic UQK branch: each nonzero C_k averages "
            "independently sampled qDRIFT grouped chunks with finite-shot MFE."
        )
    else:
        mode_note = (
            "This is the exact-stochastic UQK branch: each nonzero C_k "
            "averages independently sampled qDRIFT grouped chunks, computes "
            "MFE return probabilities exactly with Statevector for every "
            "sampled chunk, and applies the scalar phase analytically."
        )

    output_npz = output_npz_path(UQK_MODE)
    output_metadata = output_metadata_path(UQK_MODE)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_npz,
        S=overlap_matrix,
        correlations=stored_correlations,
        measured_correlations=measured_correlations,
        correlation_powers=np.arange(MAX_CORRELATION_POWER + 1, dtype=int),
        krylov_dimension=np.array(KRYLOV_DIMENSION, dtype=int),
        dt=np.array(float(circuit_metadata["options"]["dt"]), dtype=float),
    )

    metadata = {
        "schema_version": 1,
        "generated_at_utc": now_utc(),
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "inputs": {
            "qpy": str(INPUT_QPY),
            "circuit_metadata_json": str(INPUT_CIRCUIT_METADATA_JSON),
            "molecule_metadata_json": str(INPUT_MOLECULE_METADATA_JSON),
            "hermitian_pair_json": str(INPUT_HERMITIAN_PAIR_JSON),
            "grouped_pauli_json": str(INPUT_GROUPED_PAULI_JSON),
        },
        "outputs": {
            "npz": str(output_npz),
            "metadata_json": str(output_metadata),
        },
        "options": {
            "uqk_mode": UQK_MODE,
            "valid_uqk_modes": sorted(VALID_UQK_MODES),
            "krylov_dimension": KRYLOV_DIMENSION,
            "max_correlation_power": MAX_CORRELATION_POWER,
            "dt": dt,
            "trotter_order": int(circuit_metadata["options"]["trotter_sequence_order"]),
            "shots_per_mfe_experiment": SHOTS_PER_MFE_EXPERIMENT,
            "backend_mode": BACKEND_MODE,
            "valid_backend_modes": sorted(VALID_BACKEND_MODES),
            "output_file_stem_prefix": OUTPUT_FILE_STEM_PREFIX,
            "output_label_override": OUTPUT_LABEL_OVERRIDE,
            "output_label_suffix": OUTPUT_LABEL_SUFFIX,
            "noisy_simulation_method": NOISY_SIMULATION_METHOD,
            "transpile_mfe_circuits_for_noisy_backend": (
                TRANSPILE_MFE_CIRCUITS_FOR_NOISY_BACKEND
            ),
            "noisy_transpile_optimization_level": (
                NOISY_TRANSPILE_OPTIMIZATION_LEVEL
            ),
            "simple_noise_basis_gates": SIMPLE_NOISE_BASIS_GATES,
            "simple_noise_one_qubit_depolarizing_probability": (
                SIMPLE_NOISE_ONE_QUBIT_DEPOLARIZING_PROBABILITY
            ),
            "simple_noise_two_qubit_depolarizing_probability": (
                SIMPLE_NOISE_TWO_QUBIT_DEPOLARIZING_PROBABILITY
            ),
            "simple_noise_readout_error_probability": (
                SIMPLE_NOISE_READOUT_ERROR_PROBABILITY
            ),
            "simple_noise_include_thermal_relaxation": (
                SIMPLE_NOISE_INCLUDE_THERMAL_RELAXATION
            ),
            "simple_noise_t1_seconds": SIMPLE_NOISE_T1_SECONDS,
            "simple_noise_t2_seconds": SIMPLE_NOISE_T2_SECONDS,
            "simple_noise_one_qubit_gate_time_seconds": (
                SIMPLE_NOISE_ONE_QUBIT_GATE_TIME_SECONDS
            ),
            "simple_noise_two_qubit_gate_time_seconds": (
                SIMPLE_NOISE_TWO_QUBIT_GATE_TIME_SECONDS
            ),
            "ibm_model_source": IBM_MODEL_SOURCE,
            "ibm_model_fake_backend_class": IBM_MODEL_FAKE_BACKEND_CLASS,
            "ibm_model_runtime_backend_name": IBM_MODEL_RUNTIME_BACKEND_NAME,
            "ibm_model_runtime_instance": IBM_MODEL_RUNTIME_INSTANCE,
            "ibm_model_compress_to_active_space": (
                IBM_MODEL_COMPRESS_TO_ACTIVE_SPACE
            ),
            "ibm_model_basis_gates": IBM_MODEL_BASIS_GATES,
            "ibm_model_two_qubit_gate": IBM_MODEL_TWO_QUBIT_GATE,
            "qdrift_segment_count_Nd": QDRIFT_SEGMENT_COUNT_ND,
            "stochastic_instances_per_correlation": STOCHASTIC_INSTANCES_PER_CORRELATION,
            "stochastic_weight_convention": STOCHASTIC_WEIGHT_CONVENTION,
            "random_seed": RANDOM_SEED,
            "enforce_c0_exact": ENFORCE_C0_EXACT,
            "mfe_verbose_for_first_nonzero_power": MFE_VERBOSE_FOR_FIRST_NONZERO_POWER,
        },
        "option_documentation": {
            "uqk_mode": {
                "standard": "Use deterministic non-scalar Trotter blocks, finite-shot MFE, and the analytic scalar phase.",
                "exact_trotter": "Use deterministic non-scalar Trotter blocks, exact Statevector MFE return probabilities, and the analytic scalar phase.",
                "stochastic": "Average qDRIFT-sampled grouped chunks for total time T_k=k*dt with finite-shot MFE.",
                "exact_stochastic": "Average qDRIFT-sampled grouped chunks for total time T_k=k*dt with exact Statevector MFE return probabilities.",
            },
            "krylov_dimension": "Matrix dimension M; S has shape M x M.",
            "max_correlation_power": "Must be >= M-1 for S; using M also prepares C_M for projected U later.",
            "shots_per_mfe_experiment": "Shots for each of F1, F2_plus, F2_i in finite-shot modes. In stochastic mode this is per sampled instance. exact_trotter and exact_stochastic record but ignore this value.",
            "backend_mode": {
                "local_noiseless_statevector": "Aer statevector simulator with shot sampling; no credentials or hardware.",
                "local_noisy_simple": "Aer noisy simulator with compact hand-controlled depolarizing, thermal, and readout noise.",
                "local_noisy_ibm_model": "Aer noisy simulator with IBM fake/runtime backend-derived noise. The default compresses target averages to the active-space qubit count.",
                "exact_mfe_note": "exact_trotter and exact_stochastic ignore BACKEND_MODE and evaluate MFE probabilities with local Statevector simulation.",
            },
            "qdrift_segment_count_Nd": "Number of sampled grouped factors per stochastic chunk.",
            "stochastic_instances_per_correlation": "Number of independent stochastic chunks averaged for each nonzero C_k.",
            "stochastic_weight_convention": {
                "group_pauli_l1_norm": (
                    "For non-scalar K_mu=sum_rho alpha_mu,rho P_mu,rho, "
                    "use h_mu=w_mu=sum_rho |alpha_mu,rho|, G_mu=K_mu/h_mu, "
                    "lambda=sum_mu w_mu, and p_mu=w_mu/lambda."
                )
            },
            "rough_numeric_ranges": {
                "krylov_dimension": "2-4 for debugging, 5-10 for early studies, larger values need care with conditioning.",
                "shots_per_mfe_experiment": "100-1000 smoke tests, 2000-10000 smoother simulator studies.",
                "qdrift_segment_count_Nd": "2-8 debugging, 10-100 convergence studies.",
                "stochastic_instances_per_correlation": "1-5 plumbing checks, 10-100 convergence studies.",
            },
        },
        "molecule": molecule_metadata["molecule"],
        "active_space": molecule_metadata["active_space"],
        "hf_reference": molecule_metadata["hf_reference"],
        "occupied_qubits": occupied_qubits_from_occupation(occupation),
        "source_circuit_archive": {
            "num_group_circuits": len(group_circuits),
            "source_qiskit_version": circuit_metadata.get("qiskit_version"),
            "source_depth_statistics": circuit_metadata.get("depth_statistics"),
            "source_target": circuit_metadata.get("target"),
        },
        "backend": backend_metadata,
        "mfe_execution": {
            "finite_shot_sampling": UQK_MODE not in EXACT_MFE_UQK_MODES,
            "exact_statevector_probabilities": UQK_MODE in EXACT_MFE_UQK_MODES,
            "shots_per_mfe_experiment_used": (
                None
                if UQK_MODE in EXACT_MFE_UQK_MODES
                else SHOTS_PER_MFE_EXPERIMENT
            ),
        },
        "stochastic_sampling": {
            "mode": UQK_MODE in QDRIFT_UQK_MODES,
            "finite_shot_sampling": UQK_MODE == "stochastic",
            "exact_statevector_probabilities": UQK_MODE == "exact_stochastic",
            "sampled_non_scalar_groups": len(sampling_model["entries"]),
            "weight_sum_lambda": sampling_model["weight_sum_lambda"],
            "lambda_excludes_zero_body_scalar": True,
            "scalar_energy_excluded_from_lambda": sampling_model["scalar_energy"],
            "scalar_energy_applied_analytically_to_correlations": (
                sampling_model["scalar_energy"]
            ),
            "weight_convention": STOCHASTIC_WEIGHT_CONVENTION,
            "entries": sampling_model["entries"],
            "qdrift_formula": (
                "Write H=E_scalar I + sum_mu K_mu. For non-scalar groups, "
                "define h_mu=w_mu=sum_rho |alpha_mu,rho|, G_mu=K_mu/h_mu, "
                "lambda=sum_mu w_mu, and p_mu=w_mu/lambda. A sampled chunk "
                "for T_k=k*dt uses prod_s exp[-i (lambda*T_k/N_d) G_mu_s]. "
                "The zero-body scalar phase exp(-i E_scalar T_k) is applied "
                "analytically after MFE."
            ),
            "implemented_sampled_circuit": (
                "For total time T_k=k*dt, each sampled instance is "
                "prod_s exp[-i theta_k G_mu_s], theta_k=lambda*T_k/N_d. "
                "Each G_mu is rebuilt from grouped Pauli terms using "
                "PauliEvolutionGate(SparsePauliOp(K_mu/h_mu), time=theta_k)."
            ),
            "current_archive_convention_note": (
                "The grouped Pauli archive stores K_mu directly rather than a "
                "separate h_mu and normalized G_mu. This script defines an "
                "effective positive h_mu from the Pauli L1 norm and leaves all "
                "coefficient signs in G_mu."
            ),
        },
        "full_trotter_step": {
            "scalar_groups_skipped_for_mfe": skipped_scalar_group_indices,
            "non_scalar_group_indices": full_step_group_indices,
            "scalar_phase_applied_analytically_to_correlations": (
                sampling_model["scalar_energy"]
            ),
            "depth": int(full_step.depth()),
            "size": int(full_step.size()),
            "operation_counts": operation_counts(full_step),
        },
        "correlations": complex_array_to_records(stored_correlations),
        "measured_correlations": complex_array_to_records(measured_correlations),
        "overlap_matrix": complex_matrix_to_nested_records(overlap_matrix),
        "diagnostics": {
            "hermiticity_error_frobenius": hermiticity_error,
            "condition_number": condition_number,
        },
        "mfe_by_power": per_power_metadata,
        "notes": [
            mode_note,
            "C0 is evaluated for diagnostics and then stored as exactly 1 when ENFORCE_C0_EXACT is true.",
            "The S matrix uses Toeplitz symmetry S_mn=C_(n-m) and C_-k=conj(C_k).",
            "The local_noisy_ibm_model mode uses a local Aer simulator. Runtime credentials are only needed if IBM_MODEL_SOURCE='runtime_backend'.",
        ],
    }
    with output_metadata.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print_header("Saved Outputs")
    print_kv("NPZ:", output_npz)
    print_kv("Metadata JSON:", output_metadata)


if __name__ == "__main__":
    main()
