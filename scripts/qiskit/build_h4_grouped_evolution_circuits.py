# Manual run:
#   conda activate qiskit_env_v1
#   python scripts/qiskit/build_h4_grouped_evolution_circuits.py
#
# Summary:
#   Read grouped OpenFermion/Jordan-Wigner Pauli terms and build one Qiskit
#   circuit per Hermitian-pair group. Each circuit represents
#   U_mu(dt) = exp(-i dt sum_l alpha_mu_l P_mu_l). The circuits are transpiled
#   for the selected target mode and written to QPY, with a JSON metadata
#   sidecar. This script prints an intentionally verbose guide while it runs.
#
# Hard-coded options:
#   INPUT_GROUPED_PAULI_JSON = data/hamiltonians/h4_linear_sto3g_grouped_paulis.json
#   OUTPUT_QPY = circuits/transpiled/h4_linear_sto3g_grouped_evolution.qpy
#   OUTPUT_METADATA_JSON = circuits/transpiled/h4_linear_sto3g_grouped_evolution_metadata.json
#   DT = 0.1
#   EVOLUTION_METHOD = pauli_evolution_gate
#   TARGET_MODE = generic_simulator
#   IBM_BACKEND_NAME = ibm_brisbane
#   TRANSPILER_OPTIMIZATION_LEVEL = 1
#   TROTTER_SEQUENCE_ORDER = 1
#   VERBOSE = True

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import qiskit
from qiskit import QuantumCircuit, qpy, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_GROUPED_PAULI_JSON = (
    REPO_ROOT / "data" / "hamiltonians" / "h4_linear_sto3g_grouped_paulis.json"
)
OUTPUT_QPY = (
    REPO_ROOT / "circuits" / "transpiled" / "h4_linear_sto3g_grouped_evolution.qpy"
)
OUTPUT_METADATA_JSON = (
    REPO_ROOT
    / "circuits"
    / "transpiled"
    / "h4_linear_sto3g_grouped_evolution_metadata.json"
)

DT = 0.1
EVOLUTION_METHOD = "pauli_evolution_gate"
PAULI_EVOLUTION_SYNTHESIS = "lie_trotter"

# TARGET_MODE selects only the transpilation target. This script builds and
# saves circuits; it does not submit jobs.
#
# Available target modes:
#   "generic_simulator"
#       Local, topology-free transpilation to GENERIC_BASIS_GATES. This is the
#       safest default while developing and inspecting circuit construction.
#   "ibm_qpu_paid"
#       Real IBM Quantum paid-tier hardware target through Qiskit Runtime. Set
#       IBM_BACKEND_NAME to the chosen device name and make sure the paid-tier
#       account is saved in the qiskit_env_v1 environment. This mode transpiles
#       against the selected backend but still does not run on hardware.
TARGET_MODE = "generic_simulator"
IBM_RUNTIME_CHANNEL = "ibm_quantum_platform"
IBM_BACKEND_NAME = "ibm_brisbane"
IBM_INSTANCE = None
IBM_USE_FRACTIONAL_GATES = False
GENERIC_BASIS_GATES = ["rz", "sx", "x", "cx"]

# TRANSPILER_OPTIMIZATION_LEVEL controls how aggressively Qiskit rewrites each
# grouped circuit after construction. Higher is not always better for research
# iteration: it can take longer and may make circuit structure harder to read.
#
# Available optimization levels:
#   0
#       Minimal translation/routing. Best for preserving the original circuit
#       shape while debugging conventions.
#   1
#       Light optimization. Good default for readable generated circuits.
#   2
#       Medium optimization. More effort to reduce gates/depth.
#   3
#       Heavy optimization. Potentially best circuits, but slower and less
#       transparent; useful before serious hardware runs.
TRANSPILER_OPTIMIZATION_LEVEL = 1

# TROTTER_SEQUENCE_ORDER controls the sequence metadata saved beside the group
# circuits. The QPY archive contains one full-dt circuit per Hermitian-pair
# group; the metadata says how those group circuits should be composed later.
#
# Available sequence orders:
#   1
#       First-order product formula in ascending group-index order:
#       U(dt) ~= prod_mu U_mu(dt).
#   2
#       Symmetric second-order/Strang metadata:
#       prod_mu U_mu(dt/2) followed by prod_mu_reverse U_mu(dt/2).
TROTTER_SEQUENCE_ORDER = 1
COEFFICIENT_IMAG_TOLERANCE = 1.0e-10
INCLUDE_BARRIERS_BETWEEN_PAULI_TERMS = False
VERBOSE = True
PROGRESS_EVERY_N_GROUPS = 10
DETAILED_EXAMPLE_MAX_TERMS = 8

VALID_EVOLUTION_METHODS = {"manual_ladder", "pauli_evolution_gate"}
VALID_TARGET_MODES = {
    "generic_simulator": (
        "Local, topology-free transpilation to GENERIC_BASIS_GATES. "
        "No hardware coupling map is imposed."
    ),
    "ibm_qpu_paid": (
        "Paid-tier IBM Quantum hardware target via Qiskit Runtime. "
        "Uses IBM_BACKEND_NAME for transpilation only; no job is submitted."
    ),
}
VALID_TROTTER_SEQUENCE_ORDERS = {
    1: "First-order group-index product metadata with time_multiplier=1.0.",
    2: "Symmetric second-order/Strang metadata with half-step forward and reverse passes.",
}
VALID_TRANSPILER_OPTIMIZATION_LEVELS = {
    0: "Minimal translation/routing; preserves circuit shape most clearly.",
    1: "Light optimization; readable default for local development.",
    2: "Medium optimization; more effort to reduce gates and depth.",
    3: "Heavy optimization; slower and less transparent, useful before hardware runs.",
}


def now_utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def complex_from_record(record):
    return complex(float(record["real"]), float(record["imag"]))


def op_counts(circuit):
    return {name: int(count) for name, count in circuit.count_ops().items()}


def circuit_summary(circuit):
    return {
        "depth": int(circuit.depth()),
        "size": int(circuit.size()),
        "operation_counts": op_counts(circuit),
        "global_phase": float(circuit.global_phase),
    }


def print_header(title):
    if not VERBOSE:
        return
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def print_kv(label, value):
    if VERBOSE:
        print(f"{label:<34} {value}")


def validate_options():
    if EVOLUTION_METHOD not in VALID_EVOLUTION_METHODS:
        raise ValueError(
            f"EVOLUTION_METHOD must be one of {sorted(VALID_EVOLUTION_METHODS)}, "
            f"not {EVOLUTION_METHOD!r}."
        )
    if TARGET_MODE not in VALID_TARGET_MODES:
        raise ValueError(
            f"TARGET_MODE must be one of {sorted(VALID_TARGET_MODES)}, "
            f"not {TARGET_MODE!r}."
        )
    if TROTTER_SEQUENCE_ORDER not in VALID_TROTTER_SEQUENCE_ORDERS:
        raise ValueError(
            "TROTTER_SEQUENCE_ORDER must be one of "
            f"{sorted(VALID_TROTTER_SEQUENCE_ORDERS)}, not {TROTTER_SEQUENCE_ORDER!r}."
        )
    if TRANSPILER_OPTIMIZATION_LEVEL not in VALID_TRANSPILER_OPTIMIZATION_LEVELS:
        raise ValueError(
            "TRANSPILER_OPTIMIZATION_LEVEL must be one of "
            f"{sorted(VALID_TRANSPILER_OPTIMIZATION_LEVELS)}, "
            f"not {TRANSPILER_OPTIMIZATION_LEVEL!r}."
        )


def pauli_label(pauli_word, num_qubits):
    letters = ["I"] * num_qubits
    for item in pauli_word:
        letters[int(item["qubit"])] = item["pauli"]
    return "".join(reversed(letters))


def real_coefficient(term, group_index, term_index):
    coeff = complex_from_record(term["coefficient"])
    if abs(coeff.imag) > COEFFICIENT_IMAG_TOLERANCE:
        raise ValueError(
            f"Group {group_index} Pauli term {term_index} has a non-negligible "
            f"imaginary coefficient {coeff.imag}. Real-time Pauli evolution "
            "requires real Hermitian coefficients."
        )
    return float(coeff.real)


def apply_basis_change_to_z(circuit, pauli_word):
    for item in pauli_word:
        qubit = int(item["qubit"])
        pauli = item["pauli"]
        if pauli == "X":
            circuit.h(qubit)
        elif pauli == "Y":
            circuit.sdg(qubit)
            circuit.h(qubit)
        elif pauli == "Z":
            pass
        else:
            raise ValueError(f"Unsupported Pauli label {pauli!r}.")


def undo_basis_change_from_z(circuit, pauli_word):
    for item in reversed(pauli_word):
        qubit = int(item["qubit"])
        pauli = item["pauli"]
        if pauli == "X":
            circuit.h(qubit)
        elif pauli == "Y":
            circuit.h(qubit)
            circuit.s(qubit)
        elif pauli == "Z":
            pass
        else:
            raise ValueError(f"Unsupported Pauli label {pauli!r}.")


def apply_pauli_rotation(circuit, pauli_word, theta):
    if not pauli_word:
        circuit.global_phase += -theta
        return

    ordered_word = sorted(pauli_word, key=lambda item: int(item["qubit"]))
    qubits = [int(item["qubit"]) for item in ordered_word]
    target = qubits[-1]
    controls = qubits[:-1]

    apply_basis_change_to_z(circuit, ordered_word)
    for control in controls:
        circuit.cx(control, target)
    circuit.rz(2.0 * theta, target)
    for control in reversed(controls):
        circuit.cx(control, target)
    undo_basis_change_from_z(circuit, ordered_word)


def grouped_sparse_pauli_op(group, num_qubits):
    labels = []
    coefficients = []
    group_index = int(group["group_index"])
    for term in group["pauli_terms"]:
        term_index = int(term["term_index"])
        labels.append(pauli_label(term["pauli_word"], num_qubits))
        coefficients.append(real_coefficient(term, group_index, term_index))
    return SparsePauliOp(labels, coeffs=coefficients)


def pauli_evolution_synthesis():
    if PAULI_EVOLUTION_SYNTHESIS == "lie_trotter":
        return LieTrotter(reps=1)
    raise ValueError(
        f"Unsupported PAULI_EVOLUTION_SYNTHESIS={PAULI_EVOLUTION_SYNTHESIS!r}."
    )


def append_pauli_evolution_gate(circuit, group, num_qubits):
    operator = grouped_sparse_pauli_op(group, num_qubits)
    gate = PauliEvolutionGate(
        operator,
        time=DT,
        synthesis=pauli_evolution_synthesis(),
    )
    circuit.append(gate, range(num_qubits))


def build_raw_group_circuit(group, num_qubits):
    group_index = int(group["group_index"])
    circuit = QuantumCircuit(num_qubits, name=f"U_mu_{group_index:04d}_dt")

    if EVOLUTION_METHOD == "manual_ladder":
        for term in group["pauli_terms"]:
            term_index = int(term["term_index"])
            alpha = real_coefficient(term, group_index, term_index)
            theta = DT * alpha
            apply_pauli_rotation(circuit, term["pauli_word"], theta)
            if INCLUDE_BARRIERS_BETWEEN_PAULI_TERMS:
                circuit.barrier()
    elif EVOLUTION_METHOD == "pauli_evolution_gate":
        append_pauli_evolution_gate(circuit, group, num_qubits)

    circuit.metadata = {
        "group_index": group_index,
        "source_group_index": int(group["source_group_index"]),
        "source_classification": group["source_classification"],
        "num_pauli_terms": int(group["num_pauli_terms"]),
        "dt": DT,
        "construction": EVOLUTION_METHOD,
        "pauli_evolution_synthesis": (
            PAULI_EVOLUTION_SYNTHESIS
            if EVOLUTION_METHOD == "pauli_evolution_gate"
            else None
        ),
    }
    return circuit


def load_ibm_paid_backend():
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:
        raise RuntimeError(
            "TARGET_MODE='ibm_qpu_paid' requires qiskit_ibm_runtime in "
            "qiskit_env_v1."
        ) from exc

    service = QiskitRuntimeService(channel=IBM_RUNTIME_CHANNEL)
    backend_kwargs = {
        "name": IBM_BACKEND_NAME,
        "use_fractional_gates": IBM_USE_FRACTIONAL_GATES,
    }
    if IBM_INSTANCE is not None:
        backend_kwargs["instance"] = IBM_INSTANCE
    return service.backend(**backend_kwargs)


def resolve_transpile_backend():
    if TARGET_MODE == "generic_simulator":
        return None
    if TARGET_MODE == "ibm_qpu_paid":
        return load_ibm_paid_backend()
    raise ValueError(f"Unsupported TARGET_MODE={TARGET_MODE!r}.")


def transpile_for_target(raw_circuit, target_backend):
    if TARGET_MODE == "generic_simulator":
        return transpile(
            raw_circuit,
            basis_gates=GENERIC_BASIS_GATES,
            optimization_level=TRANSPILER_OPTIMIZATION_LEVEL,
        )
    if TARGET_MODE == "ibm_qpu_paid":
        return transpile(
            raw_circuit,
            backend=target_backend,
            optimization_level=TRANSPILER_OPTIMIZATION_LEVEL,
        )
    raise ValueError(f"Unsupported TARGET_MODE={TARGET_MODE!r}.")


def trotter_sequence(groups):
    indices = [int(group["group_index"]) for group in groups]
    if TROTTER_SEQUENCE_ORDER == 1:
        return [
            {"group_index": idx, "time_multiplier": 1.0, "position": pos}
            for pos, idx in enumerate(indices)
        ]
    if TROTTER_SEQUENCE_ORDER == 2:
        sequence = []
        for idx in indices:
            sequence.append({"group_index": idx, "time_multiplier": 0.5})
        for idx in reversed(indices):
            sequence.append({"group_index": idx, "time_multiplier": 0.5})
        for pos, item in enumerate(sequence):
            item["position"] = pos
        return sequence
    raise ValueError("TROTTER_SEQUENCE_ORDER must be 1 or 2.")


def target_backend_summary(target_backend):
    if target_backend is None:
        return None
    return {
        "name": getattr(target_backend, "name", None),
        "version": getattr(target_backend, "backend_version", None),
        "num_qubits": getattr(target_backend, "num_qubits", None),
        "provider": str(getattr(target_backend, "provider", None)),
    }


def print_startup(pauli_archive, source_hash, target_backend):
    print_header("Grouped Pauli Evolution Circuit Builder")
    print(
        "Each saved circuit is one grouped Hermitian-pair factor:\n"
        "  U_mu(dt) = exp(-i dt sum_l alpha_mu_l P_mu_l)\n"
        "The full first-order Trotter step is the product of these group circuits "
        "in group-index order."
    )

    print_header("Hard-Coded User Options")
    print_kv("Input grouped Pauli JSON:", INPUT_GROUPED_PAULI_JSON)
    print_kv("Input SHA256:", source_hash)
    print_kv("Output QPY circuit archive:", OUTPUT_QPY)
    print_kv("Output metadata JSON:", OUTPUT_METADATA_JSON)
    print_kv("dt:", DT)
    print_kv("Evolution method keyword:", EVOLUTION_METHOD)
    print_kv("Valid evolution methods:", sorted(VALID_EVOLUTION_METHODS))
    print_kv("PauliEvolution synthesis:", PAULI_EVOLUTION_SYNTHESIS)
    print_kv("Target mode:", TARGET_MODE)
    print_kv("Valid target modes:", sorted(VALID_TARGET_MODES))
    print_kv("Target mode meaning:", VALID_TARGET_MODES[TARGET_MODE])
    print_kv("IBM Runtime channel:", IBM_RUNTIME_CHANNEL)
    print_kv("IBM backend name:", IBM_BACKEND_NAME)
    print_kv("IBM instance:", IBM_INSTANCE)
    print_kv("IBM fractional gates:", IBM_USE_FRACTIONAL_GATES)
    print_kv("Generic basis gates:", GENERIC_BASIS_GATES)
    print_kv("Transpiler optimization:", TRANSPILER_OPTIMIZATION_LEVEL)
    print_kv(
        "Optimization meaning:",
        VALID_TRANSPILER_OPTIMIZATION_LEVELS[TRANSPILER_OPTIMIZATION_LEVEL],
    )
    print_kv("Valid optimization levels:", sorted(VALID_TRANSPILER_OPTIMIZATION_LEVELS))
    print_kv("Trotter sequence order:", TROTTER_SEQUENCE_ORDER)
    print_kv(
        "Trotter order meaning:",
        VALID_TROTTER_SEQUENCE_ORDERS[TROTTER_SEQUENCE_ORDER],
    )
    print_kv("Valid Trotter orders:", sorted(VALID_TROTTER_SEQUENCE_ORDERS))
    print_kv("Verbose printing:", VERBOSE)

    print_header("Loaded Molecule And Archive Summary")
    molecule = pauli_archive["molecule"]
    active = pauli_archive["active_space"]
    hf_ref = pauli_archive["hf_reference"]
    diagnostics = pauli_archive["diagnostics"]
    print_kv("Molecule label:", molecule.get("label"))
    print_kv("Basis:", molecule.get("basis"))
    print_kv("Geometry units:", molecule.get("geometry_units"))
    print_kv("Number of qubits:", active.get("num_qubits"))
    print_kv("Number of electrons:", active.get("num_electrons"))
    print_kv("HF occupation n_p list:", hf_ref.get("occupation_little_endian"))
    print_kv("Qiskit count bitstring:", hf_ref.get("qiskit_counts_bitstring_if_measured_q_to_c_same_index"))
    print_kv("Grouped Pauli groups:", diagnostics.get("grouped_pauli_groups"))
    print_kv("Total Pauli terms:", diagnostics.get("total_pauli_terms"))
    print_kv("Identity terms:", diagnostics.get("identity_pauli_terms"))
    print_kv("Max terms in one group:", diagnostics.get("max_pauli_terms_in_group"))

    print_header("Coefficient And Angle Convention")
    print(
        "For each Pauli term alpha * P, this script implements exp(-i dt alpha P).\n"
        "The selected keyword controls the construction path:\n"
        "  manual_ladder: explicitly basis-change, CNOT-parity-ladder, and apply\n"
        "    RZ(2 * dt * alpha), since Qiskit RZ(phi) = exp(-i phi Z / 2).\n"
        "  pauli_evolution_gate: build SparsePauliOp(labels, coeffs=alpha) and\n"
        "    append PauliEvolutionGate(operator, time=dt) with LieTrotter(reps=1).\n"
        "    A checked one-qubit Z example gives the same RZ(2 * dt * alpha) angle.\n"
        "Identity terms alpha * I become global phase. Qiskit may store that phase\n"
        "modulo 2*pi after synthesis. Coefficients are required to be real within\n"
        "COEFFICIENT_IMAG_TOLERANCE."
    )

    print_header("Target And Transpilation")
    if TARGET_MODE == "generic_simulator":
        print(
            "TARGET_MODE='generic_simulator' means no hardware coupling map is imposed.\n"
            f"Circuits are transpiled to the basis {GENERIC_BASIS_GATES} with "
            f"optimization level {TRANSPILER_OPTIMIZATION_LEVEL}."
        )
    elif TARGET_MODE == "ibm_qpu_paid":
        print(
            "TARGET_MODE='ibm_qpu_paid' loads a real IBM Quantum backend through "
            "Qiskit Runtime and transpiles against that backend's target. This "
            "script still only saves circuits; it does not submit hardware jobs."
        )
        print_kv("Resolved IBM backend:", target_backend_summary(target_backend))
    print_kv("Selected construction path:", EVOLUTION_METHOD)


def print_detailed_example(group, raw_circuit, transpiled_circuit, num_qubits):
    print_header("Detailed Example: First Nontrivial Group")
    print_kv("Group index:", group["group_index"])
    print_kv("Source classification:", group["source_classification"])
    print_kv("Pauli terms in group:", group["num_pauli_terms"])
    print_kv("Evolution method:", EVOLUTION_METHOD)
    for term in group["pauli_terms"][:DETAILED_EXAMPLE_MAX_TERMS]:
        coeff = complex_from_record(term["coefficient"])
        label = pauli_label(term["pauli_word"], num_qubits)
        print(
            f"  term {term['term_index']:>2}: "
            f"alpha={coeff.real:+.12f}{coeff.imag:+.12f}j  "
            f"P={term['pauli_word_string']:<28} qiskit_label={label}"
        )
    if group["num_pauli_terms"] > DETAILED_EXAMPLE_MAX_TERMS:
        print(f"  ... {group['num_pauli_terms'] - DETAILED_EXAMPLE_MAX_TERMS} more terms")
    print_kv("Raw depth:", raw_circuit.depth())
    print_kv("Raw ops:", op_counts(raw_circuit))
    print_kv("Transpiled depth:", transpiled_circuit.depth())
    print_kv("Transpiled ops:", op_counts(transpiled_circuit))


def main():
    validate_options()

    if not INPUT_GROUPED_PAULI_JSON.exists():
        raise FileNotFoundError(
            f"Missing grouped Pauli archive: {INPUT_GROUPED_PAULI_JSON}. "
            "Run scripts/openfermion/convert_h4_hermitian_pairs_to_openfermion.py first."
        )

    with INPUT_GROUPED_PAULI_JSON.open("r", encoding="utf-8") as handle:
        pauli_archive = json.load(handle)

    source_hash = sha256_file(INPUT_GROUPED_PAULI_JSON)
    groups = pauli_archive["groups"]
    num_qubits = int(pauli_archive["active_space"]["num_qubits"])
    target_backend = resolve_transpile_backend()

    print_startup(pauli_archive, source_hash, target_backend)

    raw_circuits = []
    transpiled_circuits = []
    group_metadata = []
    detailed_example_printed = False

    print_header("Building And Transpiling Group Circuits")
    for position, group in enumerate(groups):
        raw = build_raw_group_circuit(group, num_qubits)
        transpiled = transpile_for_target(raw, target_backend)

        raw_summary = circuit_summary(raw)
        transpiled_summary = circuit_summary(transpiled)
        record = {
            "qpy_circuit_index": int(position),
            "group_index": int(group["group_index"]),
            "source_group_index": int(group["source_group_index"]),
            "source_classification": group["source_classification"],
            "num_pauli_terms": int(group["num_pauli_terms"]),
            "num_identity_terms": int(group["num_identity_terms"]),
            "num_non_identity_terms": int(group["num_non_identity_terms"]),
            "evolution_method": EVOLUTION_METHOD,
            "pauli_evolution_synthesis": (
                PAULI_EVOLUTION_SYNTHESIS
                if EVOLUTION_METHOD == "pauli_evolution_gate"
                else None
            ),
            "raw": raw_summary,
            "transpiled": transpiled_summary,
        }
        raw_circuits.append(raw)
        transpiled_circuits.append(transpiled)
        group_metadata.append(record)

        if (
            not detailed_example_printed
            and group["source_classification"] != "zero_body_scalar"
        ):
            print_detailed_example(group, raw, transpiled, num_qubits)
            detailed_example_printed = True

        if VERBOSE and (
            position < 5
            or (position + 1) % PROGRESS_EVERY_N_GROUPS == 0
            or position == len(groups) - 1
        ):
            print(
                f"group {group['group_index']:>4} | "
                f"{group['source_classification']:<16} | "
                f"paulis {group['num_pauli_terms']:>2} | "
                f"raw depth {raw.depth():>4} | "
                f"transpiled depth {transpiled.depth():>4} | "
                f"ops {op_counts(transpiled)}"
            )

    OUTPUT_QPY.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_QPY.open("wb") as handle:
        qpy.dump(transpiled_circuits, handle)

    raw_depths = [record["raw"]["depth"] for record in group_metadata]
    transpiled_depths = [record["transpiled"]["depth"] for record in group_metadata]
    metadata = {
        "schema_version": 1,
        "generated_at_utc": now_utc(),
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "qiskit_version": getattr(qiskit, "__version__", None),
        "input_grouped_pauli_json": str(INPUT_GROUPED_PAULI_JSON),
        "input_grouped_pauli_sha256": source_hash,
        "output_qpy": str(OUTPUT_QPY),
        "output_metadata_json": str(OUTPUT_METADATA_JSON),
        "options": {
            "dt": DT,
            "evolution_method": EVOLUTION_METHOD,
            "valid_evolution_methods": sorted(VALID_EVOLUTION_METHODS),
            "pauli_evolution_synthesis": PAULI_EVOLUTION_SYNTHESIS,
            "target_mode": TARGET_MODE,
            "valid_target_modes": VALID_TARGET_MODES,
            "ibm_runtime_channel": IBM_RUNTIME_CHANNEL,
            "ibm_backend_name": IBM_BACKEND_NAME,
            "ibm_instance": IBM_INSTANCE,
            "ibm_use_fractional_gates": IBM_USE_FRACTIONAL_GATES,
            "generic_basis_gates": GENERIC_BASIS_GATES,
            "transpiler_optimization_level": TRANSPILER_OPTIMIZATION_LEVEL,
            "valid_transpiler_optimization_levels": {
                str(key): value
                for key, value in VALID_TRANSPILER_OPTIMIZATION_LEVELS.items()
            },
            "trotter_sequence_order": TROTTER_SEQUENCE_ORDER,
            "valid_trotter_sequence_orders": {
                str(key): value
                for key, value in VALID_TROTTER_SEQUENCE_ORDERS.items()
            },
            "coefficient_imag_tolerance": COEFFICIENT_IMAG_TOLERANCE,
            "include_barriers_between_pauli_terms": INCLUDE_BARRIERS_BETWEEN_PAULI_TERMS,
        },
        "target": {
            "mode": TARGET_MODE,
            "description": VALID_TARGET_MODES[TARGET_MODE],
            "backend": target_backend_summary(target_backend),
            "note": "This script transpiles and serializes circuits; it does not submit jobs.",
        },
        "conventions": {
            "unitary": "U_mu(dt) = exp(-i dt sum_l alpha_mu_l P_mu_l)",
            "selected_evolution_method": EVOLUTION_METHOD,
            "manual_ladder_pauli_rotation": "RZ(2 * dt * alpha) inside a CNOT parity ladder",
            "pauli_evolution_gate": "SparsePauliOp coefficients are alpha and PauliEvolutionGate time is dt.",
            "pauli_evolution_synthesis": "LieTrotter(reps=1) when EVOLUTION_METHOD is pauli_evolution_gate.",
            "identity_term": "alpha * I becomes global phase; Qiskit may store it modulo 2*pi.",
            "sparse_pauli_label_order": "Qiskit labels are big-endian strings; the rightmost character acts on qubit 0.",
            "basis_change": "X uses H; Y uses Sdg then H before the ladder, then H and S after it.",
        },
        "molecule": pauli_archive["molecule"],
        "active_space": pauli_archive["active_space"],
        "hf_reference": pauli_archive["hf_reference"],
        "source_archive_diagnostics": pauli_archive["diagnostics"],
        "trotter_step_sequence": trotter_sequence(groups),
        "circuit_archive": {
            "format": "qiskit.qpy",
            "contains": "transpiled group circuits in group-index order",
            "num_circuits": len(transpiled_circuits),
        },
        "depth_statistics": {
            "raw_min": min(raw_depths) if raw_depths else 0,
            "raw_max": max(raw_depths) if raw_depths else 0,
            "raw_sum": sum(raw_depths),
            "transpiled_min": min(transpiled_depths) if transpiled_depths else 0,
            "transpiled_max": max(transpiled_depths) if transpiled_depths else 0,
            "transpiled_sum": sum(transpiled_depths),
        },
        "groups": group_metadata,
        "caveats": [
            (
                "generic_simulator does not impose a hardware coupling map."
                if TARGET_MODE == "generic_simulator"
                else "ibm_qpu_paid transpiles against a hardware target but does not submit jobs."
            ),
            "The saved QPY contains full-dt group circuits; the sequence metadata records the intended product order.",
            "Non-real Pauli coefficients above tolerance are rejected rather than silently coerced.",
            "The PauliEvolutionGate path intentionally delegates low-level synthesis to Qiskit LieTrotter(reps=1).",
        ],
    }

    with OUTPUT_METADATA_JSON.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print_header("Final Output Summary")
    print_kv("Saved QPY:", OUTPUT_QPY)
    print_kv("Saved metadata JSON:", OUTPUT_METADATA_JSON)
    print_kv("Number of circuits:", len(transpiled_circuits))
    print_kv("Raw depth min/max/sum:", f"{min(raw_depths)} / {max(raw_depths)} / {sum(raw_depths)}")
    print_kv(
        "Transpiled depth min/max/sum:",
        f"{min(transpiled_depths)} / {max(transpiled_depths)} / {sum(transpiled_depths)}",
    )
    print_kv("Evolution method used:", EVOLUTION_METHOD)
    if TARGET_MODE == "generic_simulator":
        print("Caveat: generic_simulator does not encode a real IBM device topology.")
    else:
        print("Caveat: hardware target mode transpiles only; no IBM QPU job was submitted.")


if __name__ == "__main__":
    main()
