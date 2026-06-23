# Manual run:
#   conda activate qiskit_env_v1
#   python scripts/qiskit/verify_mfe_measurement_building_blocks.py
#
# Summary:
#   Reusable Qiskit helpers for the Cortes-Gray-style multifidelity estimation
#   (MFE) measurement circuits used by the qforte-qiskit Krylov workflow. Given
#   a no-classical-register evolution circuit V, build the three measurement
#   templates F1, F2_plus, and F2_i, then convert HF-return frequencies into
#   z = <HF|V|HF>.
#
# Hard-coded convention:
#   Qubit q[p] represents spin orbital p. Measuring q[i] into c[i] gives Qiskit
#   count strings printed as c[n-1]...c[0], so the HF count key is the reversed
#   little-endian occupation list from metadata.
#
# Hard-coded explanatory output options:
#   VERBOSE = True
#   PRINT_TEMPLATE_CIRCUIT_DRAWINGS = False

from __future__ import annotations

from dataclasses import dataclass

from qiskit import QuantumCircuit


F1_LABEL = "F1"
F2_PLUS_LABEL = "F2_plus"
F2_I_LABEL = "F2_i"
VERBOSE = True
PRINT_TEMPLATE_CIRCUIT_DRAWINGS = False
TEMPLATE_DRAWING_FOLD = 120


@dataclass(frozen=True)
class MFEFidelities:
    f1: float
    f2_plus: float
    f2_i: float


@dataclass(frozen=True)
class MFEEstimate:
    f1: float
    f2_plus: float
    f2_i: float
    real: float
    imag: float
    z: complex


def resolve_verbose(verbose):
    return VERBOSE if verbose is None else bool(verbose)


def print_header(title, verbose=None):
    if not resolve_verbose(verbose):
        return
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def print_kv(label, value, verbose=None):
    if resolve_verbose(verbose):
        print(f"{label:<34} {value}")


def circuit_operation_counts(circuit):
    return {name: int(count) for name, count in circuit.count_ops().items()}


def print_template_summary(label, circuit, verbose=None):
    if not resolve_verbose(verbose):
        return
    print_kv(f"{label} circuit name:", circuit.name, verbose)
    print_kv(f"{label} qubits/classical bits:", f"{circuit.num_qubits} / {circuit.num_clbits}", verbose)
    print_kv(f"{label} depth:", circuit.depth(), verbose)
    print_kv(f"{label} operation counts:", circuit_operation_counts(circuit), verbose)
    if PRINT_TEMPLATE_CIRCUIT_DRAWINGS:
        print(f"\n{label} circuit drawing:")
        print(circuit.draw(output="text", fold=TEMPLATE_DRAWING_FOLD))


def print_mfe_context(evolution_circuit, occupation, verbose=None):
    if not resolve_verbose(verbose):
        return
    occupied_qubits = occupied_qubits_from_occupation(occupation)
    hf_key = qiskit_hf_count_key_from_occupation(occupation)
    print_header("MFE Template Builder Context", verbose)
    print(
        "Goal: estimate z = <HF|V|HF> from three HF-return sampling circuits.\n"
        "The reference state R is the vacuum |0...0>. The formulas used here are\n"
        "  Re(z) = 2 F2_plus - (F1 + 1) / 2\n"
        "  Im(z) = 2 F2_i    - (F1 + 1) / 2\n"
        "with the F2_i convention preparing (|HF> + i |vac>) / sqrt(2)."
    )
    print_kv("Number of qubits:", len(occupation), verbose)
    print_kv("HF occupation n_p:", occupation, verbose)
    print_kv("Occupied HF qubits:", occupied_qubits, verbose)
    print_kv("HF count key:", hf_key, verbose)
    print_kv("Evolution circuit V name:", evolution_circuit.name, verbose)
    print_kv("Evolution circuit V depth:", evolution_circuit.depth(), verbose)
    print_kv("Evolution circuit V ops:", circuit_operation_counts(evolution_circuit), verbose)


def print_stage_explanation(label, lines, verbose=None):
    if not resolve_verbose(verbose):
        return
    print_header(f"Building {label}", verbose)
    for line in lines:
        print(f"- {line}")


def occupation_from_metadata(metadata):
    return [int(bit) for bit in metadata["hf_reference"]["occupation_little_endian"]]


def occupied_qubits_from_occupation(occupation):
    return [index for index, bit in enumerate(occupation) if int(bit) == 1]


def qiskit_hf_count_key_from_occupation(occupation):
    return "".join(str(int(bit)) for bit in reversed(occupation))


def qiskit_hf_count_key_from_metadata(metadata):
    return qiskit_hf_count_key_from_occupation(occupation_from_metadata(metadata))


def metadata_hf_count_key(metadata):
    return metadata["hf_reference"]["qiskit_counts_bitstring_if_measured_q_to_c_same_index"]


def validate_hf_metadata(metadata):
    occupation = occupation_from_metadata(metadata)
    derived_key = qiskit_hf_count_key_from_occupation(occupation)
    recorded_key = metadata_hf_count_key(metadata)
    if derived_key != recorded_key:
        raise ValueError(
            "HF bitstring convention mismatch: derived "
            f"{derived_key!r} from occupation {occupation}, but metadata records "
            f"{recorded_key!r}."
        )
    return occupation, derived_key


def validate_evolution_circuit(evolution_circuit, num_qubits):
    if evolution_circuit.num_qubits != num_qubits:
        raise ValueError(
            f"Evolution circuit has {evolution_circuit.num_qubits} qubits, "
            f"but metadata expects {num_qubits}."
        )
    if evolution_circuit.num_clbits:
        raise ValueError(
            "Evolution circuit V should not contain classical bits or measurements."
        )


def append_hf_preparation(circuit, occupied_qubits):
    for qubit in occupied_qubits:
        circuit.x(qubit)


def append_vacuum_to_hf_superposition(circuit, occupied_qubits, phase_label):
    """Prepare HF-vacuum superpositions from the all-zero vacuum state.

    phase_label="plus" prepares (|HF> + |vac>) / sqrt(2).

    phase_label="i" prepares (|HF> + i |vac>) / sqrt(2), up to a global phase.
    The relative phase is created by H then Sdg on the pivot qubit before the
    CNOT fanout. With this convention, stochastic_srqk.tex gives
    Im(z) = 2 F2_i - (F1 + 1) / 2. If the opposite superposition
    (|HF> - i |vac>) / sqrt(2) is used later, the imaginary estimator changes
    sign.
    """

    if not occupied_qubits:
        raise ValueError("HF-vacuum MFE templates require a non-vacuum HF state.")
    if phase_label not in {"plus", "i"}:
        raise ValueError("phase_label must be 'plus' or 'i'.")

    pivot = occupied_qubits[0]
    circuit.h(pivot)
    if phase_label == "i":
        circuit.sdg(pivot)
    for qubit in occupied_qubits[1:]:
        circuit.cx(pivot, qubit)


def append_real_superposition_unprepare_to_hf(circuit, occupied_qubits):
    """Apply B_plus^dagger and map the return state to the HF bitstring.

    The direct preparation helper starts from |vac> instead of explicitly
    preparing |HF> and applying B_plus. This unprepare is equivalent to
    B_plus^dagger in the notes: first undo the real HF-vacuum superposition,
    then flip the occupied HF qubits so that the successful return is measured
    as the usual HF count key.
    """

    if not occupied_qubits:
        raise ValueError("HF-vacuum MFE templates require a non-vacuum HF state.")

    pivot = occupied_qubits[0]
    for qubit in reversed(occupied_qubits[1:]):
        circuit.cx(pivot, qubit)
    circuit.h(pivot)
    append_hf_preparation(circuit, occupied_qubits)


def append_measure_all(circuit):
    circuit.measure(range(circuit.num_qubits), range(circuit.num_qubits))


def build_f1_template(evolution_circuit, occupation, verbose=None):
    num_qubits = len(occupation)
    occupied_qubits = occupied_qubits_from_occupation(occupation)
    validate_evolution_circuit(evolution_circuit, num_qubits)

    print_stage_explanation(
        F1_LABEL,
        [
            "Start from the computational vacuum |0...0>.",
            f"Flip occupied HF qubits {occupied_qubits} to prepare |HF>.",
            "Apply the supplied evolution circuit V.",
            "Measure every qubit into the same-index classical bit.",
            "The estimated F1 is the fraction of shots returning the HF count key.",
        ],
        verbose,
    )

    circuit = QuantumCircuit(num_qubits, num_qubits, name="mfe_F1_hf_return")
    append_hf_preparation(circuit, occupied_qubits)
    circuit.compose(evolution_circuit, inplace=True)
    append_measure_all(circuit)
    print_template_summary(F1_LABEL, circuit, verbose)
    return circuit


def build_f2_plus_template(evolution_circuit, occupation, verbose=None):
    num_qubits = len(occupation)
    occupied_qubits = occupied_qubits_from_occupation(occupation)
    validate_evolution_circuit(evolution_circuit, num_qubits)

    print_stage_explanation(
        F2_PLUS_LABEL,
        [
            "Prepare (|HF> + |vac>) / sqrt(2) from the vacuum using one pivot qubit and CNOT fanout.",
            "Apply the supplied evolution circuit V.",
            "Apply the real-superposition unprepare, equivalent to B_plus^dagger.",
            "Flip occupied qubits so the successful return is measured as the usual HF key.",
            "The estimated F2_plus recovers the real quadrature through the MFE formula.",
        ],
        verbose,
    )

    circuit = QuantumCircuit(num_qubits, num_qubits, name="mfe_F2_plus_hf_return")
    append_vacuum_to_hf_superposition(circuit, occupied_qubits, "plus")
    circuit.compose(evolution_circuit, inplace=True)
    append_real_superposition_unprepare_to_hf(circuit, occupied_qubits)
    append_measure_all(circuit)
    print_template_summary(F2_PLUS_LABEL, circuit, verbose)
    return circuit


def build_f2_i_template(evolution_circuit, occupation, verbose=None):
    num_qubits = len(occupation)
    occupied_qubits = occupied_qubits_from_occupation(occupation)
    validate_evolution_circuit(evolution_circuit, num_qubits)

    print_stage_explanation(
        F2_I_LABEL,
        [
            "Prepare (|HF> + i |vac>) / sqrt(2), up to global phase.",
            "This implementation uses H then Sdg on the pivot qubit before CNOT fanout.",
            "Apply V, then unprepare with the same real B_plus^dagger used by F2_plus.",
            "With this convention, Im(z) = 2 F2_i - (F1 + 1) / 2.",
            "Using (|HF> - i |vac>) instead would flip the final imaginary sign.",
        ],
        verbose,
    )

    circuit = QuantumCircuit(num_qubits, num_qubits, name="mfe_F2_i_hf_return")
    append_vacuum_to_hf_superposition(circuit, occupied_qubits, "i")
    circuit.compose(evolution_circuit, inplace=True)
    append_real_superposition_unprepare_to_hf(circuit, occupied_qubits)
    append_measure_all(circuit)
    print_template_summary(F2_I_LABEL, circuit, verbose)
    return circuit


def build_mfe_templates(evolution_circuit, occupation, verbose=None):
    print_mfe_context(evolution_circuit, occupation, verbose)
    templates = {
        F1_LABEL: build_f1_template(evolution_circuit, occupation, verbose),
        F2_PLUS_LABEL: build_f2_plus_template(evolution_circuit, occupation, verbose),
        F2_I_LABEL: build_f2_i_template(evolution_circuit, occupation, verbose),
    }
    print_header("MFE Template Build Complete", verbose)
    if resolve_verbose(verbose):
        print(
            "The returned dictionary contains F1, F2_plus, and F2_i circuits. "
            "Run each circuit, count the HF bitstring, then pass the counts to "
            "estimate_z_from_counts()."
        )
    return templates


def hf_return_frequency(counts, hf_count_key):
    total_shots = sum(int(value) for value in counts.values())
    if total_shots <= 0:
        raise ValueError("Cannot estimate a fidelity from zero shots.")
    hf_returns = int(counts.get(hf_count_key, 0))
    return hf_returns / total_shots


def fidelities_from_counts(counts_by_label, hf_count_key, verbose=None):
    fidelities = MFEFidelities(
        f1=hf_return_frequency(counts_by_label[F1_LABEL], hf_count_key),
        f2_plus=hf_return_frequency(counts_by_label[F2_PLUS_LABEL], hf_count_key),
        f2_i=hf_return_frequency(counts_by_label[F2_I_LABEL], hf_count_key),
    )
    print_header("MFE Count Fractions", verbose)
    print_kv("HF count key:", hf_count_key, verbose)
    print_kv("F1 = HF returns / shots:", f"{fidelities.f1:.10f}", verbose)
    print_kv("F2_plus = HF returns / shots:", f"{fidelities.f2_plus:.10f}", verbose)
    print_kv("F2_i = HF returns / shots:", f"{fidelities.f2_i:.10f}", verbose)
    return fidelities


def estimate_z_from_fidelities(fidelities, verbose=None):
    real = 2.0 * fidelities.f2_plus - 0.5 * (fidelities.f1 + 1.0)
    imag = 2.0 * fidelities.f2_i - 0.5 * (fidelities.f1 + 1.0)
    estimate = MFEEstimate(
        f1=fidelities.f1,
        f2_plus=fidelities.f2_plus,
        f2_i=fidelities.f2_i,
        real=real,
        imag=imag,
        z=complex(real, imag),
    )
    print_header("MFE Arithmetic", verbose)
    if resolve_verbose(verbose):
        print("Using the sign convention from stochastic_srqk.tex:")
        print("  Re(z) = 2 F2_plus - (F1 + 1) / 2")
        print("  Im(z) = 2 F2_i    - (F1 + 1) / 2")
    print_kv("Re(z):", f"{estimate.real:+.10f}", verbose)
    print_kv("Im(z):", f"{estimate.imag:+.10f}", verbose)
    print_kv("z:", f"{estimate.z.real:+.10f}{estimate.z.imag:+.10f}j", verbose)
    return estimate


def estimate_z_from_counts(counts_by_label, hf_count_key, verbose=None):
    return estimate_z_from_fidelities(
        fidelities_from_counts(counts_by_label, hf_count_key, verbose),
        verbose,
    )
