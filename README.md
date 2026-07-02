# qforte-qiskit

This repository is a research workspace for IBM hardware-oriented unitary
quantum Krylov calculations using molecular Hamiltonians prepared by qforte.
It is intentionally not a production package. Scripts should stay small,
direct, and easy to read.

## Working Style

- Put user-facing options as hard-coded variables near the top of each script.
- Do not add command-line flags.
- Start each script with comments showing the exact manual command, the conda
  environment, a short summary, and the relevant hard-coded options.
- Prefer straightforward inline logic over many small helper functions.
- Keep generated data, transpiled circuits, matrices, and hardware job metadata
  out of source directories.
- Never store IBM Quantum tokens or account secrets in this repository.

## Environments

Use the existing conda environments for the two halves of the workflow:

```bash
conda activate qfe_env_v1
```

Use this for qforte-centric preparation work: molecule generation,
Hermitian-pair Hamiltonian construction, and qforte/OpenFermion conversion if
that environment has the needed dependency.

```bash
conda activate qiskit_env_v1
```

Use this for Qiskit-centric work: grouped circuit construction, transpilation,
simulator or hardware execution, multifidelity estimation, Krylov matrix
assembly, direct validation, and post-processing.

## Directory Layout

- `notes/`: source notes and derivations that guide implementation.
- `scripts/qforte/`: qforte-side scripts for molecule and Hamiltonian data.
- `scripts/qiskit/`: Qiskit-side scripts for circuits, execution, and analysis.
- `scripts/shared/`: small shared helpers, only when they make scripts clearer.
- `data/molecules/`: generated molecule metadata.
- `data/hamiltonians/`: generated Hermitian-pair and Pauli Hamiltonian archives.
- `circuits/transpiled/`: generated Qiskit circuit archives.
- `results/krylov_matrices/`: generated overlap/correlation matrix data.
- `results/summaries/`: generated energy summaries and comparisons.
- `results/hardware_jobs/`: non-secret hardware job metadata and retrieved counts.
- `hardware_runs/`: script-driven IBM hardware UQK planning, submission,
  retrieval, and assembly workflow.
- `docs/`: human-readable workflow notes.

## Workflow Order

1. Build linear H4/STO-3G molecule metadata with qforte/Psi4.
2. Build the qforte Hermitian-pair Hamiltonian as an `SQOpPool`.
3. Convert Hermitian-pair groups to OpenFermion and Jordan-Wigner Pauli terms.
4. Build and transpile Qiskit circuits for each grouped time-evolution factor.
5. Build the multifidelity-estimation circuit/counting primitives.
6. Assemble the standard UQK overlap matrix using full Trotter blocks and finite-shot MFE.
7. Assemble the exact-trotter UQK overlap matrix using full Trotter blocks and exact MFE probabilities.
8. Assemble the stochastic UQK overlap matrix using sampled grouped blocks and finite-shot MFE.
9. Assemble the exact-stochastic UQK overlap matrix using sampled grouped blocks and exact MFE probabilities.
10. Add a safe IBM hardware dry-run/submission path.
11. Validate overlap matrices by direct noiseless linear algebra.
12. Solve the UQK generalized eigenvalue problem and summarize energies.

Notebook 02 includes `Nd_<QDRIFT_SEGMENT_COUNT_ND>_sipc_<STOCHASTIC_INSTANCES_PER_CORRELATION>`
in stochastic and exact-stochastic output labels so sweeps do not overwrite
different qDRIFT runtime settings.

Notebook 03 runs the UQK overlap hyperparameter sweeps against a clearly named
`S_reference` matrix and writes cached runs, data, plots, tables, and logs under
`notebooks/<molecule>/experiments/uqk_overlap_hyperparameters/`.

See `gameplan.md` for the copy-paste implementation prompts.
