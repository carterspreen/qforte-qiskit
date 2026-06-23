# Run Order

This file records the intended human run order for the workflow. It is only a
scaffold note for now; later prompts will add the actual scripts and exact
commands.

## qforte Environment

Activate:

```bash
conda activate qfe_env_v1
```

Planned qforte-side steps:

1. Generate molecule metadata for linear H4/STO-3G.
2. Build and save the qforte Hermitian-pair Hamiltonian.
3. Convert grouped second-quantized terms to grouped Jordan-Wigner Pauli data.

## Qiskit Environment

Activate:

```bash
conda activate qiskit_env_v1
```

Planned Qiskit-side steps:

1. Build and transpile grouped time-evolution circuits.
2. Verify HF/vacuum preparation and multifidelity-estimation arithmetic.
3. Build standard UQK overlap matrices from full Trotter blocks.
4. Build stochastic UQK overlap matrices from sampled grouped blocks.
5. Exercise IBM hardware handling in dry-run mode before any real submission.
6. Validate overlap matrices against noiseless direct linear algebra.
7. Solve the projected unitary generalized eigenvalue problem.
