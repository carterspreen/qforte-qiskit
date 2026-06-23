# qforte-qiskit Codex Gameplan

This file is a draft sequence of copy-paste Codex prompts for implementing the
workflow in `notes/core_workflow.txt`. The prompts intentionally keep each step
small, readable, and reviewable. Do not run these prompts all at once. Paste one
prompt, review the resulting diff, then move to the next prompt.

Standing assumptions for every prompt:

- This repo is a research workspace, not a production package.
- Prefer simple, readable scripts over abstractions.
- Avoid long helper chains. Inline simple logic when it keeps the script easier
  to read.
- No command-line flags. User-facing options should be hard-coded near the top
  of each script.
- Every script should begin with comments giving the exact manual run command,
  a short summary, and the relevant hard-coded options.
- qforte-centric scripts should run in `qfe_env_v1`.
- Qiskit-centric scripts should run in `qiskit_env_v1`.
- Consult `notes/core_workflow.txt`, `notes/stochastic_srqk.tex`, and the local
  qforte source at `/Users/nstair/Src/my_qforte/qforte` before implementing
  math- or API-sensitive behavior.
- Keep generated data, circuits, and results out of source directories.
- Never commit IBM/Qiskit credentials or write tokens into repo files.

## Prompt 1: Create The Repository Scaffold Only

```text
You are working in `/Users/nstair/Src/qforte-qiskit`.

Read `notes/core_workflow.txt` and `notes/stochastic_srqk.tex`. Also inspect the
local qforte source at `/Users/nstair/Src/my_qforte/qforte`, especially:

- `tests/test_adapter.py`
- `tests/test_sq_pool.py`
- `tests/test_jw_transform.py`
- `src/qforte/system/system_factory.py`
- `src/qforte/adapters/molecule_adapters.py`
- `src/qforte/maths/eigsolve.py`

Create only a simple directory scaffold for this workflow. Do not implement
chemistry, OpenFermion conversion, Qiskit circuits, IBM hardware access, or
matrix element estimation yet.

The scaffold should support:

- qforte-side scripts for molecule and Hamiltonian preparation
- qiskit-side scripts for circuit construction, transpilation, execution, and
  post-processing
- small shared utilities only if they clearly improve readability
- generated molecule metadata
- generated Hamiltonian term archives
- generated transpiled circuits
- generated Krylov matrices and final summaries
- notes or docs for human-readable run order

Keep the scaffold small and obvious. Add `.gitignore` entries for generated
large or machine-local artifacts, but do not ignore the source scripts or notes.
Add a short `README.md` that explains the workflow order, the two conda
environments, and the no-flags/hard-coded-options convention.

Verify by listing the new files/directories and running `git status --short`.
Stop after the scaffold is created.
```

## Prompt 2: Generate The Linear H4 Molecule Metadata

```text
You are working in `/Users/nstair/Src/qforte-qiskit`.

Implement the first workflow bullet from `notes/core_workflow.txt`: generate a
linear H4 molecule in qforte using Psi4, STO-3G basis, and save human-readable
metadata.

Before editing, read:

- `notes/core_workflow.txt`
- the molecular setup sections of `notes/stochastic_srqk.tex`
- `/Users/nstair/Src/my_qforte/qforte/tests/test_adapter.py`
- `/Users/nstair/Src/my_qforte/qforte/src/qforte/system/molecular_info.py`
- `/Users/nstair/Src/my_qforte/qforte/src/qforte/adapters/molecule_adapters.py`

Create a qforte-side script with hard-coded options near the top. It should:

- use `qforte.system_factory` with `system_type="molecule"` and
  `build_type="psi4"`
- build linear H4 in STO-3G, neutral singlet, spacing initially matching the
  qforte test convention if appropriate
- request FCI for 8 roots if qforte/Psi4 supports this cleanly
- save JSON metadata with geometry, basis, charge, multiplicity, electron count,
  orbital/qubit counts, nuclear repulsion energy, HF energy, FCI roots, HF
  reference bitstring/occupation convention, qforte source path used, and a
  timestamp
- avoid serializing large operator objects in this step

Each script must start with comments showing the exact manual command, the conda
environment, a short summary, and the hard-coded options.

Also add a tiny verification script or test that reads the JSON and checks that
the required metadata keys exist and that the number of requested FCI roots is
recorded. If 8 FCI roots are not available, record the reason in the JSON and
in the script output instead of hiding the failure.

Run the qforte script in `qfe_env_v1` if available, then run the verification.
Report the generated file path and key energies.
```

## Prompt 3: Build And Save The qforte Hermitian-Pair Hamiltonian

```text
You are working in `/Users/nstair/Src/qforte-qiskit`.

Implement the second workflow bullet: build the Hermitian-pair form of the H4
Hamiltonian as a `qforte.SQOpPool`, then save it in a simple, inspectable file.

Before editing, read:

- `notes/core_workflow.txt`
- the Trotterized time evolution section of `notes/stochastic_srqk.tex`
- `/Users/nstair/Src/my_qforte/qforte/src/qforte/sq_op_pool.cc`
- `/Users/nstair/Src/my_qforte/qforte/tests/test_sq_pool.py`
- `/Users/nstair/Src/my_qforte/qforte/src/qforte/qkd/srqk.py` around its
  `add_hermitian_pairs` usage

Create a qforte-side script that:

- rebuilds or loads the same H4 molecule settings from Prompt 2
- accesses `mol.sq_hamiltonian`
- creates `hermitian_pairs = qforte.SQOpPool()`
- calls `hermitian_pairs.add_hermitian_pairs(1.0, mol.sq_hamiltonian)`
- writes a JSON or NPZ-plus-JSON archive containing every grouped term as:
  group index, outer coefficient, inner SQOperator terms, creation indices,
  annihilation indices, complex coefficients, and enough metadata to reconstruct
  the pool later
- clearly separates any scalar/zero-body term from nontrivial grouped terms
- records term counts and simple sanity diagnostics

Keep the file format boring and explicit. JSON is preferred unless complex
number handling becomes too clumsy; if NPZ is used, include a JSON sidecar with
schema and metadata.

Add a verification script that reloads the archive and checks:

- every group has the expected fields
- term counts are nonzero
- creation and annihilation index lists are lists of integers
- complex coefficients round-trip correctly

Run the script and verification in `qfe_env_v1`. Stop after this step.
```

## Prompt 4: Convert Hermitian-Pair Groups To OpenFermion And JW Pauli Terms

```text
You are working in `/Users/nstair/Src/qforte-qiskit`.

Implement workflow bullets 3 and 4: read the saved qforte Hermitian-pair
Hamiltonian groups, convert each grouped SQOperator to an OpenFermion
`FermionOperator`, apply Jordan-Wigner, and save grouped Pauli terms.

Before editing, read:

- `notes/core_workflow.txt`
- the Jordan-Wigner and Pauli Hamiltonian sections of `notes/stochastic_srqk.tex`
- the Hermitian-pair archive schema produced by Prompt 3

Create a qforte-side or bridge script, whichever environment has OpenFermion
installed, with hard-coded paths and options near the top. It should:

- load the Hermitian-pair archive from Prompt 3
- convert each qforte second-quantized monomial into an OpenFermion
  `FermionOperator` with the same coefficient convention
- preserve the parent Hermitian-pair group index
- apply OpenFermion Jordan-Wigner to each group
- save a grouped Pauli archive with coefficient, Pauli word, qubit count,
  source group index, source SQ terms, and metadata documenting endianness and
  qubit/orbital indexing

Do not add qforte validation of the Jordan-Wigner transform. Trust
OpenFermion's Jordan-Wigner implementation for this bridge step.

Add a verification script that reloads the grouped Pauli archive and checks:

- every group has the expected fields
- source group indices are preserved
- each Pauli term has a parseable Pauli word and complex coefficient
- qubit indices are integers in range
- scalar/identity terms are represented explicitly
- total group and Pauli-term counts are nonzero

Run the conversion and verification. Report the output archive path, number of
Hermitian-pair groups, and total Pauli strings.
```

## Prompt 5: Build And Transpile Qiskit Circuits For Each Grouped U_mu

```text
You are working in `/Users/nstair/Src/qforte-qiskit`.

Implement workflow bullets 5 and 6: read the grouped Pauli archive and create
Qiskit circuits for each grouped time-evolution factor
`U_mu = exp(-i dt h_mu H_mu)`, then transpile and save the circuits.

Before editing, read:

- `notes/core_workflow.txt`
- the Trotterized time evolution section of `notes/stochastic_srqk.tex`
- the grouped Pauli archive schema from Prompt 4
- the current Qiskit APIs installed in `qiskit_env_v1`

Create a Qiskit-side script with hard-coded options near the top:

- input grouped Pauli archive path
- output circuit archive path
- `dt`
- evolution construction keyword, such as `"manual_ladder"` or
  `"pauli_evolution_gate"`
- target mode, such as `"generic_simulator"` or a named fake/backend target
- transpiler optimization level
- whether to build first-order or second-order group sequence metadata
- verbose/explanatory printing, enabled by default

For each Hermitian-pair group:

- build the grouped factor by composing Pauli evolution subcircuits for all
  Pauli strings in that group
- preserve group boundaries in saved metadata
- keep the implementation readable; support an explicit manual CNOT-ladder
  construction and an optional `PauliEvolutionGate` construction selected by
  the hard-coded evolution keyword
- if `PauliEvolutionGate` is used, build a `SparsePauliOp` with the grouped
  Pauli coefficients, set the Qiskit evolution time to `dt`, use an explicit
  synthesis choice such as `LieTrotter(reps=1)`, and verify/document the
  coefficient sign and rotation-angle convention
- transpile each grouped circuit for the selected target
- save circuits in an appropriate Qiskit-native or stable serialized format,
  with JSON metadata for options, target, qubit count, dt, group index, depth,
  operation counts, and source archive hash/path

Make the build script intentionally explanatory when it runs. It should print
enough information to serve as a guide for what the script is doing, not just a
bare success/failure log. Include:

- a startup banner explaining that each saved circuit represents one grouped
  Hermitian-pair factor `U_mu = exp(-i dt sum_l alpha_mu_l P_mu_l)`
- the exact input/output paths and all hard-coded user-facing options
- the selected evolution construction keyword and what the alternate keyword
  would do
- the target/transpilation choice and what it means
- the loaded molecule and active-space summary, including qubit count and HF
  bitstring convention
- grouped Pauli archive diagnostics, including number of groups, total Pauli
  terms, identity terms, and max Pauli terms per group
- a short explanation of how Pauli coefficients become rotation angles or
  Qiskit evolution parameters, including the sign convention being used
- periodic per-group progress lines with group index, source classification,
  number of Pauli terms, raw depth, transpiled depth, and operation counts
- a small detailed example for the first nontrivial group, showing its Pauli
  strings, coefficients, and resulting circuit summary
- final output summary with saved files, total raw/transpiled depth statistics,
  and any caveats or assumptions

Also create an inspection script that loads the saved circuit archive and prints
a readable report. It should include the same option metadata, a compact group
table, and an optional detailed view of one representative group so a human can
understand what was built without opening the serialized circuit file manually.

Run the build and inspection in `qiskit_env_v1`. Stop after circuit generation.
```

## Prompt 6: Implement HF/Vacuum Preparation And MFE Count Estimators

```text

You are working in `/Users/nstair/Src/qforte-qiskit`.

Prepare the measurement building blocks needed for workflow bullet 7, without
yet running full Krylov matrix construction.

Before editing, read:

- `notes/core_workflow.txt`
- the Multifidelity Estimation Protocol section of
  `notes/stochastic_srqk.tex`
- the metadata and circuit archives from Prompts 2 through 5

Create a Qiskit-side script or small module that builds three circuit templates
for a supplied evolution circuit V:

- F1: prepare the HF determinant, apply V, measure all qubits, and count HF
  returns
- F2_plus: prepare `(HF + vacuum) / sqrt(2)`, apply V, unprepare with the
  matching real-superposition circuit, measure all qubits, and count HF returns
- F2_i: prepare `(HF + i vacuum) / sqrt(2)`, apply V, unprepare with the
  matching real-superposition circuit or documented phase convention, measure
  all qubits, and count HF returns

Then implement the MFE arithmetic from `stochastic_srqk.tex`:

- Re(z) = 2 F2_plus - (F1 + 1) / 2
- Im(z) = 2 F2_i - (F1 + 1) / 2, with sign convention documented
- z = Re(z) + i Im(z)

Keep credential and backend handling out of this prompt. Use local simulator
counts or fake counts to unit-test only the circuit and estimator plumbing.

Add a verification script that:

- checks the HF bitstring convention from metadata
- runs at least one identity-evolution case where z should be close to 1
- runs at least one simple phase/evolution case where Re and Im are nontrivial
- prints estimated z and expected z

Run in `qiskit_env_v1`. Stop after the MFE estimator building block works.
```

## Prompt 7: Build Standard UQK Overlap Matrix S With Full Trotter Blocks

```text
You are working in `/Users/nstair/Src/qforte-qiskit`.

Implement the standard unitary quantum Krylov overlap-matrix workflow from
bullet 7A and bullet 8. Use full Trotter blocks for each time step and MFE to
estimate the Toeplitz correlation sequence C_k.

Before editing, read:

- `notes/core_workflow.txt`
- the Unitary Quantum Krylov Method section of `notes/stochastic_srqk.tex`
- the Multifidelity Estimation Protocol section of `notes/stochastic_srqk.tex`
- the scripts and metadata from Prompts 2 through 6

Create a Qiskit-side script with hard-coded options near the top:

- input transpiled grouped circuits archive
- molecule metadata path
- Krylov dimension M
- dt
- Trotter order, initially first order unless second order is already encoded
- shots per MFE experiment
- backend mode, initially local noiseless simulator
- random seed
- output NPZ path for S and correlation values

The script should:

- build V_k as k repeated full Trotter steps for k = 0, ..., M
- estimate C_k = <HF|U^k|HF> using the MFE circuits from Prompt 6
- enforce C_0 = 1 when appropriate and record measured diagnostics separately
- assemble S_mn = C_(n-m), using C_-k = conj(C_k)
- save S as NPZ with metadata JSON sidecar
- record raw counts, fidelities, z estimates, shots, backend, circuit depths,
  and all hard-coded options

Keep the implementation explicit and readable. It is acceptable if this first
version is not optimized.

Run a small noiseless simulator case and print S, the Hermiticity error, and
condition number. Stop after S is saved.
```

## Prompt 8: Add Stochastic UQK / qDRIFT Sampled Blocks For S

```text
You are working in `/Users/nstair/Src/qforte-qiskit`.

Extend the overlap-matrix workflow from Prompt 7 to support stochastic unitary
quantum Krylov from bullet 7B. Keep the standard UQK path intact.

Before editing, read:

- `notes/core_workflow.txt`
- the Measuring Matrix Elements with Stochastic Compilation section of
  `notes/stochastic_srqk.tex`
- the grouped Hermitian-pair and grouped Pauli archives
- the standard UQK S-builder from Prompt 7

Add hard-coded user options:

- `UQK_MODE = "standard"` or `"stochastic"`
- qDRIFT segment count `N_d`
- number of stochastic circuit instances per correlation value
- random seed
- weight convention, initially `w_mu = abs(h_mu)` from the parent
  Hermitian-pair group coefficient

For stochastic mode:

- sample grouped Hermitian-pair indices according to p_mu = w_mu / sum(w)
- build each stochastic chunk from grouped circuits, preserving group boundaries
- use the qDRIFT time rescaling described in `stochastic_srqk.tex`
- run MFE on each sampled instance
- average z estimates and counts-derived diagnostics across stochastic samples
- save the same S NPZ schema as standard mode, plus stochastic sampling
  metadata and sampled group histories

Keep the distinction between standard UQK and stochastic UQK very clear in code
comments and metadata. Standard mode uses full Trotter blocks; stochastic mode
uses sampled grouped blocks. Both use MFE for S.

Run one tiny simulator case in each mode and report depths, sampled groups, S
Hermiticity error, and condition number.
```

## Prompt 9: Add Safe IBM/Qiskit Hardware Access Path

```text
You are working in `/Users/nstair/Src/qforte-qiskit`.

Add the hardware-execution path requested in workflow bullet 7, but do not run a
real QPU job unless explicitly asked after this prompt is complete.

Before editing, read:

- the current official IBM Quantum / Qiskit Runtime documentation for
  authentication, service construction, backend selection, job submission, and
  local credential storage
- the current Qiskit APIs installed in `qiskit_env_v1`
- the standard and stochastic S-builder scripts

Implement safe credential handling:

- no IBM token, CRN, instance string, hub/group/project, or account-specific
  secret may be committed to this repo
- prefer environment variables or the official user-level Qiskit credential
  store outside the repo, following current official docs
- scripts should fail with a clear message if required credentials are missing
- generated metadata may record backend name, job ID, and non-secret service
  settings, but never tokens

Add hard-coded options for:

- backend mode: local simulator, noisy simulator if available, or IBM hardware
- backend name
- maximum circuits submitted in one batch
- shots
- whether to actually submit jobs, defaulting to False

The script should be able to:

- list or validate the selected backend when credentials are present
- prepare jobs from the same MFE circuits used for simulator mode
- save job metadata and retrieval information in results
- retrieve finished job counts and feed them into the existing MFE/S assembly
  path

Add a dry-run verification that exercises backend selection and circuit
preparation without submitting a real job. Report exactly what command would
run and what environment variables or official credential setup are required.
```

## Prompt 10: Direct Linear-Algebra Validation Of S

```text
You are working in `/Users/nstair/Src/qforte-qiskit`.

Implement workflow bullet 9: compute the UQK overlap matrix S by direct
linear-algebra/statevector simulation and compare it against the MFE result for
full Trotter blocks with no noise.

Before editing, read:

- `notes/core_workflow.txt`
- the UQK and Toeplitz sections of `notes/stochastic_srqk.tex`
- the standard UQK S-builder from Prompt 7
- qforte's direct Krylov/statevector examples in
  `/Users/nstair/Src/my_qforte/qforte/src/qforte/qkd/srqk.py`

Create a Qiskit-side validation script with hard-coded options near the top:

- molecule metadata path
- grouped circuit archive path
- MFE S NPZ path
- Krylov dimension M
- dt
- Trotter order
- output comparison path

The script should:

- prepare the HF statevector
- apply the same full Trotter block sequence used by standard UQK
- compute C_k = <HF|U^k|HF> directly from statevectors
- assemble S_direct using Toeplitz symmetry
- load S_MFE from Prompt 7
- compare S_direct and S_MFE by max absolute difference, Frobenius norm, and
  Hermiticity error
- save the comparison and print a compact summary

If the MFE simulator path has finite-shot noise, allow the script to use a very
large shot count or an exact probability path, but keep the option hard-coded
and documented.
```

## Prompt 11: Build The UQK Generalized Eigenvalue Solver

```text
You are working in `/Users/nstair/Src/qforte-qiskit`.

Implement workflow bullet 10: read S and the needed shifted UQK correlation
data, solve the projected unitary generalized eigenvalue problem, and save
energy estimates.

Before editing, read:

- `notes/core_workflow.txt`
- the UQK generalized eigenproblem section of `notes/stochastic_srqk.tex`
- `/Users/nstair/Src/my_qforte/qforte/src/qforte/maths/eigsolve.py`
- `/Users/nstair/Src/my_qforte/qforte/src/qforte/abc/qsdabc.py`
- `/Users/nstair/Src/my_qforte/qforte/src/qforte/qkd/srqk.py` around
  `_solve_qk_geig` and `_qk_geig_reduced_rank`

Create a post-processing script with hard-coded options near the top:

- input S/correlation NPZ path
- molecule metadata path
- Krylov dimension to use
- dt
- overlap eigenvalue threshold, default matching qforte's canonical GEV
  threshold behavior unless we choose otherwise
- output summary JSON path

The script should:

- load C_k values through at least k = M, since U_mn = C_(n + 1 - m)
- assemble S_mn = C_(n - m)
- assemble projected U_mn = C_(n + 1 - m)
- solve U c = lambda S c using a qforte-style canonical orthogonalization:
  diagonalize S, discard eigenvalues below threshold, transform, solve the
  reduced ordinary eigenproblem, and back-transform eigenvectors
- convert phases to energies with E = -Arg(lambda) / dt, recording the modulo
  ambiguity 2*pi/dt
- sort/report roots in a documented way
- save a summary containing molecule metadata, HF energy, FCI roots, dt, M,
  threshold, overlap condition number, retained rank, lambda values, phases,
  energies, and residual diagnostics

Add a verification mode using the direct-linear-algebra S/C data from Prompt 10
when available. Print the predicted UQK energies next to HF and FCI roots.
```

## Prompt 12: End-To-End Smoke Script And Documentation Pass

```text
You are working in `/Users/nstair/Src/qforte-qiskit`.

Do a documentation and smoke-test pass over the workflow implemented by the
previous prompts. Do not add new scientific capabilities.

Read:

- `README.md`
- all scripts created in previous prompts
- `notes/core_workflow.txt`
- `notes/stochastic_srqk.tex`

Make the run order clear for a human:

- molecule metadata
- Hermitian-pair Hamiltonian
- OpenFermion/JW grouped Pauli archive
- Qiskit grouped circuits/transpilation
- MFE estimator check
- standard UQK S
- stochastic UQK S
- hardware dry run
- direct linear-algebra S validation
- UQK GEV solve

Add one optional smoke-test script or README section with exact commands for a
small local-only run. Keep commands separated by conda environment. Do not use
flags. Do not run hardware jobs.

Then run the local smoke checks that are reasonable on this machine and report:

- commands run
- generated files
- pass/fail status
- any known limitations or next implementation questions
```
