# Hardware UQK Run Todo

This directory is script-driven on purpose. The plan stage can be run many times
locally; the submit stage is the only stage that can spend IBM Runtime budget,
and it has an explicit double opt-in gate.

## Directory Contract

- `hardware_runs/1_plan_only.py`: build and transpile all MFE circuits, then save a QPY and JSON plan.
- `hardware_runs/2_submit.py`: submit the saved plan to IBM Runtime Sampler.
- `hardware_runs/3_retrieve.py`: retrieve finished jobs by saved job ID and write raw counts.
- `hardware_runs/4_assemble.py`: convert raw counts into UQK overlap matrices and hardware records.
- `hardware_runs/secrets/ibm_quantum_runtime.json`: your local secret credential file. It is gitignored.
- `hardware_runs/runs/<run_id>/`: generated plans, job IDs, retrieved counts, and assembled matrices. This is gitignored.

## User: IBM Cloud Setup

1. Log in to IBM Quantum Platform.
2. Select the account and region that owns the Runtime minutes.
3. Create or choose the IBM Quantum service instance.
4. Copy the IBM Cloud API key from the dashboard.
5. Copy the service instance CRN from the Instances page.
6. Create `hardware_runs/secrets/ibm_quantum_runtime.json` by copying `hardware_runs/secrets/ibm_quantum_runtime.example.json`.
7. Paste only these values into the real JSON file:

```json
{
  "account_name": "qforte-hardware",
  "token": "<IBM Cloud API key>",
  "instance": "<IBM Quantum instance CRN>",
  "channel": "ibm_quantum_platform",
  "backend_name": "ibm_brisbane"
}
```

Do not paste credentials into notebooks, source files, logs, or chat. The real JSON file is ignored by git.

## Codex: Local Planning Check

Run a local fake-backend plan first. This does not touch IBM Cloud and cannot be submitted.

```bash
conda activate qiskit_env_v1
python hardware_runs/1_plan_only.py dry_run
```

Expected dry-run defaults:

- `standard`: `Nmfe = 1024`
- `stochastic`: `Nd = 1`, `Nw = 5`, `Nmfe = 512`
- `KRYLOV_DIMENSION = 4`
- measured powers: `k = 1, 2, 3`
- `C0` enforced exactly during assembly

Inspect:

- `hardware_runs/runs/diatomic_h2_sto_3g_uqk_hardware_preliminary_dry_run_M_4_kmax_3_seed_330623/plan/plan_metadata.json`

## Codex: Runtime-Backed Plan

After the credential JSON exists, generate a plan against the real backend target.
This contacts IBM Runtime to get backend information, but still does not submit QPU work.

```bash
QFORTE_IBM_BACKEND_SOURCE=runtime python hardware_runs/1_plan_only.py dry_run
```

Inspect the plan metadata before submitting:

- backend source must be `runtime`
- circuit count and total executions should match expectations
- ISA circuit depths and operation counts should look reasonable
- no token or CRN should appear in metadata

## Codex/User: Dry-Run Submission Gate

Before submitting, edit `hardware_runs/2_submit.py`:

```python
SUBMIT_TO_QPU = True
CONFIRMATION_TEXT = "SUBMIT_QFORTE_HARDWARE"
```

Then submit the dry run:

```bash
python hardware_runs/2_submit.py dry_run
```

Immediately inspect:

- `hardware_runs/runs/<run_id>/submit/submitted_jobs.json`

It must contain job IDs. Once those IDs are saved, the launching Python process does not need to remain alive.

## Codex: Retrieve Dry-Run Results

Run retrieval whenever jobs have finished. It is safe to run repeatedly.

```bash
python hardware_runs/3_retrieve.py dry_run
```

Inspect:

- `hardware_runs/runs/<run_id>/retrieve/retrieval_summary.json`
- `hardware_runs/runs/<run_id>/retrieve/jobs/*.json`

If not all jobs are ready, wait and rerun the same command.

## Codex: Assemble Dry-Run Matrices

```bash
python hardware_runs/4_assemble.py dry_run
```

Inspect:

- `hardware_runs/runs/<run_id>/assembled/hardware_overlap_records.json`
- `hardware_runs/runs/<run_id>/assembled/hardware_overlap_records.csv`
- `hardware_runs/runs/<run_id>/assembled/<point_id>/*_hardware_uqk_overlap_matrix.npz`

The assembler also tries to compare against the notebook 03 `S_reference` if it exists.

## Production Run

Only proceed after dry-run retrieval and assembly succeed.

Production defaults:

- `standard`: `Nmfe = 10_000`
- `stochastic`: `Nd = 1, 2, 4, 8`
- `stochastic`: `Nw = 500`
- `stochastic`: `Nmfe = 200`
- measured powers: `k = 1, 2, 3`
- transpilation optimization level: `3`

Generate the production plan against Runtime:

```bash
QFORTE_IBM_BACKEND_SOURCE=runtime python hardware_runs/1_plan_only.py production_run
```

Review the plan metadata carefully. Then submit:

```bash
python hardware_runs/2_submit.py production_run
```

Retrieve and assemble:

```bash
python hardware_runs/3_retrieve.py production_run
python hardware_runs/4_assemble.py production_run
```

## Safety Checks Before Any Submit

- `hardware_runs/secrets/ibm_quantum_runtime.json` exists locally and is ignored by git.
- `git status --short` does not show the real credential JSON.
- The plan metadata says `backend_source = runtime`.
- The submit manifest path does not already exist unless intentional resubmission is enabled.
- The expected total executions fit IBM Sampler job limits.
- The expected total Runtime cost is acceptable relative to the 400 minute budget.

## Pitfalls To Watch

- A fake-backend plan is useful for local validation but must not be submitted.
- Real backend calibration can change between planning and execution.
- The hardware transpiler can increase circuit width, depth, and two-qubit gate count.
- Hardware noise may favor smaller `Nd` than the noiseless notebook 03 curves.
- Retrieval should be repeated until all job IDs have finished.
- Counts are stored locally only after retrieval; job IDs are the durable handle to cloud results.
- The plotting notebook should read assembled `hardware_overlap_records.json`, not raw job files.
