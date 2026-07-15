# Resubmitting Remaining `Nd=8` Hardware Jobs

This note is for completing the current production hardware run after the first
retrieval assembled the complete points and skipped the incomplete `Nd=8` point.

## Current State

- Run ID:
  `diatomic_h2_sto_3g_uqk_hardware_preliminary_production_run_M_4_kmax_3_seed_330623`
- Original backend:
  `ibm_miami`
- Production plan:
  `hardware_runs/runs/<run_id>/plan/`
- Original submit manifest:
  `hardware_runs/runs/<run_id>/submit/submitted_jobs.json`
- Current retrieval summary:
  `hardware_runs/runs/<run_id>/retrieve/retrieval_summary.json`
- Current partial assembly:
  `hardware_runs/runs/<run_id>/assembled/hardware_overlap_records.json`

The following points have already been assembled:

- `standard_Nmfe_10000`
- `stochastic_Nd_1_Nw_500_Nmfe_200`
- `stochastic_Nd_2_Nw_500_Nmfe_200`
- `stochastic_Nd_4_Nw_500_Nmfe_200`

The remaining incomplete point is:

- `stochastic_Nd_8_Nw_500_Nmfe_200`

The five missing original job groups are:

| Job group | Original job ID | Last known status |
|---|---|---|
| `stochastic_Nd_8_Nw_500_Nmfe_200_chunk_4` | `d95sbqd2su3c739g91i0` | `QUEUED` |
| `stochastic_Nd_8_Nw_500_Nmfe_200_chunk_5` | `d95sbrkqp3as739ptlr0` | `QUEUED` |
| `stochastic_Nd_8_Nw_500_Nmfe_200_chunk_6` | `d95sbskqp3as739ptls0` | `QUEUED` |
| `stochastic_Nd_8_Nw_500_Nmfe_200_chunk_7` | `d95sbtsqp3as739ptlu0` | `QUEUED` |
| `stochastic_Nd_8_Nw_500_Nmfe_200_chunk_8` | `d95sbuotcv6s73dj3bf0` | `QUEUED` |

## First Try: Retrieval, Not Resubmission

Before resubmitting anything, rerun retrieval:

```bash
conda run -n qiskit_env_v1 python hardware_runs/3_retrieve.py production_run
```

Then inspect:

```bash
jq '[.jobs[] | select(.result_ready != true) | {job_group_id, job_id, status_at_retrieve}]' \
  hardware_runs/runs/diatomic_h2_sto_3g_uqk_hardware_preliminary_production_run_M_4_kmax_3_seed_330623/retrieve/retrieval_summary.json
```

If all jobs are done, run:

```bash
conda run -n qiskit_env_v1 python hardware_runs/4_assemble.py production_run
```

That will assemble the full `Nd=8` point.

## Do Not Do This

Do not simply set `ALLOW_RESUBMIT = True` in `hardware_runs/2_submit.py` and
rerun `2_submit.py production_run`.

That script currently submits the whole production plan again, not just the five
missing chunks. It would also overwrite or confuse the original submit manifest
unless additional resubmission handling is added.

Also do not submit the existing `hardware_circuits.qpy` to a different backend.
That QPY was transpiled as ISA circuits for the original backend.

## If Resubmission Is Needed On The Same Backend

Ask Codex to add a small targeted resubmission workflow. The script should:

1. Read the existing production `plan_metadata.json`.
2. Read the existing `retrieval_summary.json`.
3. Identify only job groups with `result_ready != true`.
4. Load the existing production QPY.
5. Submit only those missing `circuit_indices`.
6. Save a separate manifest, for example:
   `submit/resubmitted_missing_nd8_<timestamp>.json`.
7. Never overwrite `submit/submitted_jobs.json`.
8. Record `supersedes_job_id` for each original queued job.

Before resubmitting, decide whether to cancel the old queued jobs in IBM
Quantum. If the old jobs are not cancelled, they may eventually run too, which
could spend additional budget. Do not cancel jobs unless the PI/user explicitly
approves it.

After targeted resubmission, retrieval also needs to know about the new job IDs.
Either:

- update `3_retrieve.py` to accept an additional manifest path, or
- create a merge step that combines original retrieved jobs plus replacement
  jobs into one retrieval summary for assembly.

## If Resubmitting To A New Backend

Changing backend requires extra care.

Required rule:

- Replan/transpile for the new backend before submitting.

Recommended workflow:

1. Edit only `hardware_runs/secrets/ibm_quantum_runtime.json` to change
   `backend_name`.
2. Do not commit or paste that file anywhere.
3. Generate a new runtime-backed plan under a new run ID, for example:

```bash
QFORTE_IBM_BACKEND_SOURCE=runtime \
conda run -n qiskit_env_v1 python hardware_runs/1_plan_only.py \
  production_run \
  diatomic_h2_sto_3g_uqk_hardware_preliminary_production_run_nd8_resubmit_<backend>_M_4_kmax_3_seed_330623
```

4. Use the new QPY only for the missing chunk circuit indices from the original
   `Nd=8` point.
5. Save a resubmission manifest that records:
   - original run ID
   - new run ID
   - original backend
   - replacement backend
   - original job IDs being replaced
   - replacement job IDs
   - original circuit indices

Important caveat:

- A mixed-backend `Nd=8` matrix is scientifically different from a single-backend
  `Nd=8` matrix. If chunks 0-3 come from `ibm_miami` and chunks 4-8 come from
  another backend, label the assembled result as mixed-backend hardware data.

## Final Assembly Check

After all replacement jobs are retrieved, run:

```bash
conda run -n qiskit_env_v1 python hardware_runs/4_assemble.py production_run
```

Confirm:

```bash
jq '{retrieval_complete, pending_job_count, assembled_point_count, skipped_points}' \
  hardware_runs/runs/diatomic_h2_sto_3g_uqk_hardware_preliminary_production_run_M_4_kmax_3_seed_330623/assembled/hardware_overlap_records.json
```

Expected final state:

- `retrieval_complete: true`
- `pending_job_count: 0`
- `assembled_point_count: 5`
- `skipped_points: []`

