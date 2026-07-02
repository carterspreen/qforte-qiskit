"""Helpers for notebook 3 UQK overlap hyperparameter sweeps.

The notebook should read like an experiment plan. This module keeps the
mechanical pieces here: cached UQK matrix generation, reference handling,
Frobenius-error calculation, CSV/JSON writing, and plotting.
"""

from __future__ import annotations

import contextlib
import csv
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .qiskit_uqk_helpers import (
    assert_manifest_molecule_label,
    load_json,
    load_module_from_path,
    print_header,
    print_kv,
    repo_root_from_helper,
    save_json,
    workflow_file,
)


@dataclass(frozen=True)
class ExperimentContext:
    notebooks_root: Path
    molecule_name: str
    manifest_path: Path
    manifest: dict
    experiment_root: Path
    results_dir: Path
    reference_dir: Path
    data_dir: Path
    plots_dir: Path
    tables_dir: Path
    logs_dir: Path
    metadata_dir: Path
    dt: float


def now_utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_label_part(value):
    return (
        str(value)
        .replace(".", "p")
        .replace("-", "m")
        .replace("/", "_")
        .replace(" ", "_")
    )


def sweep_label(context, backend_mode, krylov_dimension):
    return (
        f"backend_{safe_label_part(backend_mode)}"
        f"_M_{int(krylov_dimension)}"
        f"_dt_{safe_label_part(context.dt)}"
    )


def current_s_reference_metadata(context):
    info_path = context.data_dir / "S_reference_info.json"
    metadata = {"S_reference_info_json": str(info_path)}
    if info_path.exists():
        metadata["S_reference"] = load_json(info_path)
    return metadata


def resolve_run_jobs(run_jobs, RUN_JOBS):
    if RUN_JOBS is not None:
        return bool(RUN_JOBS)
    return bool(run_jobs)


def matrix_error_metrics(reference, candidate):
    diff = np.asarray(reference) - np.asarray(candidate)
    reference_norm = float(np.linalg.norm(reference))
    absolute_fro = float(np.linalg.norm(diff))
    relative_fro = (
        float(absolute_fro / reference_norm)
        if reference_norm > 0.0
        else float("nan")
    )
    return {
        "absolute_frobenius_error": absolute_fro,
        "relative_frobenius_error": relative_fro,
        "max_abs_entry_error": float(np.max(np.abs(diff))),
        "reference_frobenius_norm": reference_norm,
    }


def prepare_experiment_context(
    notebooks_root,
    molecule_name="diatomic_h2_sto_3g",
    experiment_name="uqk_overlap_hyperparameters",
):
    notebooks_root = Path(notebooks_root).resolve()
    manifest_path = (
        notebooks_root / molecule_name / "metadata" / "workflow_manifest.json"
    )
    manifest = load_json(manifest_path)
    experiment_root = notebooks_root / molecule_name / "experiments" / experiment_name
    paths = {
        "results_dir": experiment_root / "results",
        "reference_dir": experiment_root / "S_reference",
        "data_dir": experiment_root / "data",
        "plots_dir": experiment_root / "plots",
        "tables_dir": experiment_root / "tables",
        "logs_dir": experiment_root / "logs",
        "metadata_dir": experiment_root / "metadata",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    circuit_metadata = load_json(workflow_file(manifest, "grouped_evolution_metadata_json"))
    dt = float(circuit_metadata["options"]["dt"])

    context = ExperimentContext(
        notebooks_root=notebooks_root,
        molecule_name=molecule_name,
        manifest_path=manifest_path,
        manifest=manifest,
        experiment_root=experiment_root,
        dt=dt,
        **paths,
    )
    save_json(
        context.metadata_dir / "experiment_context.json",
        {
            "generated_at_utc": now_utc(),
            "molecule_name": molecule_name,
            "workflow_manifest": str(manifest_path),
            "experiment_root": str(experiment_root),
            "dt": dt,
            "directories": {key: str(value) for key, value in paths.items()},
        },
    )
    return context


def _configure_builder_module(module, context, output_dir, options):
    manifest = context.manifest
    module.INPUT_QPY = workflow_file(manifest, "grouped_evolution_qpy")
    module.INPUT_CIRCUIT_METADATA_JSON = workflow_file(
        manifest, "grouped_evolution_metadata_json"
    )
    module.INPUT_MOLECULE_METADATA_JSON = workflow_file(manifest, "molecule_metadata_json")
    module.INPUT_HERMITIAN_PAIR_JSON = workflow_file(manifest, "hermitian_pairs_json")
    module.INPUT_GROUPED_PAULI_JSON = workflow_file(manifest, "grouped_paulis_json")
    module.OUTPUT_DIR = Path(output_dir)

    for check_context, path in [
        ("Molecule metadata", module.INPUT_MOLECULE_METADATA_JSON),
        ("Hermitian-pair metadata", module.INPUT_HERMITIAN_PAIR_JSON),
        ("Grouped-Pauli metadata", module.INPUT_GROUPED_PAULI_JSON),
        ("Grouped-evolution metadata", module.INPUT_CIRCUIT_METADATA_JSON),
    ]:
        assert_manifest_molecule_label(manifest, path, check_context)

    module.UQK_MODE = options["uqk_mode"]
    module.KRYLOV_DIMENSION = int(options["krylov_dimension"])
    module.MAX_CORRELATION_POWER = int(options["max_correlation_power"])
    module.SHOTS_PER_MFE_EXPERIMENT = int(options["shots_per_mfe_experiment"])
    module.BACKEND_MODE = options["backend_mode"]
    module.OUTPUT_FILE_STEM_PREFIX = ""
    module.OUTPUT_LABEL_OVERRIDE = options["output_label"]
    module.OUTPUT_LABEL_SUFFIX = ""
    module.NOISY_SIMULATION_METHOD = options.get(
        "noisy_simulation_method",
        module.NOISY_SIMULATION_METHOD,
    )
    module.NOISY_TRANSPILE_OPTIMIZATION_LEVEL = int(
        options.get(
            "noisy_transpile_optimization_level",
            module.NOISY_TRANSPILE_OPTIMIZATION_LEVEL,
        )
    )
    module.SIMPLE_NOISE_ONE_QUBIT_DEPOLARIZING_PROBABILITY = float(
        options.get(
            "simple_noise_one_qubit_depolarizing_probability",
            module.SIMPLE_NOISE_ONE_QUBIT_DEPOLARIZING_PROBABILITY,
        )
    )
    module.SIMPLE_NOISE_TWO_QUBIT_DEPOLARIZING_PROBABILITY = float(
        options.get(
            "simple_noise_two_qubit_depolarizing_probability",
            module.SIMPLE_NOISE_TWO_QUBIT_DEPOLARIZING_PROBABILITY,
        )
    )
    module.SIMPLE_NOISE_READOUT_ERROR_PROBABILITY = float(
        options.get(
            "simple_noise_readout_error_probability",
            module.SIMPLE_NOISE_READOUT_ERROR_PROBABILITY,
        )
    )
    module.IBM_MODEL_SOURCE = options.get("ibm_model_source", module.IBM_MODEL_SOURCE)
    module.IBM_MODEL_FAKE_BACKEND_CLASS = options.get(
        "ibm_model_fake_backend_class",
        module.IBM_MODEL_FAKE_BACKEND_CLASS,
    )
    module.IBM_MODEL_RUNTIME_BACKEND_NAME = options.get(
        "ibm_model_runtime_backend_name",
        module.IBM_MODEL_RUNTIME_BACKEND_NAME,
    )
    module.IBM_MODEL_COMPRESS_TO_ACTIVE_SPACE = bool(
        options.get(
            "ibm_model_compress_to_active_space",
            module.IBM_MODEL_COMPRESS_TO_ACTIVE_SPACE,
        )
    )
    module.QDRIFT_SEGMENT_COUNT_ND = int(options["qdrift_segment_count_Nd"])
    module.STOCHASTIC_INSTANCES_PER_CORRELATION = int(
        options["stochastic_instances_per_correlation"]
    )
    module.STOCHASTIC_WEIGHT_CONVENTION = options["stochastic_weight_convention"]
    module.RANDOM_SEED = int(options["random_seed"])
    module.MFE_VERBOSE_FOR_FIRST_NONZERO_POWER = bool(
        options.get("mfe_verbose_for_first_nonzero_power", False)
    )
    module.PRINT_CORRELATION_TABLE = bool(options.get("print_correlation_table", True))


def _metadata_matches(metadata, expected_options):
    options = metadata.get("options", {})
    comparisons = {
        "uqk_mode": expected_options["uqk_mode"],
        "krylov_dimension": int(expected_options["krylov_dimension"]),
        "max_correlation_power": int(expected_options["max_correlation_power"]),
        "shots_per_mfe_experiment": int(expected_options["shots_per_mfe_experiment"]),
        "backend_mode": expected_options["backend_mode"],
        "qdrift_segment_count_Nd": int(expected_options["qdrift_segment_count_Nd"]),
        "stochastic_instances_per_correlation": int(
            expected_options["stochastic_instances_per_correlation"]
        ),
        "stochastic_weight_convention": expected_options[
            "stochastic_weight_convention"
        ],
        "random_seed": int(expected_options["random_seed"]),
        "output_label_override": expected_options["output_label"],
    }
    for key, expected in comparisons.items():
        if options.get(key) != expected:
            return False, f"metadata options[{key!r}]={options.get(key)!r} != {expected!r}"
    if "dt" in options and not np.isclose(float(options["dt"]), expected_options["dt"]):
        return False, "metadata dt does not match experiment dt"
    return True, "metadata options match"


def run_uqk_overlap_cached(
    context,
    *,
    output_dir,
    output_label,
    uqk_mode,
    backend_mode,
    krylov_dimension,
    max_correlation_power,
    shots_per_mfe_experiment,
    qdrift_segment_count_Nd,
    stochastic_instances_per_correlation,
    stochastic_weight_convention,
    random_seed,
    reuse_existing=True,
    quiet=True,
    log_name=None,
    **backend_options,
):
    options = {
        "uqk_mode": uqk_mode,
        "backend_mode": backend_mode,
        "krylov_dimension": int(krylov_dimension),
        "max_correlation_power": int(max_correlation_power),
        "shots_per_mfe_experiment": int(shots_per_mfe_experiment),
        "qdrift_segment_count_Nd": int(qdrift_segment_count_Nd),
        "stochastic_instances_per_correlation": int(stochastic_instances_per_correlation),
        "stochastic_weight_convention": stochastic_weight_convention,
        "random_seed": int(random_seed),
        "output_label": output_label,
        "dt": float(context.dt),
        **backend_options,
    }

    repo_root = repo_root_from_helper()
    module = load_module_from_path(
        f"hyperparameter_build_uqk_overlap_{safe_label_part(output_label)}",
        repo_root / "scripts" / "qiskit" / "build_uqk_overlap_matrix.py",
    )
    _configure_builder_module(module, context, output_dir, options)
    output_npz = module.output_npz_path(module.UQK_MODE)
    output_metadata = module.output_metadata_path(module.UQK_MODE)
    log_path = context.logs_dir / (log_name or f"{output_label}.log")

    cache_status = "miss"
    cache_reason = "output files missing"
    if reuse_existing and output_npz.exists() and output_metadata.exists():
        metadata = load_json(output_metadata)
        matches, reason = _metadata_matches(metadata, options)
        if matches:
            cache_status = "hit"
            cache_reason = reason
        else:
            cache_reason = reason

    if cache_status != "hit":
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if quiet:
            with log_path.open("w", encoding="utf-8") as handle:
                with contextlib.redirect_stdout(handle):
                    module.main()
        else:
            module.main()
        cache_status = "generated"

    data = np.load(output_npz)
    metadata = load_json(output_metadata)
    return {
        "S": np.array(data["S"], dtype=np.complex128),
        "correlations": np.array(data["correlations"], dtype=np.complex128),
        "npz_path": output_npz,
        "metadata_path": output_metadata,
        "metadata": metadata,
        "cache_status": cache_status,
        "cache_reason": cache_reason,
        "log_path": log_path,
    }


def ensure_s_reference(
    context,
    *,
    krylov_dimension,
    backend_mode,
    random_seed,
    stochastic_weight_convention,
    s_reference_npz_override=None,
    s_reference_metadata_override=None,
    reuse_existing=True,
    quiet=True,
    **backend_options,
):
    """Load or generate the exact-trotter S_reference matrix."""

    reference_tag = sweep_label(context, backend_mode, krylov_dimension)
    info_path = context.data_dir / f"S_reference_info_{reference_tag}.json"
    alias_info_path = context.data_dir / "S_reference_info.json"

    if s_reference_npz_override:
        npz_path = Path(s_reference_npz_override)
        metadata_path = (
            Path(s_reference_metadata_override)
            if s_reference_metadata_override
            else None
        )
        data = np.load(npz_path)
        metadata = load_json(metadata_path) if metadata_path else {}
        info = {
            "source": "override",
            "npz_path": str(npz_path),
            "metadata_path": str(metadata_path) if metadata_path else None,
            "generated_or_reused": "override",
            "reference_tag": reference_tag,
            "backend_mode": backend_mode,
            "krylov_dimension": krylov_dimension,
            "dt": context.dt,
            "info_json": str(info_path),
            "active_alias_json": str(alias_info_path),
        }
        save_json(info_path, info)
        save_json(alias_info_path, info)
        return {
            "S": np.array(data["S"], dtype=np.complex128),
            "npz_path": npz_path,
            "metadata_path": metadata_path,
            "metadata": metadata,
            "info": info,
        }

    output_label = (
        f"{context.molecule_name}_S_reference_exact_trotter"
        f"_{reference_tag}"
    )
    result = run_uqk_overlap_cached(
        context,
        output_dir=context.reference_dir,
        output_label=output_label,
        uqk_mode="exact_trotter",
        backend_mode=backend_mode,
        krylov_dimension=krylov_dimension,
        max_correlation_power=krylov_dimension,
        shots_per_mfe_experiment=0,
        qdrift_segment_count_Nd=1,
        stochastic_instances_per_correlation=1,
        stochastic_weight_convention=stochastic_weight_convention,
        random_seed=random_seed,
        reuse_existing=reuse_existing,
        quiet=quiet,
        log_name=f"{output_label}.log",
        **backend_options,
    )
    info = {
        "source": "generated_exact_trotter",
        "npz_path": str(result["npz_path"]),
        "metadata_path": str(result["metadata_path"]),
        "cache_status": result["cache_status"],
        "cache_reason": result["cache_reason"],
        "krylov_dimension": krylov_dimension,
        "dt": context.dt,
        "reference_tag": reference_tag,
        "backend_mode": backend_mode,
        "info_json": str(info_path),
        "active_alias_json": str(alias_info_path),
    }
    save_json(info_path, info)
    save_json(alias_info_path, info)
    return {**result, "info": info}


def seed_for_point(base_seed, point_index):
    return int(base_seed) + int(point_index)


def add_error_fields(record, s_reference, s_candidate):
    record.update(matrix_error_metrics(s_reference, s_candidate))
    return record


def write_records_csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = sorted({key for record in records for key in record})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: (
                        str(value)
                        if isinstance(value, Path)
                        else value
                    )
                    for key, value in record.items()
                }
            )
    return path


def records_package_paths(path_base):
    path_base = Path(path_base)
    return {
        "csv": path_base.with_suffix(".csv"),
        "json": path_base.with_suffix(".json"),
    }


def load_records_package(path_base):
    paths = records_package_paths(path_base)
    if not paths["json"].exists():
        raise FileNotFoundError(
            "Existing sweep data JSON was not found. Run this cell once with "
            f"run_jobs=True to generate it: {paths['json']}"
        )
    payload = load_json(paths["json"])
    records = payload.get("records")
    if records is None:
        raise KeyError(f"Existing sweep data JSON has no 'records' field: {paths['json']}")
    return records, paths


def save_records_package(path_base, records, extra_metadata=None):
    paths = records_package_paths(path_base)
    csv_path = paths["csv"]
    json_path = paths["json"]
    write_records_csv(csv_path, records)
    save_json(
        json_path,
        {
            "generated_at_utc": now_utc(),
            "num_records": len(records),
            "records": records,
            "metadata": extra_metadata or {},
        },
    )
    return {"csv": csv_path, "json": json_path}


def derive_shot_pairs(total_shots_per_correlation, ratios_nmfe_over_nw):
    pairs = []
    for ratio in ratios_nmfe_over_nw:
        nmfe = math.sqrt(float(total_shots_per_correlation) * float(ratio))
        nw = math.sqrt(float(total_shots_per_correlation) / float(ratio))
        if not np.isclose(nmfe, round(nmfe)) or not np.isclose(nw, round(nw)):
            raise ValueError(
                "Shot budget pair is not integral for "
                f"Nk={total_shots_per_correlation}, R={ratio}."
            )
        pairs.append(
            {
                "total_shots_per_correlation": int(total_shots_per_correlation),
                "ratio_nmfe_over_nw": float(ratio),
                "shots_per_mfe_experiment": int(round(nmfe)),
                "stochastic_instances_per_correlation": int(round(nw)),
            }
        )
    return pairs


def generate_plot1_exact_stochastic_data(
    context,
    s_reference,
    *,
    nd_values,
    sipc_values,
    krylov_dimension,
    backend_mode,
    random_seed,
    stochastic_weight_convention,
    reuse_existing=True,
    run_jobs=True,
    RUN_JOBS=None,
    quiet=True,
    **backend_options,
):
    print_header("Plot 1 Data: exact_stochastic qDRIFT convergence")
    run_jobs = resolve_run_jobs(run_jobs, RUN_JOBS)
    run_tag = sweep_label(context, backend_mode, krylov_dimension)
    path_base = context.data_dir / f"plot1_exact_stochastic_overlap_error_{run_tag}"
    if not run_jobs:
        records, paths = load_records_package(path_base)
        print_kv("RUN_JOBS:", False)
        print_kv("Loaded records:", len(records))
        print_kv("Data JSON:", paths["json"])
        return records, paths

    records = []
    point_index = 0
    for sipc in sipc_values:
        for nd in nd_values:
            seed = seed_for_point(random_seed, point_index)
            output_label = (
                f"{context.molecule_name}_plot1_exact_stochastic"
                f"_{run_tag}"
                f"_Nd_{nd}_sipc_{sipc}_seed_{seed}"
            )
            result = run_uqk_overlap_cached(
                context,
                output_dir=context.results_dir / "plot1_exact_stochastic",
                output_label=output_label,
                uqk_mode="exact_stochastic",
                backend_mode=backend_mode,
                krylov_dimension=krylov_dimension,
                max_correlation_power=krylov_dimension,
                shots_per_mfe_experiment=0,
                qdrift_segment_count_Nd=nd,
                stochastic_instances_per_correlation=sipc,
                stochastic_weight_convention=stochastic_weight_convention,
                random_seed=seed,
                reuse_existing=reuse_existing,
                quiet=quiet,
                **backend_options,
            )
            record = {
                "plot": "plot1",
                "uqk_mode": "exact_stochastic",
                "backend_mode": backend_mode,
                "krylov_dimension": krylov_dimension,
                "dt": context.dt,
                "Nd": int(nd),
                "sipc": int(sipc),
                "Nw": int(sipc),
                "Nmfe": None,
                "random_seed": seed,
                "cache_status": result["cache_status"],
                "npz_path": str(result["npz_path"]),
                "metadata_path": str(result["metadata_path"]),
                "log_path": str(result["log_path"]),
            }
            add_error_fields(record, s_reference, result["S"])
            records.append(record)
            print_kv(f"Nd={nd}, sipc={sipc}", f"rel_fro={record['relative_frobenius_error']:.6e}")
            point_index += 1
    paths = save_records_package(
        path_base,
        records,
        current_s_reference_metadata(context),
    )
    return records, paths


def generate_plot2_stochastic_shot_budget_data(
    context,
    s_reference,
    *,
    nd_values,
    total_shots_values,
    ratios_by_total_shots,
    krylov_dimension,
    backend_mode,
    random_seed,
    stochastic_weight_convention,
    reuse_existing=True,
    run_jobs=True,
    RUN_JOBS=None,
    quiet=True,
    **backend_options,
):
    print_header("Plot 2 Data: finite-shot stochastic qDRIFT")
    run_jobs = resolve_run_jobs(run_jobs, RUN_JOBS)
    run_tag = sweep_label(context, backend_mode, krylov_dimension)
    path_base = context.data_dir / f"plot2_stochastic_shot_budget_overlap_error_{run_tag}"
    if not run_jobs:
        records, paths = load_records_package(path_base)
        print_kv("RUN_JOBS:", False)
        print_kv("Loaded records:", len(records))
        print_kv("Data JSON:", paths["json"])
        return records, paths

    records = []
    point_index = 0
    for total_shots in total_shots_values:
        shot_pairs = derive_shot_pairs(total_shots, ratios_by_total_shots[total_shots])
        for pair in shot_pairs:
            for nd in nd_values:
                seed = seed_for_point(random_seed, point_index)
                nmfe = pair["shots_per_mfe_experiment"]
                nw = pair["stochastic_instances_per_correlation"]
                ratio_label = safe_label_part(pair["ratio_nmfe_over_nw"])
                output_label = (
                    f"{context.molecule_name}_plot2_stochastic"
                    f"_{run_tag}"
                    f"_Nd_{nd}_Nk_{total_shots}_R_{ratio_label}"
                    f"_Nmfe_{nmfe}_Nw_{nw}_seed_{seed}"
                )
                result = run_uqk_overlap_cached(
                    context,
                    output_dir=context.results_dir / "plot2_stochastic_shot_budget",
                    output_label=output_label,
                    uqk_mode="stochastic",
                    backend_mode=backend_mode,
                    krylov_dimension=krylov_dimension,
                    max_correlation_power=krylov_dimension,
                    shots_per_mfe_experiment=nmfe,
                    qdrift_segment_count_Nd=nd,
                    stochastic_instances_per_correlation=nw,
                    stochastic_weight_convention=stochastic_weight_convention,
                    random_seed=seed,
                    reuse_existing=reuse_existing,
                    quiet=quiet,
                    **backend_options,
                )
                record = {
                    "plot": "plot2",
                    "uqk_mode": "stochastic",
                    "backend_mode": backend_mode,
                    "krylov_dimension": krylov_dimension,
                    "dt": context.dt,
                    "Nd": int(nd),
                    "sipc": int(nw),
                    "Nw": int(nw),
                    "Nmfe": int(nmfe),
                    "total_shots_per_correlation": int(total_shots),
                    "ratio_nmfe_over_nw": pair["ratio_nmfe_over_nw"],
                    "random_seed": seed,
                    "cache_status": result["cache_status"],
                    "npz_path": str(result["npz_path"]),
                    "metadata_path": str(result["metadata_path"]),
                    "log_path": str(result["log_path"]),
                }
                add_error_fields(record, s_reference, result["S"])
                records.append(record)
                print_kv(
                    f"Nk={total_shots}, R={pair['ratio_nmfe_over_nw']}, Nd={nd}",
                    f"rel_fro={record['relative_frobenius_error']:.6e}",
                )
                point_index += 1
    paths = save_records_package(
        path_base,
        records,
        current_s_reference_metadata(context),
    )
    return records, paths


def generate_plot3_standard_nmfe_data(
    context,
    s_reference,
    *,
    nmfe_values,
    krylov_dimension,
    backend_mode,
    random_seed,
    stochastic_weight_convention,
    reuse_existing=True,
    run_jobs=True,
    RUN_JOBS=None,
    quiet=True,
    **backend_options,
):
    print_header("Plot 3 Data: finite-shot standard Trotter")
    run_jobs = resolve_run_jobs(run_jobs, RUN_JOBS)
    run_tag = sweep_label(context, backend_mode, krylov_dimension)
    path_base = context.data_dir / f"plot3_standard_nmfe_overlap_error_{run_tag}"
    if not run_jobs:
        records, paths = load_records_package(path_base)
        print_kv("RUN_JOBS:", False)
        print_kv("Loaded records:", len(records))
        print_kv("Data JSON:", paths["json"])
        return records, paths

    records = []
    for point_index, nmfe in enumerate(nmfe_values):
        seed = seed_for_point(random_seed, point_index)
        output_label = (
            f"{context.molecule_name}_plot3_standard"
            f"_{run_tag}"
            f"_Nmfe_{nmfe}_seed_{seed}"
        )
        result = run_uqk_overlap_cached(
            context,
            output_dir=context.results_dir / "plot3_standard_nmfe",
            output_label=output_label,
            uqk_mode="standard",
            backend_mode=backend_mode,
            krylov_dimension=krylov_dimension,
            max_correlation_power=krylov_dimension,
            shots_per_mfe_experiment=nmfe,
            qdrift_segment_count_Nd=1,
            stochastic_instances_per_correlation=1,
            stochastic_weight_convention=stochastic_weight_convention,
            random_seed=seed,
            reuse_existing=reuse_existing,
            quiet=quiet,
            **backend_options,
        )
        record = {
            "plot": "plot3",
            "uqk_mode": "standard",
            "backend_mode": backend_mode,
            "krylov_dimension": krylov_dimension,
            "dt": context.dt,
            "Nd": None,
            "sipc": None,
            "Nw": None,
            "Nmfe": int(nmfe),
            "random_seed": seed,
            "cache_status": result["cache_status"],
            "npz_path": str(result["npz_path"]),
            "metadata_path": str(result["metadata_path"]),
            "log_path": str(result["log_path"]),
        }
        add_error_fields(record, s_reference, result["S"])
        records.append(record)
        print_kv(f"Nmfe={nmfe}", f"rel_fro={record['relative_frobenius_error']:.6e}")
    paths = save_records_package(
        path_base,
        records,
        current_s_reference_metadata(context),
    )
    return records, paths


def _records_sorted(records, *keys):
    return sorted(records, key=lambda row: tuple(row[key] for key in keys))


def _best_by_group(records, group_keys):
    grouped = {}
    for record in records:
        group = tuple(record[key] for key in group_keys)
        if group not in grouped or record["relative_frobenius_error"] < grouped[group]["relative_frobenius_error"]:
            grouped[group] = record
    return list(grouped.values())


def save_markdown_table(path, records, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for record in records:
        values = []
        for column in columns:
            value = record.get(column)
            if isinstance(value, float):
                values.append(f"{value:.6e}")
            else:
                values.append("" if value is None else str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def save_summary_table(context, stem, records, group_keys, columns):
    summary_records = _best_by_group(records, group_keys) if group_keys else records
    if group_keys:
        summary_records = sorted(
            summary_records,
            key=lambda row: tuple(
                -1 if row.get(key) is None else row.get(key)
                for key in group_keys
            ),
        )
    else:
        summary_records = list(summary_records)
    csv_path = write_records_csv(context.tables_dir / f"{stem}.csv", summary_records)
    md_path = save_markdown_table(
        context.tables_dir / f"{stem}.md",
        summary_records,
        columns,
    )
    return {"csv": csv_path, "markdown": md_path, "records": summary_records}


def _setup_matplotlib():
    import matplotlib.pyplot as plt

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass
    return plt


def _finish_plot(fig, context, stem):
    png_path = context.plots_dir / f"{stem}.png"
    pdf_path = context.plots_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return {"png": png_path, "pdf": pdf_path}


def _maybe_log_y(ax, records):
    values = [row["relative_frobenius_error"] for row in records]
    if values and min(values) > 0.0:
        ax.set_yscale("log")


def _records_run_tag(records, context):
    if not records:
        return sweep_label(context, "unknown_backend", 0)
    return sweep_label(
        context,
        records[0].get("backend_mode", "unknown_backend"),
        records[0].get("krylov_dimension", 0),
    )


def plot1_exact_stochastic(records, context):
    plt = _setup_matplotlib()
    run_tag = _records_run_tag(records, context)
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for sipc in sorted({row["sipc"] for row in records}):
        rows = _records_sorted([row for row in records if row["sipc"] == sipc], "Nd")
        ax.plot(
            [row["Nd"] for row in rows],
            [row["relative_frobenius_error"] for row in rows],
            marker="o",
            linewidth=1.8,
            label=f"Nw={sipc}",
        )
    ax.set_xscale("log", base=2)
    _maybe_log_y(ax, records)
    ax.set_xlabel("qDRIFT segment count Nd")
    ax.set_ylabel("Relative Frobenius error in S")
    ax.set_title("exact_stochastic overlap error vs Nd and Nw")
    ax.legend(title="Instances")
    fig.tight_layout()
    plot_paths = _finish_plot(
        fig,
        context,
        f"plot1_exact_stochastic_overlap_error_{run_tag}",
    )
    table = save_summary_table(
        context,
        f"plot1_exact_stochastic_best_by_Nw_{run_tag}",
        records,
        ["sipc"],
        ["sipc", "Nd", "relative_frobenius_error", "absolute_frobenius_error", "random_seed"],
    )
    return {"plots": plot_paths, "table": table}


def plot2_stochastic_shot_budget(records, context):
    plt = _setup_matplotlib()
    run_tag = _records_run_tag(records, context)
    total_shots_values = sorted({row["total_shots_per_correlation"] for row in records})
    fig, axes = plt.subplots(
        1,
        len(total_shots_values),
        figsize=(6.4 * len(total_shots_values), 5.2),
        sharey=True,
    )
    if len(total_shots_values) == 1:
        axes = [axes]
    for ax, total_shots in zip(axes, total_shots_values):
        subset = [row for row in records if row["total_shots_per_correlation"] == total_shots]
        for ratio in sorted({row["ratio_nmfe_over_nw"] for row in subset}, reverse=True):
            rows = _records_sorted(
                [row for row in subset if row["ratio_nmfe_over_nw"] == ratio],
                "Nd",
            )
            if not rows:
                continue
            label = (
                f"R={ratio:g}, Nmfe={rows[0]['Nmfe']}, Nw={rows[0]['Nw']}"
            )
            ax.plot(
                [row["Nd"] for row in rows],
                [row["relative_frobenius_error"] for row in rows],
                marker="o",
                linewidth=1.8,
                label=label,
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("qDRIFT segment count Nd")
        ax.set_title(f"Nk = Nmfe*Nw = {total_shots:,}")
        ax.legend(fontsize=8)
    _maybe_log_y(axes[0], records)
    axes[0].set_ylabel("Relative Frobenius error in S")
    fig.suptitle("stochastic overlap error vs Nd under fixed shot budgets", y=1.02)
    fig.tight_layout()
    plot_paths = _finish_plot(
        fig,
        context,
        f"plot2_stochastic_shot_budget_overlap_error_{run_tag}",
    )
    table = save_summary_table(
        context,
        f"plot2_stochastic_best_by_Nk_and_R_{run_tag}",
        records,
        ["total_shots_per_correlation", "ratio_nmfe_over_nw"],
        [
            "total_shots_per_correlation",
            "ratio_nmfe_over_nw",
            "Nmfe",
            "Nw",
            "Nd",
            "relative_frobenius_error",
            "absolute_frobenius_error",
            "random_seed",
        ],
    )
    return {"plots": plot_paths, "table": table}


def plot3_standard_nmfe(records, context):
    plt = _setup_matplotlib()
    run_tag = _records_run_tag(records, context)
    rows = _records_sorted(records, "Nmfe")
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.plot(
        [row["Nmfe"] for row in rows],
        [row["relative_frobenius_error"] for row in rows],
        marker="o",
        linewidth=1.9,
    )
    ax.set_xscale("log")
    _maybe_log_y(ax, records)
    ax.set_xlabel("Shots per MFE experiment Nmfe")
    ax.set_ylabel("Relative Frobenius error in S")
    ax.set_title("standard overlap error vs finite MFE shots")
    fig.tight_layout()
    plot_paths = _finish_plot(
        fig,
        context,
        f"plot3_standard_nmfe_overlap_error_{run_tag}",
    )
    table = save_summary_table(
        context,
        f"plot3_standard_nmfe_summary_{run_tag}",
        rows,
        [],
        ["Nmfe", "relative_frobenius_error", "absolute_frobenius_error", "random_seed"],
    )
    return {"plots": plot_paths, "table": table}
