"""Implementations of the `main.py` subcommands."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmark import get_benchmark, list_benchmarks
from descriptor.experiment import analyze_task_runs, write_run_trace
from descriptor.metrics import RunOutcome, resolve_run_outcome
from MAS import MASRunner, OpenRouterLLMClient, load_experiment_config
from MAS.prompting_baselines import (
    BASELINE_DIRECT,
    PromptingBaselineRunner,
    normalize_prompting_baseline,
)
from MAS.relay import TOPOLOGY_SELF_EVOLVED

try:
    from datetime import UTC
except ImportError:  # pragma: no cover - Python < 3.11 fallback
    from datetime import timezone

    UTC = timezone.utc  # noqa: UP017

from cli.artifacts import (
    _classify_run_exception,
    _failed_run_result,
    _run_progress_summary,
    _summary_row_from_analysis,
    _summary_task_entry_from_payload,
    _write_eval,
    _write_raw_output,
    _write_run_artifacts,
    _write_task_checkpoint,
)
from cli.common import (
    _env_truthy,
    _log_progress,
    _mean,
    _prompt_preview,
    _write_json,
    _write_summary_csv,
)
from cli.graphs import _write_system_graph_artifact
from cli.resume import (
    _load_completed_run_resume,
    _load_completed_task_resume,
    _metadata_needs_rerun,
    _task_payload_needs_rerun,
)
from cli.settings import (
    _apply_benchmark_overrides,
    _apply_mas_overrides,
    _benchmark_section_config,
    _experiment_settings_payload,
    _mas_mode_label,
    _resolve_output_paths,
    _write_experiment_settings,
)
from cli.trajectory import _compact_run_metadata


def _build_skill_updater(config: Any, runner: MASRunner) -> Any | None:
    """Online (in-experiment) skill updater for self_evolved runs, or None.

    Returns None unless the topology is self_evolved and
    `self_evolved.skill_update_batch_size > 0` (the default is 12). When enabled, every
    N freshly executed runs are reflected into the skill (labelled by process signals
    only — auditor findings + consensus — never the eval verdict) and the engine reloads
    it. Set the batch size to 0 for parallel experiments."""

    se = config.self_evolved
    if (
        config.mas.resolved_topology() != TOPOLOGY_SELF_EVOLVED
        or int(se.skill_update_batch_size) <= 0
    ):
        return None
    from MAS.self_evolved.skill import OnlineSkillUpdater, SkillReflector

    return OnlineSkillUpdater(
        reflector=SkillReflector(runner.openrouter_client, se),
        skill_path=se.skill_path,
        batch_size=int(se.skill_update_batch_size),
        on_update=runner.reload_self_evolved_skill,
        log=_log_progress,
    )


def run_command(args: argparse.Namespace) -> int:
    # 1) Load runtime knobs (OpenRouter, MAS topology, model routing, benchmark settings).
    config = load_experiment_config(args.config)
    _apply_mas_overrides(config, args)

    benchmark_name = args.benchmark
    benchmark_cfg = _benchmark_section_config(config, benchmark_name)
    benchmark_cfg = _apply_benchmark_overrides(benchmark_cfg, args)
    # 2) Instantiate the benchmark adapter and MAS runtime.
    benchmark = get_benchmark(benchmark_name, config=benchmark_cfg)

    llm_client = OpenRouterLLMClient(config.openrouter, config.models)
    base_runner = MASRunner(config, llm_client)
    prompting_baseline = normalize_prompting_baseline(
        getattr(args, "prompting_baseline", BASELINE_DIRECT)
    )
    runner: Any = base_runner
    if prompting_baseline != BASELINE_DIRECT:
        runner = PromptingBaselineRunner(
            base_runner,
            baseline=prompting_baseline,
            self_consistency_samples=int(getattr(args, "self_consistency_samples", 3)),
            self_refine_rounds=int(getattr(args, "self_refine_rounds", 3)),
        )
    skill_updater = _build_skill_updater(config, runner)
    if skill_updater is not None:
        _log_progress(
            "SKILL_ONLINE_UPDATE enabled "
            f"batch_size={config.self_evolved.skill_update_batch_size} "
            f"skill_path={config.self_evolved.skill_path}"
        )

    task_limit = args.task_limit if args.task_limit is not None else config.experiment.task_limit
    runs_per_task = (
        args.runs_per_task if args.runs_per_task is not None else config.experiment.runs_per_task
    )
    seed = args.seed if args.seed is not None else config.experiment.seed
    output_root = Path(args.output_dir or config.experiment.output_dir)
    output_paths = _resolve_output_paths(
        args=args,
        config=config,
        benchmark_name=benchmark_name,
        output_root=output_root,
    )
    output_paths.benchmark_root.mkdir(parents=True, exist_ok=True)
    output_paths.run_root.mkdir(parents=True, exist_ok=True)

    task_id_filter = [
        tid.strip() for tid in str(getattr(args, "task_ids", "") or "").split(",") if tid.strip()
    ]
    if task_id_filter:
        # Explicit id selection: load everything, keep only the requested ids in load
        # order. Overrides offset/limit so a fixed hard-example set reruns identically.
        wanted = set(task_id_filter)
        restrict = getattr(benchmark, "restrict_to_task_ids", None)
        if callable(restrict):
            # Large corpora (BrowseComp-Plus) then keep only the requested tasks in memory.
            restrict(wanted)
        tasks = [
            task for task in benchmark.load_tasks(task_limit=None) if str(task.task_id) in wanted
        ]
        missing = wanted - {str(task.task_id) for task in tasks}
        if missing:
            raise RuntimeError(
                f"--task-ids not found for benchmark '{benchmark_name}': {sorted(missing)}"
            )
    else:
        task_offset = max(int(getattr(args, "task_offset", 0) or 0), 0)
        # Load offset+limit tasks in deterministic order, then drop the leading offset so
        # this shard covers exactly tasks[offset : offset+limit] (parallel sharding).
        load_limit = (task_limit + task_offset) if task_limit is not None else None
        tasks = list(benchmark.load_tasks(task_limit=load_limit))
        if task_offset:
            tasks = tasks[task_offset:]
    if not tasks:
        raise RuntimeError(f"No tasks loaded for benchmark '{benchmark_name}'")
    _log_progress(
        "Loaded run configuration "
        f"benchmark={benchmark_name} system={args.system_label or config.mas.resolved_topology()} "
        f"tasks={len(tasks)} runs_per_task={runs_per_task} seed={seed} "
        f"prompting_baseline={prompting_baseline} "
        f"topology={config.mas.resolved_topology()} agents={config.mas.total_agents} "
        f"rounds={int(config.mas.max_turns)} discussion_rounds={int(config.mas.discussion_rounds)} "
        f"communication_budget={int(config.mas.communication_count_internally)}"
    )

    experiment_settings = _experiment_settings_payload(
        args=args,
        config=config,
        benchmark_name=benchmark_name,
        benchmark_cfg=benchmark_cfg,
        task_limit=task_limit,
        runs_per_task=runs_per_task,
        seed=seed,
        task_count=len(tasks),
        run_root=output_paths.run_root,
        output_paths=output_paths,
    )
    settings_path = output_paths.run_root / "experiment_settings.json"
    _write_experiment_settings(settings_path, experiment_settings)

    graph_payload: dict[str, Any] | None = None
    if output_paths.output_layout == "hierarchical":
        graph_payload = _write_system_graph_artifact(
            runner=runner,
            config=config,
            run_root=output_paths.run_root,
        )

    system_info = {
        "system_label": output_paths.system_label,
        "mode": _mas_mode_label(config),
        "topology": config.mas.resolved_topology(),
        "prompting_baseline": prompting_baseline,
        "self_consistency_samples": int(getattr(args, "self_consistency_samples", 3)),
        "self_refine_rounds": int(getattr(args, "self_refine_rounds", 3)),
        "agents": config.mas.total_agents,
        "agents_per_level": config.mas.resolved_agents_per_level(),
        "group_sizes": list(config.mas.group_sizes) if config.mas.group_sizes is not None else None,
        "agent_types": list(config.mas.agent_types),
        "max_turns": int(config.mas.max_turns),
        "discussion_rounds": int(config.mas.discussion_rounds),
        "termination_consensus_mode": str(config.mas.termination_consensus_mode),
        "final_vote_mode": str(config.mas.final_vote_mode),
        "peer_artifact_max_chars": int(config.mas.peer_artifact_max_chars),
        "communication_budget": int(config.mas.communication_count_internally),
        "harness_backend": str(config.self_evolved.harness_backend),
    }

    summary_rows: list[dict[str, Any]] = []
    summary_json: dict[str, Any] = {
        "timestamp": output_paths.experiment_id,
        "experiment_id": output_paths.experiment_id,
        "output_layout": output_paths.output_layout,
        "benchmark": benchmark_name,
        "system_label": output_paths.system_label,
        "system": system_info,
        "config_path": str(Path(args.config).resolve()),
        "runs_per_task": runs_per_task,
        "task_count": len(tasks),
        "experiment_settings_path": str(settings_path.resolve()),
        "tasks": [],
    }
    if graph_payload is not None:
        summary_json["graph"] = graph_payload

    default_model = str(config.models.get("default", ""))
    judge_model = str(config.models.get("judge", config.models.get("default", "")))

    for task_idx, task in enumerate(tasks):
        task_dir = (
            output_paths.run_root / task.task_id
            if output_paths.output_layout == "hierarchical"
            else output_paths.benchmark_root / task.task_id
        )
        task_dir.mkdir(parents=True, exist_ok=True)

        resumed = _load_completed_task_resume(
            task=task,
            task_dir=task_dir,
            benchmark_name=benchmark_name,
            system_info=system_info,
            runs_per_task=runs_per_task,
            default_model=default_model,
            judge_model=judge_model,
        )
        if resumed is not None:
            task_summary, row = resumed
            summary_json["tasks"].append(task_summary)
            summary_rows.append(row)
            _log_progress(
                f"TASK_RESUME_SKIP index={task_idx + 1}/{len(tasks)} task_id={task.task_id} "
                f"path={task_dir / 'task_summary.json'}"
            )
            continue

        _log_progress(f"TASK_START index={task_idx + 1}/{len(tasks)} task_id={task.task_id}")
        run_traces = []
        evaluations = []
        run_outcomes: list[RunOutcome] = []
        run_artifacts: list[dict[str, Any]] = []
        task_failure_details: list[dict[str, Any]] = []

        for run_index in range(runs_per_task):
            run_seed = seed + (task_idx * 1000) + run_index
            resumed_run = _load_completed_run_resume(task_dir=task_dir, run_index=run_index)
            if resumed_run is not None:
                (
                    final_answer,
                    trace_events,
                    run_metadata,
                    evaluation,
                    run_outcome,
                    artifact_record,
                ) = resumed_run
                run_traces.append(trace_events)
                evaluations.append(evaluation)
                run_outcomes.append(run_outcome)
                run_artifacts.append(artifact_record)
                _log_progress(
                    f"RUN_RESUME_SKIP task_id={task.task_id} run_index={run_index} "
                    f"path={task_dir / f'run_{run_index}.metadata.json'}"
                )
                continue

            run_started = datetime.now(UTC)
            run_started_s = time.time()
            _log_progress(f"RUN_START task_id={task.task_id} run_index={run_index} seed={run_seed}")
            try:
                if hasattr(runner, "set_run_checkpoint_context"):
                    runner.set_run_checkpoint_context(task_dir=task_dir, run_index=run_index)
                run = benchmark.run(
                    task=task,
                    runner=runner,
                    run_index=run_index,
                    seed=run_seed,
                )
            except Exception as exc:
                failure_category = _classify_run_exception(exc)
                _log_progress(
                    f"RUN_ERROR task_id={task.task_id} run_index={run_index} "
                    f"seed={run_seed} category={failure_category} "
                    f"error={type(exc).__name__}:{exc}"
                )
                (
                    final_answer,
                    trace_events,
                    run_metadata,
                    evaluation,
                    run_outcome,
                ) = _failed_run_result(
                    task=task,
                    benchmark_name=benchmark_name,
                    system_info=system_info,
                    run_index=run_index,
                    seed=run_seed,
                    exc=exc,
                    started_s=run_started_s,
                )
                task_failure_details.append(
                    {
                        "task_id": str(task.task_id),
                        "run_index": int(run_index),
                        "seed": int(run_seed),
                        "failure_category": failure_category,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
            else:
                final_answer = run.final_answer
                trace_events = run.trace_events
                run_metadata = run.run_metadata
                evaluation = None
                run_outcome = None
            finally:
                if hasattr(runner, "set_run_checkpoint_context"):
                    runner.set_run_checkpoint_context(task_dir=None, run_index=None)

            _log_progress(
                f"RUN_FINISH task_id={task.task_id} run_index={run_index} "
                f"elapsed_s={(datetime.now(UTC) - run_started).total_seconds():.2f} "
                f"trace_events={len(trace_events)} final_answer_chars={len(str(final_answer or ''))} "
                f"status={str(run_metadata.get('run_status', 'completed') or 'completed')}"
            )

            trace_path = task_dir / f"run_{run_index}.trace.jsonl"
            write_run_trace(trace_events, trace_path)
            run_traces.append(trace_events)
            _log_progress(
                f"TRACE_WRITTEN task_id={task.task_id} run_index={run_index} path={trace_path}"
            )

            raw_output_path = task_dir / f"run_{run_index}.raw.json"
            _write_raw_output(
                raw_output_path,
                final_answer=final_answer,
                run_metadata=run_metadata,
            )
            _log_progress(
                f"RAW_OUTPUT_WRITTEN task_id={task.task_id} run_index={run_index} path={raw_output_path}"
            )

            # 4) Let the benchmark score the model output.
            if evaluation is None:
                evaluation = benchmark.evaluate(
                    task,
                    final_answer,
                    run_metadata=run_metadata,
                )
            _log_progress(
                f"EVAL_FINISH task_id={task.task_id} run_index={run_index} "
                f"score={float(evaluation.score):.4f} success={bool(evaluation.success)}"
            )
            if run_outcome is None:
                run_outcome = resolve_run_outcome(
                    trace_events,
                    evaluation=evaluation,
                    final_answer=final_answer,
                    run_metadata=run_metadata,
                )
            if _metadata_needs_rerun(run_metadata):
                run_metadata["fallback"] = True
                run_metadata["needs_rerun"] = True
                run_metadata.setdefault("fallback_reason", "llm_or_tool_fallback")
            evaluations.append(evaluation)
            run_outcomes.append(run_outcome)

            artifact_paths = _write_run_artifacts(
                task_dir=task_dir,
                benchmark_name=benchmark_name,
                task=task,
                run_index=run_index,
                final_answer=final_answer,
                trace_events=trace_events,
                evaluation=evaluation,
                run_outcome=run_outcome,
                run_metadata=run_metadata,
                system_info=system_info,
            )
            eval_path = task_dir / f"run_{run_index}.eval.json"
            _write_eval(
                eval_path,
                evaluation,
                final_answer,
                run_outcome=run_outcome,
                metadata_summary=_compact_run_metadata(run_metadata),
                metadata_path=Path(artifact_paths["metadata_path"]),
            )
            run_artifacts.append(
                {
                    **artifact_paths,
                    "trace_path": str(trace_path.resolve()),
                    "eval_path": str(eval_path.resolve()),
                    "score": float(evaluation.score),
                    "success": bool(evaluation.success),
                    "completion": bool(run_outcome.completion),
                }
            )
            _log_progress(
                f"RUN_ARTIFACTS_WRITTEN task_id={task.task_id} run_index={run_index} "
                f"metadata_path={artifact_paths['metadata_path']} eval_path={eval_path}"
            )
            _write_task_checkpoint(
                task_dir=task_dir,
                task=task,
                benchmark_name=benchmark_name,
                system_info=system_info,
                runs_per_task=runs_per_task,
                run_artifacts=run_artifacts,
                task_failure_details=task_failure_details,
                checkpoint_reason=f"run_{run_index}_complete",
            )
            _log_progress(
                f"TASK_CHECKPOINT_WRITTEN task_id={task.task_id} "
                f"completed_runs={len(run_artifacts)}/{runs_per_task} "
                f"path={task_dir / 'task_summary.json'}"
            )

            # Online skill learning: feed this freshly executed run's playbook candidate
            # (PROCESS SIGNALS ONLY — never the eval verdict, to avoid biasing the study)
            # to the updater; it pauses to reflect into the skill every N runs. No-op
            # unless self_evolved + skill_update_batch_size > 0.
            if skill_updater is not None:
                candidate = (run_metadata.get("self_evolved") or {}).get(
                    "playbook_update_candidate"
                )
                skill_updater.record(candidate)

        # 5) Convert trace+eval into descriptor artifacts and analysis outputs.
        _log_progress(f"TASK_ANALYZE_START task_id={task.task_id}")
        analysis = analyze_task_runs(
            task_id=task.task_id,
            benchmark_name=benchmark_name,
            run_traces=run_traces,
            evaluations=evaluations,
            run_outcomes=run_outcomes,
            output_dir=task_dir,
        )
        _log_progress(
            f"TASK_ANALYZE_FINISH task_id={task.task_id} "
            f"avg_score={float(analysis['evaluation'].get('avg_score', 0.0)):.4f} "
            f"success_rate={float(analysis['evaluation'].get('success_rate', 0.0)):.4f} "
            f"completion_rate={float(analysis['evaluation'].get('completion_rate', 0.0)):.4f}"
        )
        run_status_summary, run_failure_count, fallback_count = _run_progress_summary(run_artifacts)
        needs_rerun = run_failure_count > 0 or fallback_count > 0 or bool(task_failure_details)
        rerun_details = {
            "run_failure_count": run_failure_count,
            "fallback_count": fallback_count,
            "run_status_summary": run_status_summary,
            "failure_details": task_failure_details,
        }

        task_summary_payload = {
            "task_id": str(task.task_id),
            "benchmark": benchmark_name,
            "system": system_info,
            "task_dir": str(task_dir.resolve()),
            "prompt_preview": _prompt_preview(task.prompt),
            "reference_answer": task.reference_answer,
            "checkpoint_complete": True,
            "checkpoint_reason": "task_analysis_complete",
            "checkpoint_updated_at": datetime.now(UTC).isoformat(),
            "completed_runs": len(run_artifacts),
            "runs_per_task": int(runs_per_task),
            "evaluation": analysis["evaluation"],
            "descriptor": analysis["descriptor"],
            "stage_bottleneck": analysis["stage_bottleneck"],
            "runs": run_artifacts,
            "run_status_summary": run_status_summary,
            "run_failure_count": run_failure_count,
            "fallback_count": fallback_count,
            "needs_rerun": needs_rerun,
            "rerun_details": rerun_details,
            "artifacts": {
                "analysis_path": str((task_dir / "analysis.json").resolve()),
                "descriptor_json_path": str((task_dir / "descriptor.json").resolve()),
                "descriptor_csv_path": str((task_dir / "descriptor.csv").resolve()),
            },
        }
        _write_json(task_dir / "task_summary.json", task_summary_payload)

        task_summary = {
            "task_id": task.task_id,
            "prompt_preview": _prompt_preview(task.prompt),
            "reference_answer": task.reference_answer,
            "task_dir": str(task_dir.resolve()),
            "evaluation": analysis["evaluation"],
            "descriptor": analysis["descriptor"],
            "stage_bottleneck": analysis["stage_bottleneck"],
            "run_status_summary": run_status_summary,
            "run_failure_count": run_failure_count,
            "fallback_count": fallback_count,
            "needs_rerun": needs_rerun,
            "artifacts": {
                "task_summary_path": str((task_dir / "task_summary.json").resolve()),
                "analysis_path": str((task_dir / "analysis.json").resolve()),
            },
        }
        summary_json["tasks"].append(task_summary)

        row = _summary_row_from_analysis(
            benchmark_name=benchmark_name,
            system_label=output_paths.system_label,
            topology=config.mas.resolved_topology(),
            agents=config.mas.total_agents,
            default_model=default_model,
            judge_model=judge_model,
            task_id=str(task.task_id),
            task_dir=task_dir,
            analysis=analysis,
        )
        row.update(
            {
                "run_failure_count": run_failure_count,
                "fallback_count": fallback_count,
                "needs_rerun": needs_rerun,
            }
        )
        summary_rows.append(row)
        _write_json(output_paths.run_root / "summary.json", summary_json)
        _write_summary_csv(output_paths.run_root / "summary.csv", summary_rows)
        _log_progress(
            f"TASK_FINISH index={task_idx + 1}/{len(tasks)} task_id={task.task_id} "
            f"task_dir={task_dir} failures={run_failure_count} fallbacks={fallback_count} "
            f"needs_rerun={needs_rerun}"
        )

    total_run_failures = sum(
        int(task.get("run_failure_count", 0) or 0)
        for task in summary_json["tasks"]
        if isinstance(task, dict)
    )
    total_fallbacks = sum(
        int(task.get("fallback_count", 0) or 0)
        for task in summary_json["tasks"]
        if isinstance(task, dict)
    )
    rerun_tasks = [
        str(task.get("task_id", ""))
        for task in summary_json["tasks"]
        if isinstance(task, dict) and bool(task.get("needs_rerun", False))
    ]
    summary_json["run_status_summary"] = {
        "run_failure_count": total_run_failures,
        "fallback_count": total_fallbacks,
        "needs_rerun_task_count": len(rerun_tasks),
        "needs_rerun_task_ids": rerun_tasks,
    }

    summary_json_path = output_paths.run_root / "summary.json"
    summary_csv_path = output_paths.run_root / "summary.csv"
    _write_json(summary_json_path, summary_json)
    _write_summary_csv(summary_csv_path, summary_rows)
    _log_progress(
        f"SUMMARY_WRITTEN summary_json={summary_json_path} summary_csv={summary_csv_path}"
    )
    _log_progress(
        "RUN_STATUS_SUMMARY "
        f"benchmark={benchmark_name} system={output_paths.system_label} "
        f"run_failures={total_run_failures} fallbacks={total_fallbacks} "
        f"needs_rerun_tasks={len(rerun_tasks)} "
        f"needs_rerun_task_ids={','.join(rerun_tasks) if rerun_tasks else 'none'}"
    )

    print(f"Run complete: {output_paths.run_root}")
    if (
        rerun_tasks
        and _env_truthy("MAS_REQUIRE_LIVE_LLM")
        and not _env_truthy("MAS_DISABLE_LIVE_LLM")
    ):
        return 2
    return 0


def list_benchmarks_command(_: argparse.Namespace) -> int:
    for name in list_benchmarks():
        print(name)
    return 0


def benchmark_info_command(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config) if args.config else None
    benchmark_cfg: dict[str, Any]
    if config is None:
        benchmark_cfg = {}
    else:
        benchmark_cfg = _benchmark_section_config(config, args.benchmark)

    benchmark = get_benchmark(args.benchmark, config=benchmark_cfg)
    info = benchmark.requirements()
    print(json.dumps(info, indent=2, sort_keys=True))
    return 0


def summarize_experiment_command(args: argparse.Namespace) -> int:
    experiment_root = Path(args.experiment_root).expanduser().resolve()
    if not experiment_root.exists():
        raise FileNotFoundError(f"Experiment root not found: {experiment_root}")

    experiment_rows: list[dict[str, Any]] = []
    experiment_manifest: dict[str, Any] = {
        "experiment_root": str(experiment_root),
        "benchmarks": [],
    }

    for benchmark_dir in sorted(path for path in experiment_root.iterdir() if path.is_dir()):
        benchmark_rows: list[dict[str, Any]] = []
        benchmark_manifest: dict[str, Any] = {
            "benchmark": benchmark_dir.name,
            "systems": [],
        }
        for system_dir in sorted(path for path in benchmark_dir.iterdir() if path.is_dir()):
            summary_json_path = system_dir / "summary.json"
            settings_path = system_dir / "experiment_settings.json"
            if not settings_path.exists():
                continue

            task_entries: list[dict[str, Any]] = []
            for task_summary_path in sorted(system_dir.glob("*/task_summary.json")):
                try:
                    payload = json.loads(task_summary_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue
                entry = _summary_task_entry_from_payload(task_summary_path.parent, payload)
                if _task_payload_needs_rerun(task_summary_path.parent, payload):
                    entry["needs_rerun"] = True
                    if int(entry.get("fallback_count", 0) or 0) == 0:
                        entry["fallback_count"] = 1
                task_entries.append(entry)

            settings = json.loads(settings_path.read_text(encoding="utf-8"))
            if summary_json_path.exists():
                summary = json.loads(summary_json_path.read_text(encoding="utf-8"))
                tasks = task_entries or list(summary.get("tasks", []))
            else:
                summary = {
                    "task_count": int(
                        settings.get("benchmark", {}).get("task_count", len(task_entries)) or 0
                    ),
                    "runs_per_task": int(settings.get("runtime", {}).get("runs_per_task", 0) or 0),
                    "tasks": task_entries,
                }
                tasks = task_entries
            scoreable_tasks = [
                task
                for task in tasks
                if isinstance(task, dict)
                and not bool(task.get("needs_rerun", False))
                and int(task.get("evaluation", {}).get("valid_count", 1) or 0) > 0
            ]
            scores = [
                float(task.get("evaluation", {}).get("avg_score", 0.0)) for task in scoreable_tasks
            ]
            success_rates = [
                float(task.get("evaluation", {}).get("success_rate", 0.0))
                for task in scoreable_tasks
            ]
            completion_rates = [
                float(task.get("evaluation", {}).get("completion_rate", 0.0))
                for task in scoreable_tasks
            ]
            run_failure_count = sum(
                int(task.get("run_failure_count", 0) or 0)
                for task in tasks
                if isinstance(task, dict)
            )
            fallback_count = sum(
                int(task.get("fallback_count", 0) or 0) for task in tasks if isinstance(task, dict)
            )
            needs_rerun_task_ids = [
                str(task.get("task_id", ""))
                for task in tasks
                if isinstance(task, dict) and bool(task.get("needs_rerun", False))
            ]

            row = {
                "benchmark": benchmark_dir.name,
                "system_label": system_dir.name,
                "topology": settings.get("system", {}).get("mas", {}).get("resolved_topology", ""),
                "agents": settings.get("system", {}).get("mas", {}).get("number_of_agents", 0),
                "default_model": settings.get("models", {}).get("default", ""),
                "judge_model": settings.get("models", {}).get(
                    "judge",
                    settings.get("models", {}).get("default", ""),
                ),
                "task_count": int(summary.get("task_count", 0)),
                "runs_per_task": int(summary.get("runs_per_task", 0)),
                "avg_task_score": _mean(scores),
                "avg_task_success_rate": _mean(success_rates),
                "avg_task_completion_rate": _mean(completion_rates),
                "completed_task_count": len(tasks),
                "scored_task_count": len(scoreable_tasks),
                "missing_task_count": max(int(summary.get("task_count", 0)) - len(tasks), 0),
                "run_failure_count": run_failure_count,
                "fallback_count": fallback_count,
                "needs_rerun_task_count": len(needs_rerun_task_ids),
                "needs_rerun_task_ids": ",".join(needs_rerun_task_ids),
                "system_root": str(system_dir.resolve()),
                "summary_json_path": str(summary_json_path.resolve()),
                "summary_csv_path": str((system_dir / "summary.csv").resolve()),
                "graph_png_path": str((system_dir / "mas_graph.png").resolve()),
            }
            benchmark_rows.append(row)
            experiment_rows.append(row)
            benchmark_manifest["systems"].append(row)

        if benchmark_rows:
            _write_json(benchmark_dir / "benchmark_summary.json", benchmark_manifest)
            _write_summary_csv(benchmark_dir / "benchmark_summary.csv", benchmark_rows)
            experiment_manifest["benchmarks"].append(benchmark_manifest)

    _write_json(experiment_root / "experiment_summary.json", experiment_manifest)
    _write_summary_csv(experiment_root / "experiment_summary.csv", experiment_rows)
    print(f"Experiment summary complete: {experiment_root}")
    total_missing = sum(int(row.get("missing_task_count", 0) or 0) for row in experiment_rows)
    total_run_failures = sum(int(row.get("run_failure_count", 0) or 0) for row in experiment_rows)
    total_fallbacks = sum(int(row.get("fallback_count", 0) or 0) for row in experiment_rows)
    total_rerun_tasks = sum(
        int(row.get("needs_rerun_task_count", 0) or 0) for row in experiment_rows
    )
    print(
        "Experiment run-status summary: "
        f"missing_tasks={total_missing} "
        f"run_failures={total_run_failures} "
        f"fallbacks={total_fallbacks} "
        f"needs_rerun_tasks={total_rerun_tasks}"
    )
    for row in experiment_rows:
        if (
            int(row.get("missing_task_count", 0) or 0) > 0
            or int(row.get("run_failure_count", 0) or 0) > 0
            or int(row.get("fallback_count", 0) or 0) > 0
            or int(row.get("needs_rerun_task_count", 0) or 0) > 0
        ):
            print(
                "  - "
                f"{row.get('benchmark')}::{row.get('system_label')} "
                f"completed={row.get('completed_task_count')}/{row.get('task_count')} "
                f"missing={row.get('missing_task_count')} "
                f"run_failures={row.get('run_failure_count')} "
                f"fallbacks={row.get('fallback_count')} "
                f"needs_rerun={row.get('needs_rerun_task_count')}"
            )
    return 0
