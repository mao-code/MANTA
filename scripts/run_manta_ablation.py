#!/usr/bin/env python3
"""Run one seed-42 sample for each MANTA variant on 30 tasks per benchmark.

The driver deliberately executes one task per ``main.py run`` process. Completed
task artifacts are the checkpoint: restarting this command skips them, reconciles
their process-only playbook candidates, and continues with the next task.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TASKS_PER_BENCHMARK = 30
RUNS_PER_TASK = 1
DEFAULT_MANIFEST = ROOT / "config" / "manta_ablation_tasks_30_seed42.json"
STABLETOOLBENCH_MANIFEST = (
    ROOT / "config" / "manta_ablation_stabletoolbench_tasks_30_seed42.json"
)
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "full_experiment"
DEFAULT_EXPERIMENT_ID = "manta_ablation_30_seed42_batch10"
DEFAULT_MODEL = "google/gemma-4-31b-it:nitro"
DEFAULT_BENCHMARKS = ("browsecomp", "workbench", "plancraft")
SEED_SKILL = ROOT / "config" / "topology_skill.md"
PLAYBOOK = ROOT / "config" / "topology_playbook.json"
REFLECTION_BATCH_SIZE = 10
STABLETOOLBENCH_URL = "http://127.0.0.1:8080/virtual"


@dataclass(frozen=True)
class Variant:
    slug: str
    description: str
    initial_planner_mode: str = "task_conditioned"
    max_turns: int = 2
    repair_budget: int = 1
    playbook_read: bool = True
    online_reflection: bool = True


VARIANTS = (
    Variant("full", "Full MANTA"),
    Variant(
        "no_initial_planner",
        "Fixed coordinator-worker initial topology; MANTA repair retained",
        initial_planner_mode="fixed",
    ),
    Variant(
        "no_online_adaptation",
        "Task-conditioned initial topology with repair disabled",
        max_turns=1,
        repair_budget=0,
    ),
    Variant(
        "frozen_long_term",
        "Read the initial long-term playbook without updating it",
        online_reflection=False,
    ),
    Variant(
        "no_learned_playbook",
        "Run without learned long-term playbook content",
        playbook_read=False,
        online_reflection=False,
    ),
)

BENCHMARK_TOML = {
    "browsecomp": """
[browsecomp]
decrypted_path = "benchmark/browsecomp/data/browsecomp_plus_decrypted.jsonl"
qrel_evidence_path = "benchmark/browsecomp/topics-qrels/qrel_evidence.txt"
qrel_golds_path = "benchmark/browsecomp/topics-qrels/qrel_golds.txt"
auto_download = true
eval_mode = "substring"
enable_tools = true
tool_k = 5
include_get_document = true
tool_snippet_max_tokens = 160
max_tool_iterations = 6
""",
    "workbench": """
[workbench]
domain = "multi_domain"
tool_selection = "domains"
max_tool_iterations = 20
tasks_version = "v1"
""",
    "plancraft": """
[plancraft]
split = "val.small"
max_steps = 30
resolution = "high"
""",
    "stabletoolbench": f"""
[stabletoolbench]
auto_download = true
auto_download_server_assets = true
task_sets = ["G1_instruction"]
virtual_server_url = "{STABLETOOLBENCH_URL}"
enable_tools = true
max_tool_iterations = 8
skip_missing_cache_tasks = true
eval_mode = "fac"
judge_model = "gpt-4.1-mini"
""",
}

_STOP_REQUESTED = False
_CURRENT_CHILD: subprocess.Popen[str] | None = None
_STABLETOOLBENCH_SERVER: subprocess.Popen[str] | None = None


def _json_string(value: str | Path) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_benchmarks(args: argparse.Namespace) -> tuple[str, ...]:
    selected = tuple(
        part.strip() for part in str(args.benchmarks or "").split(",") if part.strip()
    )
    if not selected:
        selected = DEFAULT_BENCHMARKS
    duplicates = sorted({name for name in selected if selected.count(name) > 1})
    invalid = sorted(set(selected) - set(BENCHMARK_TOML))
    if duplicates or invalid:
        raise ValueError(
            f"Invalid --benchmarks selection; duplicates={duplicates}, invalid={invalid}, "
            f"supported={sorted(BENCHMARK_TOML)}"
        )
    return selected


def _load_manifest(path: Path, selected_benchmarks: tuple[str, ...]) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict) or int(payload.get("seed", -1)) != 42:
        raise ValueError(f"Invalid seed-42 task manifest: {path}")
    benchmarks = payload.get("benchmarks")
    order = payload.get("execution_order")
    if not isinstance(benchmarks, dict) or not isinstance(order, list):
        raise ValueError(f"Manifest is missing benchmarks or execution_order: {path}")
    for benchmark in selected_benchmarks:
        task_ids = benchmarks.get(benchmark, {}).get("task_ids", [])
        if len(task_ids) != TASKS_PER_BENCHMARK or len(set(map(str, task_ids))) != (
            TASKS_PER_BENCHMARK
        ):
            raise ValueError(
                f"Manifest must contain {TASKS_PER_BENCHMARK} unique {benchmark} task ids"
            )
    pairs = [
        (str(item.get("benchmark", "")), str(item.get("task_id", "")))
        for item in order
        if isinstance(item, dict) and str(item.get("benchmark", "")) in selected_benchmarks
    ]
    expected = {
        (benchmark, str(task_id))
        for benchmark, entry in benchmarks.items()
        if benchmark in selected_benchmarks
        for task_id in entry.get("task_ids", [])
    }
    expected_task_count = len(selected_benchmarks) * TASKS_PER_BENCHMARK
    if len(pairs) != expected_task_count or set(pairs) != expected:
        raise ValueError(
            "Manifest execution_order must contain each selected task exactly once "
            f"({expected_task_count} tasks total)"
        )
    return payload


def _experiment_root(args: argparse.Namespace) -> Path:
    return Path(args.output_root).expanduser().resolve() / str(args.experiment_id)


def _state_root(args: argparse.Namespace) -> Path:
    return _experiment_root(args) / "_ablation_state"


def _variant_skill(state_root: Path, variant: Variant) -> Path:
    return state_root / "skills" / variant.slug / "topology_skill.md"


def _config_path(state_root: Path, variant: Variant, benchmark: str) -> Path:
    return state_root / "configs" / variant.slug / f"{benchmark}.toml"


def _variant_config(
    *,
    variant: Variant,
    benchmark: str,
    model: str,
    skill_path: Path,
) -> str:
    return f"""[openrouter]
api_key = ""
base_url = "https://openrouter.ai/api/v1"
timeout_s = 600

[experiment]
output_dir = "artifacts/benchmark_traces/manta_ablation"
runs_per_task = {RUNS_PER_TASK}
seed = 42

[models]
default = {_json_string(model)}

[mas]
levels = 1
intra_level_link_ratio = 1.0
full_linked = true
topology = "self_evolved"
number_of_agents = 5
agent_types = ["general"]
communication_count_internally = 2
turn_mode = "multi_turn"
max_turns = {variant.max_turns}
discussion_rounds = 1
minimum_discussion_rounds = 1
termination_consensus_mode = "llm_judge"
final_vote_mode = "llm_judge"
peer_artifact_max_chars = 0
enable_dynamic_roles = true

[self_evolved]
harness_backend = "openrouter"
initial_planner_mode = {_json_string(variant.initial_planner_mode)}
max_initial_agents = 5
max_total_agents = 10
max_turns = {variant.max_turns}
repair_budget = {variant.repair_budget}
audit_mode = "hybrid"
playbook_path = {_json_string(PLAYBOOK.resolve())}
skill_path = {_json_string(skill_path.resolve())}
playbook_read = {str(variant.playbook_read).lower()}
# Online reflection is coordinated by this resumable driver, not main.py.
skill_update_batch_size = 0
default_packet_max_chars = 0
{BENCHMARK_TOML[benchmark].strip()}
"""


def prepare(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    manifest_path = Path(args.manifest).expanduser().resolve()
    selected_benchmarks = _selected_benchmarks(args)
    manifest = _load_manifest(manifest_path, selected_benchmarks)
    experiment_root = _experiment_root(args)
    state_root = _state_root(args)
    state_root.mkdir(parents=True, exist_ok=True)

    run_manifest_path = state_root / "experiment_manifest.json"
    existing = _read_json(run_manifest_path)
    requested = {
        "experiment_id": str(args.experiment_id),
        "seed": 42,
        "runs_per_task": RUNS_PER_TASK,
        "model": str(args.model),
        "task_manifest_path": str(manifest_path),
        "task_manifest_sha256": _sha256(manifest_path),
        "seed_skill_path": str(SEED_SKILL.resolve()),
        "seed_skill_sha256": _sha256(SEED_SKILL),
        "reflection_batch_size": REFLECTION_BATCH_SIZE,
        "benchmarks": list(selected_benchmarks),
        "variants": [
            {
                "slug": variant.slug,
                "description": variant.description,
                "initial_planner_mode": variant.initial_planner_mode,
                "max_turns": variant.max_turns,
                "repair_budget": variant.repair_budget,
                "playbook_read": variant.playbook_read,
                "online_reflection": variant.online_reflection,
            }
            for variant in VARIANTS
        ],
    }
    if isinstance(existing, dict):
        if "benchmarks" not in existing and selected_benchmarks == DEFAULT_BENCHMARKS:
            requested.pop("benchmarks")
        # The experiment uses isolated skill copies after preparation. A different
        # experiment may legitimately update the repository skill while this run is
        # paused, so resume against the seed hash recorded at creation time.
        requested["seed_skill_path"] = existing.get("seed_skill_path")
        requested["seed_skill_sha256"] = existing.get("seed_skill_sha256")
    if existing is not None and existing != requested:
        raise RuntimeError(
            f"Existing experiment settings differ from this command: {run_manifest_path}. "
            "Use the original arguments or a new --experiment-id."
        )
    if existing is None:
        _write_json(run_manifest_path, requested)

    for variant in VARIANTS:
        skill_path = _variant_skill(state_root, variant)
        if not skill_path.exists():
            skill_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SEED_SKILL, skill_path)
        for benchmark in selected_benchmarks:
            path = _config_path(state_root, variant, benchmark)
            content = _variant_config(
                variant=variant,
                benchmark=benchmark,
                model=str(args.model),
                skill_path=skill_path,
            )
            if path.exists() and path.read_text(encoding="utf-8") != content:
                raise RuntimeError(
                    f"Generated config changed for an existing experiment: {path}. "
                    "Use a new --experiment-id."
                )
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")

    experiment_root.mkdir(parents=True, exist_ok=True)
    return manifest, state_root


def _task_root(args: argparse.Namespace, variant: Variant, benchmark: str, task_id: str) -> Path:
    return _experiment_root(args) / benchmark / variant.slug / task_id


def _task_complete(
    args: argparse.Namespace, variant: Variant, benchmark: str, task_id: str
) -> bool:
    root = _task_root(args, variant, benchmark, task_id)
    summary = _read_json(root / "task_summary.json")
    metadata = _read_json(root / "run_0.metadata.json")
    evaluation = _read_json(root / "run_0.eval.json")
    return bool(
        isinstance(summary, dict)
        and (bool(args.allow_mock) or not bool(summary.get("needs_rerun", True)))
        and isinstance(metadata, dict)
        and str(metadata.get("run_status", "completed")) == "completed"
        and isinstance(evaluation, dict)
        and evaluation.get("success") is not None
    )


def _reflection_state_path(state_root: Path, variant: Variant) -> Path:
    return state_root / "reflection" / f"{variant.slug}.json"


def _load_reflection_state(state_root: Path, variant: Variant) -> dict[str, Any]:
    state = _read_json(_reflection_state_path(state_root, variant), default={})
    if not isinstance(state, dict):
        state = {}
    state.setdefault("seen_run_keys", [])
    state.setdefault("pending", [])
    state.setdefault("updates", [])
    state.setdefault("inflight", None)
    # Used only by self_evolved.reflection_mode = "evidence_gated": the summaries of all
    # reflected runs, and the supported findings behind the last applied update.
    state.setdefault("history", [])
    state.setdefault("findings", None)
    return state


def _save_reflection_state(state_root: Path, variant: Variant, state: dict[str, Any]) -> None:
    _write_json(_reflection_state_path(state_root, variant), state)


def _record_candidate(
    *,
    args: argparse.Namespace,
    state_root: Path,
    variant: Variant,
    benchmark: str,
    task_id: str,
) -> None:
    if not variant.online_reflection:
        return
    from MAS.self_evolved.skill import summary_from_candidate

    run_key = f"{benchmark}/{task_id}/run_0"
    state = _load_reflection_state(state_root, variant)
    if run_key in set(map(str, state["seen_run_keys"])):
        return
    metadata = _read_json(_task_root(args, variant, benchmark, task_id) / "run_0.metadata.json")
    candidate = ((metadata or {}).get("self_evolved") or {}).get("playbook_update_candidate")
    if not isinstance(candidate, dict):
        raise RuntimeError(f"Completed run has no playbook candidate: {run_key}")
    state["seen_run_keys"].append(run_key)
    state["pending"].append(
        {
            "run_key": run_key,
            "summary": summary_from_candidate(candidate),
        }
    )
    _save_reflection_state(state_root, variant, state)


def _resume_inflight_reflection(
    *, state_root: Path, variant: Variant, state: dict[str, Any]
) -> bool:
    inflight = state.get("inflight")
    if not isinstance(inflight, dict):
        return False
    skill_path = _variant_skill(state_root, variant)
    before_sha = str(inflight.get("skill_before_sha256", ""))
    if _sha256(skill_path) == before_sha:
        return False
    state["updates"].append(
        {
            "run_keys": list(inflight.get("run_keys", [])),
            "changed": True,
            "reason": "recovered_after_skill_write",
            "skill_sha256": _sha256(skill_path),
        }
    )
    state["history"] = list(state["history"]) + list(inflight.get("summaries", []))
    state["findings"] = None
    state["inflight"] = None
    _save_reflection_state(state_root, variant, state)
    return True


def _reflect_pending(
    *,
    args: argparse.Namespace,
    state_root: Path,
    variant: Variant,
) -> None:
    if not variant.online_reflection:
        return
    from MAS.config import load_experiment_config
    from MAS.llm import OpenRouterLLMClient
    from MAS.self_evolved.skill import SkillReflector, TopologySkill

    state = _load_reflection_state(state_root, variant)
    _resume_inflight_reflection(state_root=state_root, variant=variant, state=state)
    state = _load_reflection_state(state_root, variant)
    while len(state["pending"]) >= REFLECTION_BATCH_SIZE:
        batch = list(state["pending"][:REFLECTION_BATCH_SIZE])
        state["pending"] = list(state["pending"][REFLECTION_BATCH_SIZE:])
        skill_path = _variant_skill(state_root, variant)
        state["inflight"] = {
            "run_keys": [str(row["run_key"]) for row in batch],
            "summaries": [dict(row["summary"]) for row in batch],
            "skill_before_sha256": _sha256(skill_path),
        }
        _save_reflection_state(state_root, variant, state)

        benchmark = str(batch[0]["run_key"]).split("/", 1)[0]
        config = load_experiment_config(_config_path(state_root, variant, benchmark))
        client = OpenRouterLLMClient(config.openrouter, config.models)
        reflector = SkillReflector(client, config.self_evolved)
        skill = TopologySkill.load(skill_path)
        summaries = [dict(row["summary"]) for row in batch]
        result = reflector.reflect(
            current_skill=skill.text,
            run_summaries=summaries,
            history=list(state["history"]) + summaries,
            previous_findings=state["findings"],
        )
        state["history"] = list(state["history"]) + summaries
        if result.changed:
            state["findings"] = list(result.findings)
            skill.save(result.skill_markdown)
        state["updates"].append(
            {
                "run_keys": [str(row["run_key"]) for row in batch],
                "changed": bool(result.changed),
                "reason": str(result.reason),
                "skill_sha256": _sha256(skill_path),
                "llm": dict(result.llm),
            }
        )
        state["inflight"] = None
        _save_reflection_state(state_root, variant, state)
        print(
            f"[reflection] variant={variant.slug} batch={len(batch)} "
            f"changed={result.changed} reason={result.reason}",
            flush=True,
        )


def _task_command(
    *,
    args: argparse.Namespace,
    state_root: Path,
    variant: Variant,
    benchmark: str,
    task_ids: list[str],
) -> list[str]:
    python = str(ROOT / ".venv" / "bin" / "python")
    if not Path(python).exists():
        python = sys.executable
    return [
        python,
        "main.py",
        "run",
        "--config",
        str(_config_path(state_root, variant, benchmark)),
        "--benchmark",
        benchmark,
        "--output-dir",
        str(Path(args.output_root).expanduser().resolve()),
        "--output-layout",
        "hierarchical",
        "--experiment-id",
        str(args.experiment_id),
        "--system-label",
        variant.slug,
        "--topology",
        "self_evolved",
        "--agents",
        "5",
        "--mas-rounds",
        str(variant.max_turns),
        "--discussion-rounds",
        "1",
        "--communication-budget",
        "2",
        "--task-ids",
        ",".join(task_ids),
        "--runs-per-task",
        str(RUNS_PER_TASK),
        "--seed",
        "42",
    ]


def _run_child(command: list[str], log_path: Path, *, allow_mock: bool) -> int:
    global _CURRENT_CHILD
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if allow_mock:
        env.pop("MAS_REQUIRE_LIVE_LLM", None)
    else:
        env["MAS_REQUIRE_LIVE_LLM"] = "1"
        env.pop("MAS_DISABLE_LIVE_LLM", None)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(command)}\n")
        log.flush()
        _CURRENT_CHILD = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        started = time.monotonic()
        next_update = started + 60
        while _CURRENT_CHILD.poll() is None:
            if _STOP_REQUESTED:
                os.killpg(_CURRENT_CHILD.pid, signal.SIGTERM)
            now = time.monotonic()
            if now >= next_update:
                print(
                    f"[running] pid={_CURRENT_CHILD.pid} elapsed_min={(now - started) / 60:.1f} "
                    f"log={log_path}",
                    flush=True,
                )
                next_update = now + 60
            time.sleep(1)
        returncode = int(_CURRENT_CHILD.returncode or 0)
        _CURRENT_CHILD = None
        return returncode


def _signal_handler(signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"\n[stop] received signal={signum}; stopping after child shutdown", flush=True)
    if _CURRENT_CHILD is not None and _CURRENT_CHILD.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(_CURRENT_CHILD.pid, signal.SIGTERM)


def _stabletoolbench_healthcheck() -> bool:
    health_url = STABLETOOLBENCH_URL.rsplit("/virtual", 1)[0] + "/healthz"
    try:
        with urllib.request.urlopen(health_url, timeout=2) as response:  # noqa: S310
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def _start_stabletoolbench_server(state_root: Path) -> None:
    global _STABLETOOLBENCH_SERVER
    if _stabletoolbench_healthcheck():
        print(f"[stabletoolbench] using server at {STABLETOOLBENCH_URL}", flush=True)
        return

    cache_root = ROOT / "benchmark" / "stabletoolbench" / "tool_response_cache"
    if not cache_root.exists():
        raise FileNotFoundError(
            f"StableToolBench cache is missing: {cache_root}. "
            "Run the StableToolBench setup before starting the ablation."
        )

    log_path = state_root / "logs" / "stabletoolbench_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as server_log:
        _STABLETOOLBENCH_SERVER = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "scripts" / "stabletoolbench_virtual_server.py"),
                "--host",
                "127.0.0.1",
                "--port",
                "8080",
                "--path",
                "/virtual",
                "--cache-root",
                str(cache_root),
            ],
            cwd=ROOT,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if _STABLETOOLBENCH_SERVER.poll() is not None:
            raise RuntimeError(f"StableToolBench server exited; see {log_path}")
        if _stabletoolbench_healthcheck():
            print(f"[stabletoolbench] started server at {STABLETOOLBENCH_URL}", flush=True)
            return
        time.sleep(0.25)
    _stop_stabletoolbench_server()
    raise RuntimeError(f"StableToolBench server did not become ready; see {log_path}")


def _stop_stabletoolbench_server() -> None:
    global _STABLETOOLBENCH_SERVER
    server = _STABLETOOLBENCH_SERVER
    _STABLETOOLBENCH_SERVER = None
    if server is None or server.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(server.pid, signal.SIGTERM)
    try:
        server.wait(timeout=10)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(server.pid, signal.SIGKILL)
        server.wait(timeout=5)


def _finalize_summaries(
    *,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    state_root: Path,
    variant: Variant,
    selected_benchmarks: tuple[str, ...],
) -> None:
    for benchmark in selected_benchmarks:
        task_ids = [str(value) for value in manifest["benchmarks"][benchmark]["task_ids"]]
        command = _task_command(
            args=args,
            state_root=state_root,
            variant=variant,
            benchmark=benchmark,
            task_ids=task_ids,
        )
        log_path = state_root / "logs" / variant.slug / f"finalize_{benchmark}.log"
        code = _run_child(command, log_path, allow_mock=bool(args.allow_mock))
        if code != 0:
            raise RuntimeError(
                f"Summary finalization failed for {variant.slug}/{benchmark}; see {log_path}"
            )


def run(args: argparse.Namespace) -> int:
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    manifest, state_root = prepare(args)
    selected_benchmarks = _selected_benchmarks(args)
    execution_order = [
        item
        for item in manifest["execution_order"]
        if str(item["benchmark"]) in selected_benchmarks
    ]
    pid_path = state_root / "runner.pid"
    if pid_path.exists():
        try:
            existing_pid = int(pid_path.read_text(encoding="utf-8").strip())
            os.kill(existing_pid, 0)
        except (OSError, ValueError):
            pid_path.unlink(missing_ok=True)
        else:
            raise RuntimeError(f"Ablation runner is already active with pid={existing_pid}")
    pid_path.write_text(str(os.getpid()), encoding="utf-8")

    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    for signum in previous_handlers:
        signal.signal(signum, _signal_handler)

    try:
        if "stabletoolbench" in selected_benchmarks:
            _start_stabletoolbench_server(state_root)
        total = RUNS_PER_TASK * len(VARIANTS) * len(execution_order)
        completed_at_start = sum(
            _task_complete(
                args,
                variant,
                str(item["benchmark"]),
                str(item["task_id"]),
            )
            for variant in VARIANTS
            for item in execution_order
        )
        print(
            f"[start] experiment={_experiment_root(args)} completed={completed_at_start}/{total}",
            flush=True,
        )
        ordinal = 0
        for variant in VARIANTS:
            print(f"\n[variant] {variant.slug}: {variant.description}", flush=True)
            for item in execution_order:
                ordinal += 1
                benchmark = str(item["benchmark"])
                task_id = str(item["task_id"])
                if _task_complete(args, variant, benchmark, task_id):
                    _record_candidate(
                        args=args,
                        state_root=state_root,
                        variant=variant,
                        benchmark=benchmark,
                        task_id=task_id,
                    )
                    _reflect_pending(args=args, state_root=state_root, variant=variant)
                    print(
                        f"[skip] {ordinal}/{total} {variant.slug}/{benchmark}/{task_id}",
                        flush=True,
                    )
                    continue
                if _STOP_REQUESTED:
                    return 130
                command = _task_command(
                    args=args,
                    state_root=state_root,
                    variant=variant,
                    benchmark=benchmark,
                    task_ids=[task_id],
                )
                log_path = (
                    state_root / "logs" / variant.slug / f"{ordinal:03d}_{benchmark}_{task_id}.log"
                )
                print(
                    f"[task] {ordinal}/{total} {variant.slug}/{benchmark}/{task_id} log={log_path}",
                    flush=True,
                )
                code = _run_child(command, log_path, allow_mock=bool(args.allow_mock))
                if _STOP_REQUESTED:
                    return 130
                if code != 0 or not _task_complete(args, variant, benchmark, task_id):
                    raise RuntimeError(
                        f"Task failed: {variant.slug}/{benchmark}/{task_id}; see {log_path}"
                    )
                _record_candidate(
                    args=args,
                    state_root=state_root,
                    variant=variant,
                    benchmark=benchmark,
                    task_id=task_id,
                )
                _reflect_pending(args=args, state_root=state_root, variant=variant)

            if _STOP_REQUESTED:
                return 130
            _finalize_summaries(
                args=args,
                manifest=manifest,
                state_root=state_root,
                variant=variant,
                selected_benchmarks=selected_benchmarks,
            )

        summary_log = state_root / "logs" / "summarize.log"
        summary_command = [
            str(ROOT / ".venv" / "bin" / "python"),
            "main.py",
            "summarize-experiment",
            "--experiment-root",
            str(_experiment_root(args)),
        ]
        code = _run_child(summary_command, summary_log, allow_mock=bool(args.allow_mock))
        if code != 0:
            raise RuntimeError(f"Experiment summarization failed; see {summary_log}")
        print(f"[complete] {_experiment_root(args)}", flush=True)
        return 0
    finally:
        _stop_stabletoolbench_server()
        pid_path.unlink(missing_ok=True)
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)


def status(args: argparse.Namespace) -> int:
    manifest, state_root = prepare(args)
    selected_benchmarks = _selected_benchmarks(args)
    execution_order = [
        item
        for item in manifest["execution_order"]
        if str(item["benchmark"]) in selected_benchmarks
    ]
    total_per_variant = len(execution_order)
    for variant in VARIANTS:
        completed = sum(
            _task_complete(args, variant, str(item["benchmark"]), str(item["task_id"]))
            for item in execution_order
        )
        reflection = _load_reflection_state(state_root, variant)
        print(
            f"{variant.slug:22s} {completed:2d}/{total_per_variant} tasks "
            f"reflection_updates={len(reflection['updates'])} "
            f"pending={len(reflection['pending'])}"
        )
    pid_path = state_root / "runner.pid"
    print(f"runner_pid={pid_path.read_text().strip() if pid_path.exists() else 'not running'}")
    print(f"experiment_root={_experiment_root(args)}")
    return 0


def stop(args: argparse.Namespace) -> int:
    pid_path = _state_root(args) / "runner.pid"
    if not pid_path.exists():
        print("Ablation runner is not running.")
        return 0
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        os.kill(pid, signal.SIGTERM)
    except (OSError, ValueError):
        pid_path.unlink(missing_ok=True)
        print("Removed a stale runner PID file.")
        return 0
    print(f"Stop requested for ablation runner pid={pid}.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "run", "status", "stop"),
        help="Prepare files, run/resume, show progress, or request a clean stop.",
    )
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--benchmarks",
        default=",".join(DEFAULT_BENCHMARKS),
        help=(
            "Comma-separated benchmark subset. "
            f"Supported: {','.join(BENCHMARK_TOML)}."
        ),
    )
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help="Permit deterministic mock LLM calls. Never use this for paper results.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        manifest, state_root = prepare(args)
        print(f"Prepared {len(manifest['execution_order'])} tasks under {state_root}")
        return 0
    if args.command == "run":
        return run(args)
    if args.command == "status":
        return status(args)
    return stop(args)


if __name__ == "__main__":
    raise SystemExit(main())
