from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib

from .relay import auto_topology_for_agents, normalize_topology_name

load_dotenv()  # auto-load .env (OPENROUTER_API_KEY, HF_TOKEN, etc.)


@dataclass
class OpenRouterConfig:
    api_key: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    http_referer: str | None = None
    x_title: str | None = None
    timeout_s: float = 600.0

    def validate(self) -> None:
        if not self.base_url:
            raise ValueError("openrouter.base_url must be a non-empty string")
        if self.timeout_s <= 0:
            raise ValueError("openrouter.timeout_s must be > 0")


@dataclass
class MASConfig:
    levels: int
    intra_level_link_ratio: float
    full_linked: bool
    topology: str = "auto"
    number_of_agents: int | None = None
    agents_per_level: list[int] | None = None
    group_sizes: list[int] | None = None
    agent_types: list[str] = field(default_factory=lambda: ["general"])
    communication_count_internally: int = 1
    turn_mode: str = "single_turn"
    max_turns: int = 20
    discussion_rounds: int = 1
    minimum_discussion_rounds: int = 1
    termination_consensus_mode: str = "llm_judge"
    final_vote_mode: str = "llm_judge"
    peer_artifact_max_chars: int = 0
    enable_dynamic_roles: bool = True

    def validate(self) -> None:
        if self.levels < 1:
            raise ValueError("mas.levels must be >= 1")

        if not 0.0 <= self.intra_level_link_ratio <= 1.0:
            raise ValueError("mas.intra_level_link_ratio must be between 0.0 and 1.0")

        if self.communication_count_internally < 0:
            raise ValueError("mas.communication_count_internally must be >= 0")

        if self.turn_mode not in {"single_turn", "multi_turn"}:
            raise ValueError("mas.turn_mode must be one of: single_turn, multi_turn")

        if self.max_turns < 1:
            raise ValueError("mas.max_turns must be >= 1")

        if self.discussion_rounds < 1:
            raise ValueError("mas.discussion_rounds must be >= 1")

        if self.minimum_discussion_rounds < 0:
            raise ValueError("mas.minimum_discussion_rounds must be >= 0")

        if self.termination_consensus_mode not in {"llm_judge", "lexical"}:
            raise ValueError("mas.termination_consensus_mode must be one of: llm_judge, lexical")

        if self.final_vote_mode not in {"llm_judge", "deterministic"}:
            raise ValueError("mas.final_vote_mode must be one of: llm_judge, deterministic")

        if self.peer_artifact_max_chars < 0:
            raise ValueError("mas.peer_artifact_max_chars must be >= 0 (0 = unlimited)")

        if self.agents_per_level is not None:
            if len(self.agents_per_level) != self.levels:
                raise ValueError(
                    "mas.agents_per_level length must equal mas.levels "
                    f"({len(self.agents_per_level)} != {self.levels})"
                )
            for idx, count in enumerate(self.agents_per_level):
                if count < 1:
                    raise ValueError(f"mas.agents_per_level[{idx}] must be >= 1 (got {count})")

        if self.number_of_agents is not None and self.number_of_agents < 1:
            raise ValueError("mas.number_of_agents must be >= 1 when provided")

        if self.agents_per_level is None and self.number_of_agents is None:
            raise ValueError("Either mas.number_of_agents or mas.agents_per_level must be provided")

        if self.agents_per_level is not None and self.number_of_agents is not None:
            total = sum(self.agents_per_level)
            if total != self.number_of_agents:
                raise ValueError(
                    "mas.number_of_agents does not match sum(mas.agents_per_level) "
                    f"({self.number_of_agents} != {total})"
                )

        if not self.agent_types:
            raise ValueError("mas.agent_types must contain at least one agent type")
        if any(not item.strip() for item in self.agent_types):
            raise ValueError("mas.agent_types entries must be non-empty strings")

        normalized_topology = normalize_topology_name(self.topology)
        if normalized_topology == "auto" and self.total_agents == 1:
            self.topology = auto_topology_for_agents(self.total_agents)
        elif normalized_topology != "auto":
            self.topology = normalized_topology

        if self.group_sizes is not None:
            if not self.group_sizes:
                raise ValueError("mas.group_sizes must not be empty")
            if any(int(item) < 1 for item in self.group_sizes):
                raise ValueError("mas.group_sizes entries must be >= 1")
            if sum(int(item) for item in self.group_sizes) != self.total_agents:
                raise ValueError(
                    "sum(mas.group_sizes) must match total agents "
                    f"({sum(int(item) for item in self.group_sizes)} != {self.total_agents})"
                )

    def resolved_topology(self) -> str:
        normalized = normalize_topology_name(self.topology)
        if normalized == "auto":
            return auto_topology_for_agents(self.total_agents)
        return normalized

    @property
    def total_agents(self) -> int:
        if self.agents_per_level is not None:
            return int(sum(self.agents_per_level))
        assert self.number_of_agents is not None
        return int(self.number_of_agents)

    def resolved_agents_per_level(self) -> list[int]:
        if self.agents_per_level is not None:
            return list(self.agents_per_level)

        assert self.number_of_agents is not None
        base = self.number_of_agents // self.levels
        remainder = self.number_of_agents % self.levels
        per_level = [base] * self.levels
        for idx in range(remainder):
            per_level[idx] += 1
        for idx, count in enumerate(per_level):
            if count < 1:
                raise ValueError(
                    "Derived mas.agents_per_level contains a zero-sized level at index "
                    f"{idx}. Increase mas.number_of_agents or lower mas.levels."
                )
        return per_level


@dataclass
class SelfEvolvedConfig:
    """Settings for the self-evolved topology system (topology = "self_evolved")."""

    harness_backend: str = "openrouter"
    # ``fixed`` skips the LLM initial-planning call and uses the deterministic
    # coordinator-worker fallback while leaving auditing and repair unchanged.
    # This is an ablation-only control for task-conditioned topology planning.
    initial_planner_mode: str = "task_conditioned"
    max_initial_agents: int = 5
    max_total_agents: int = 10
    # Outer orchestration (EXECUTE -> AUDIT -> CONTROL -> MUTATE) turns: one initial
    # execution of the planned topology plus up to (max_turns - 1) repair re-executions.
    # The controller stops early on a decision-grade answer / stall, so this is a ceiling,
    # not a target. Retrieval tasks are force-capped to 1 turn in the engine (OOM guard).
    max_turns: int = 5
    # Trace-backed repair mutations allowed per run. Each repair also consumes a turn,
    # so the effective count is min(repair_budget, max_turns - 1). 0 disables repair.
    repair_budget: int = 4
    # Hybrid keeps deterministic failure checks as a safety floor and adds a
    # trace-grounded open-set model observer. ``llm_judge`` is retained for
    # experiments that intentionally allow the model to veto heuristic repairs.
    audit_mode: str = "hybrid"
    playbook_path: str = "config/topology_playbook.json"
    # Agent-maintained markdown skill (the long-term playbook as a SKILL.md). When this
    # file exists it is the planner's primary long-term memory (loaded in full at plan
    # time); the JSON playbook above is the deterministic fallback when it is absent.
    skill_path: str = "config/topology_skill.md"
    playbook_read: bool = True
    # Online (in-experiment) skill learning, in runs-per-update. Default 12 (≈ 4
    # tasks at the standard 3 runs/task): a single `run` command pauses every 12
    # freshly executed runs, reflects those outcomes into the skill, and reloads it
    # for the rest of the experiment — true online self-evolution. Because this writes
    # the shared skill file mid-experiment, run ONE sequential process; set 0 for
    # parallel experiments (post-hoc reflection only, via
    # scripts/reflect_topology_skill.py).
    skill_update_batch_size: int = 12
    default_packet_max_chars: int = 0  # 0 = full fidelity; optional generous structural budget
    # How runs with state-changing tools are protected from duplicate writes.
    # "transactional_star": contract every topology into readers -> one committer star.
    # "commit_once": keep the planned topology; write calls are validated and recorded as
    # proposals, and the selected answer's proposals execute once after finalize.
    write_protocol: str = "transactional_star"
    # How the reflection agent turns run outcomes into lessons.
    # "batch": the agent reads the latest batch's raw process counts (original behaviour).
    # "evidence_gated": the agent reads all runs so far, and may state only topology
    # comparisons that pass a significance test computed in code (skill.supported_findings).
    reflection_mode: str = "batch"

    def validate(self) -> None:
        if self.harness_backend not in {"openrouter", "claude_agent_sdk"}:
            raise ValueError(
                "self_evolved.harness_backend must be one of: openrouter, claude_agent_sdk"
            )
        if self.initial_planner_mode not in {"task_conditioned", "fixed"}:
            raise ValueError(
                "self_evolved.initial_planner_mode must be one of: task_conditioned, fixed"
            )
        if self.max_initial_agents < 1:
            raise ValueError("self_evolved.max_initial_agents must be >= 1")
        if self.max_total_agents < self.max_initial_agents:
            raise ValueError(
                "self_evolved.max_total_agents must be >= self_evolved.max_initial_agents"
            )
        if not 1 <= self.max_turns <= 10:
            raise ValueError(
                "self_evolved.max_turns must be between 1 and 10 (one initial turn plus "
                "up to max_turns - 1 repair turns)"
            )
        if self.repair_budget < 0:
            raise ValueError("self_evolved.repair_budget must be >= 0")
        if self.audit_mode not in {"heuristic", "hybrid", "llm_judge"}:
            raise ValueError("self_evolved.audit_mode must be one of: heuristic, hybrid, llm_judge")
        if self.skill_update_batch_size < 0:
            raise ValueError("self_evolved.skill_update_batch_size must be >= 0")
        if self.default_packet_max_chars < 0:
            raise ValueError("self_evolved.default_packet_max_chars must be >= 0")
        if self.write_protocol not in {"transactional_star", "commit_once"}:
            raise ValueError(
                "self_evolved.write_protocol must be one of: transactional_star, commit_once"
            )
        if self.reflection_mode not in {"batch", "evidence_gated"}:
            raise ValueError("self_evolved.reflection_mode must be one of: batch, evidence_gated")


@dataclass
class ExperimentRuntimeConfig:
    output_dir: str = "outputs"
    runs_per_task: int = 3
    seed: int = 42
    task_limit: int | None = None

    def validate(self) -> None:
        if self.runs_per_task < 1:
            raise ValueError("experiment.runs_per_task must be >= 1")
        if self.seed < 0:
            raise ValueError("experiment.seed must be >= 0")
        if self.task_limit is not None and self.task_limit < 1:
            raise ValueError("experiment.task_limit must be >= 1 when provided")


@dataclass
class ExperimentConfig:
    openrouter: OpenRouterConfig
    mas: MASConfig
    experiment: ExperimentRuntimeConfig
    self_evolved: SelfEvolvedConfig = field(default_factory=SelfEvolvedConfig)
    models: dict[str, str] = field(default_factory=dict)
    browsecomp: dict[str, Any] = field(default_factory=dict)
    stabletoolbench: dict[str, Any] = field(default_factory=dict)
    finance_agent: dict[str, Any] = field(default_factory=dict)
    plancraft: dict[str, Any] = field(default_factory=dict)
    math500: dict[str, Any] = field(default_factory=dict)
    workbench: dict[str, Any] = field(default_factory=dict)
    scicode: dict[str, Any] = field(default_factory=dict)
    agentbench: dict[str, Any] = field(default_factory=dict)
    webshop: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.openrouter.validate()
        self.mas.validate()
        self.experiment.validate()
        self.self_evolved.validate()

        if "default" not in self.models:
            raise ValueError("models.default is required in the config")


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    data = tomllib.loads(path.read_text(encoding="utf-8"))

    openrouter_raw = _as_dict(data.get("openrouter"), "[openrouter]")
    experiment_raw = _as_dict(data.get("experiment"), "[experiment]")
    mas_raw = _as_dict(data.get("mas"), "[mas]")
    models_raw = _as_dict(data.get("models"), "[models]")

    env_api_key = os.getenv("OPENROUTER_API_KEY")
    api_key = env_api_key if env_api_key else _opt_str(openrouter_raw.get("api_key"))

    # Per-LLM-call wall-clock timeout. Optional MAS_OPENROUTER_TIMEOUT_S env override
    # (opt-in; default behavior unchanged when unset) lets a run fail a slow provider
    # call fast so the retry loop re-routes to a faster provider, instead of blocking
    # up to the 600s ceiling on a transient slow upstream.
    env_timeout_s = os.getenv("MAS_OPENROUTER_TIMEOUT_S")
    openrouter = OpenRouterConfig(
        api_key=api_key,
        base_url=_opt_str(openrouter_raw.get("base_url")) or "https://openrouter.ai/api/v1",
        http_referer=_opt_str(openrouter_raw.get("http_referer")),
        x_title=_opt_str(openrouter_raw.get("x_title")),
        timeout_s=float(env_timeout_s)
        if env_timeout_s
        else float(openrouter_raw.get("timeout_s", 600.0)),
    )

    agents_per_level = mas_raw.get("agents_per_level")
    if agents_per_level is not None:
        if not isinstance(agents_per_level, list):
            raise ValueError("mas.agents_per_level must be a list of integers")
        parsed_agents_per_level = [int(value) for value in agents_per_level]
    else:
        parsed_agents_per_level = None

    agent_types = mas_raw.get("agent_types", ["general"])
    if isinstance(agent_types, str):
        agent_types = [agent_types]
    if not isinstance(agent_types, list):
        raise ValueError("mas.agent_types must be a list of strings")

    mas = MASConfig(
        levels=int(mas_raw.get("levels", 1)),
        intra_level_link_ratio=float(mas_raw.get("intra_level_link_ratio", 1.0)),
        full_linked=bool(mas_raw.get("full_linked", True)),
        topology=str(mas_raw.get("topology", "auto")),
        number_of_agents=(
            int(mas_raw["number_of_agents"]) if "number_of_agents" in mas_raw else None
        ),
        agents_per_level=parsed_agents_per_level,
        group_sizes=(
            [int(value) for value in mas_raw.get("group_sizes", [])]
            if "group_sizes" in mas_raw
            else None
        ),
        agent_types=[str(item) for item in agent_types],
        communication_count_internally=int(mas_raw.get("communication_count_internally", 1)),
        turn_mode=str(mas_raw.get("turn_mode", "single_turn")),
        max_turns=int(mas_raw.get("max_turns", 20)),
        discussion_rounds=int(mas_raw.get("discussion_rounds", 1)),
        minimum_discussion_rounds=int(mas_raw.get("minimum_discussion_rounds", 1)),
        termination_consensus_mode=str(mas_raw.get("termination_consensus_mode", "llm_judge")),
        final_vote_mode=str(mas_raw.get("final_vote_mode", "llm_judge")),
        peer_artifact_max_chars=int(mas_raw.get("peer_artifact_max_chars", 0)),
        enable_dynamic_roles=bool(mas_raw.get("enable_dynamic_roles", True)),
    )

    experiment = ExperimentRuntimeConfig(
        output_dir=str(experiment_raw.get("output_dir", "outputs")),
        runs_per_task=int(experiment_raw.get("runs_per_task", 3)),
        seed=int(experiment_raw.get("seed", 42)),
        task_limit=(int(experiment_raw["task_limit"]) if "task_limit" in experiment_raw else None),
    )

    self_evolved_raw = _as_dict(data.get("self_evolved"), "[self_evolved]")
    self_evolved = SelfEvolvedConfig(
        harness_backend=str(self_evolved_raw.get("harness_backend", "openrouter")),
        initial_planner_mode=str(self_evolved_raw.get("initial_planner_mode", "task_conditioned")),
        max_initial_agents=int(self_evolved_raw.get("max_initial_agents", 5)),
        max_total_agents=int(self_evolved_raw.get("max_total_agents", 10)),
        max_turns=int(self_evolved_raw.get("max_turns", 5)),
        repair_budget=int(self_evolved_raw.get("repair_budget", 4)),
        audit_mode=str(self_evolved_raw.get("audit_mode", "hybrid")),
        playbook_path=str(self_evolved_raw.get("playbook_path", "config/topology_playbook.json")),
        skill_path=str(self_evolved_raw.get("skill_path", "config/topology_skill.md")),
        playbook_read=bool(self_evolved_raw.get("playbook_read", True)),
        skill_update_batch_size=int(self_evolved_raw.get("skill_update_batch_size", 12)),
        default_packet_max_chars=int(self_evolved_raw.get("default_packet_max_chars", 0)),
        write_protocol=str(self_evolved_raw.get("write_protocol", "transactional_star")),
        reflection_mode=str(self_evolved_raw.get("reflection_mode", "batch")),
    )

    models = {str(key): str(value) for key, value in models_raw.items()}

    config = ExperimentConfig(
        openrouter=openrouter,
        mas=mas,
        experiment=experiment,
        self_evolved=self_evolved,
        models=models,
        browsecomp=_as_dict(data.get("browsecomp", {}), "[browsecomp]"),
        stabletoolbench=_as_dict(data.get("stabletoolbench", {}), "[stabletoolbench]"),
        finance_agent=_as_dict(data.get("finance_agent", {}), "[finance_agent]"),
        plancraft=_as_dict(data.get("plancraft", {}), "[plancraft]"),
        math500=_as_dict(data.get("math500", {}), "[math500]"),
        workbench=_as_dict(data.get("workbench", {}), "[workbench]"),
        scicode=_as_dict(data.get("scicode", {}), "[scicode]"),
        agentbench=_as_dict(data.get("agentbench", {}), "[agentbench]"),
        webshop=_as_dict(data.get("webshop", {}), "[webshop]"),
    )
    config.validate()
    return config


def _as_dict(value: Any, section_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{section_name} must be a table in TOML")
    return dict(value)


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    return value
