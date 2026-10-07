"""Self-evolved topology engine: meta-control loop around the TurnExecutor.

Meta-loop per run (the dynamic target agent system lives inside step 3):
1. PLAN      Topology Planner proposes the initial TopologySpec.
2. SPAWN     Orchestrator (deterministic code) instantiates agents, personas,
             and context policies.
3. EXECUTE   TurnExecutor runs one collaboration turn over the current spec.
4. AUDIT     Trace Auditor inspects the turn (phase 3).
5. CONTROL   Ordered code-level checks decide stop vs. a trace-backed repair
             (bounded by self_evolved.repair_budget and max_turns).
6. MUTATE    On the repair path, the Orchestrator applies the planner's
             topology mutation and re-runs one turn (phase 3).
7. FINALIZE  Deterministic/judge vote over one incumbent from every turn.

Agents never decide loop termination; the controller logic here does.
"""

from __future__ import annotations

import json
import math
import re
import time
from difflib import SequenceMatcher
from types import SimpleNamespace
from typing import Any

from answer_utils import has_substantive_answer

from ..artifacts import (
    ArtifactRecord,
    TerminationDecision,
    artifacts_by_id,
    build_artifact,
    latest_artifact_by_agent,
)
from ..config import SelfEvolvedConfig
from ..langgraph_engine import (
    UNSUPPORTED_FINAL_ANSWER,
    ExperimentSpec,
    LangGraphMASEngine,
    LangGraphRunResult,
    WorkflowDocumentation,
)
from ..monitor import DescriptorHook, NullDescriptor
from ..relay import TOPOLOGY_SELF_EVOLVED
from .auditor import TraceAuditorAgent
from .context import SharedContextController
from .executor import TurnExecutor, TurnResult, apply_updates, state_changing_tool_names
from .harness import AgentHarness
from .planner import TopologyPlannerAgent
from .playbook import PlaybookMaintainer, ShortTermTopologyPlaybook, TopologyPlaybook, task_key
from .retrieval import augment_with_constraint_search
from .spec import AgentNode, ContextPolicy, GroupSpec, TopologySpec
from .transaction import (
    augment_with_transaction_tools,
    calendar_scheduling_mode,
    commit_proposals,
    select_commit_proposals,
    successful_mutation_record,
    wrap_write_tools_as_proposals,
)


class SelfEvolvedEngine:
    """Drop-in engine with the same ``run`` signature as LangGraphMASEngine."""

    def __init__(self, harness: AgentHarness, se_config: SelfEvolvedConfig) -> None:
        self.llm_client = harness
        self.se_config = se_config
        self._stage = LangGraphMASEngine(harness)
        self.planner = TopologyPlannerAgent(harness, se_config)
        self.auditor = TraceAuditorAgent(harness, se_config)
        self._playbook: TopologyPlaybook | None = None
        self._playbook_loaded = False
        self._skill_cache: str | None = None
        self._skill_loaded = False

    def run(
        self,
        *,
        task: Any,
        run_index: int,
        seed: int,
        spec: ExperimentSpec,
        agent_types: list[str],
        tools: list[dict[str, Any]] | None = None,
        max_tool_iterations: int = 8,
        descriptor: DescriptorHook | None = None,
    ) -> LangGraphRunResult:
        spec = spec.normalized()
        if not agent_types:
            raise ValueError("agent_types must contain at least one entry")

        task_prompt = getattr(task, "prompt", "")
        calendar_mode = calendar_scheduling_mode(task_prompt)
        tools = augment_with_constraint_search(list(tools or []))
        tools = augment_with_transaction_tools(tools, task_prompt=task_prompt)

        num_agents = min(int(spec.num_agents), int(self.se_config.max_initial_agents))
        # Tool-bearing tasks run a multi-iteration tool loop per agent. Extra agents
        # mostly re-issue the same calls (same search corpus / same API) and converge on
        # the same evidence, so they multiply runtime/memory without adding quality —
        # depth (reading / tool-chaining), not agent breadth, is the lever. On a
        # memory-tight host the wide fan-out also gets the run OOM-killed (SIGTERM)
        # before it finalizes. Cap the initial breadth so the run completes; repair (a
        # second turn) stays enabled below for non-retrieval tool tasks to keep quality.
        is_retrieval = any(
            isinstance(tool, dict) and str(tool.get("name", "")) == "get_document"
            for tool in (tools or [])
        )
        if tools:
            if is_retrieval:
                # Retrieval runs a single turn (no repair doubling, see max_turns below),
                # so the OOM risk is bounded. Broad multi-hop search needs several
                # searchers covering different facets — static systems use the full agent
                # budget here and succeed where a 3-agent cap under-provisions search and
                # leaves a near-single searcher. Keep the configured initial breadth.
                num_agents = min(num_agents, int(self.se_config.max_initial_agents))
            else:
                # Non-retrieval tool tasks get a repair turn; cap breadth so the second
                # turn cannot OOM and to limit duplicate side-effecting tool calls.
                num_agents = min(num_agents, 3)

        # 1. PLAN — phase 2 replaces this with the LLM Topology Planner.
        plan_started = time.perf_counter()
        topo_spec, plan_payload = self._propose_initial_spec(
            task, num_agents, str(spec.benchmark_name or ""), bool(tools)
        )
        mutation_tool_names = state_changing_tool_names(list(tools or []))
        commit_once = (
            bool(mutation_tool_names) and self.se_config.write_protocol == "commit_once"
        )
        commit_handlers: dict[str, Any] = {}
        if commit_once:
            # The planned topology is kept; writes become proposals committed once.
            tools, commit_handlers = wrap_write_tools_as_proposals(
                list(tools or []), mutation_tool_names
            )
            plan_payload["transaction_protocol"] = {
                "mutation_tools": sorted(mutation_tool_names),
                "policy": "commit_once",
                "calendar_scheduling_mode": calendar_mode,
            }
        elif mutation_tool_names:
            topo_spec = self._transactional_star_spec(
                topo_spec, minimum_agents=min(3, max(2, num_agents))
            )
            plan_payload["transaction_protocol"] = {
                "committer_id": topo_spec.group(topo_spec.root_group_id).leader_id,
                "mutation_tools": sorted(mutation_tool_names),
                "policy": "readers_then_single_committer",
                "calendar_scheduling_mode": calendar_mode,
            }
        plan_latency_ms = max((time.perf_counter() - plan_started) * 1000.0, 1.0)
        topo_spec.validate(max_agents=int(self.se_config.max_total_agents))
        layout = topo_spec.to_layout()

        # 2. SPAWN — personas + state initialization.
        agent_type_by_agent = {
            agent_id: agent_types[idx % len(agent_types)]
            for idx, agent_id in enumerate(layout.agent_ids)
        }
        domain_personas, role_assignment_payload = self._assign_personas(
            task=task, spec=spec, layout=layout
        )

        workflow_definition = self._workflow_definition().to_payload()
        state = self._initial_state(
            task=task,
            run_index=run_index,
            seed=seed,
            spec=spec,
            layout=layout,
            agent_type_by_agent=agent_type_by_agent,
            tools=list(tools or []),
            max_tool_iterations=max_tool_iterations,
            descriptor=descriptor or NullDescriptor(),
            domain_personas=domain_personas,
            role_assignment_payload=role_assignment_payload,
            workflow_definition=workflow_definition,
        )
        if commit_once:
            state["self_evolved_write_protocol"] = "commit_once"
            state["self_evolved_calendar_scheduling_mode"] = calendar_mode
        elif mutation_tool_names:
            state["self_evolved_mutation_tool_names"] = sorted(mutation_tool_names)
            state["self_evolved_committer_id"] = str(
                topo_spec.group(topo_spec.root_group_id).leader_id or ""
            )
            state["self_evolved_calendar_scheduling_mode"] = calendar_mode

        spec_versions: list[dict[str, Any]] = [topo_spec.to_payload()]
        audit_reports: list[dict[str, Any]] = []
        mutation_payload: dict[str, Any] | None = None
        mutation_payloads: list[dict[str, Any]] = []
        short_term_playbook = ShortTermTopologyPlaybook()

        self._emit_meta_event(
            state,
            actor="topology_planner",
            event_type="plan",
            node_name="topology_plan",
            payload={
                "topology_spec": layout.to_payload(),
                "spec": topo_spec.to_payload(),
                "rationale": plan_payload.get("rationale", ""),
                "used_fallback": plan_payload.get("used_fallback", True),
                "fallback_reason": plan_payload.get("fallback_reason", ""),
                "playbook_keys": plan_payload.get("playbook_keys", []),
            },
            token_in=int(plan_payload.get("llm", {}).get("token_in", 0)),
            token_out=int(plan_payload.get("llm", {}).get("token_out", 0)),
            latency_ms=plan_latency_ms,
            cost_usd=float(plan_payload.get("llm", {}).get("cost_usd", 0.0)),
        )
        self._emit_meta_event(
            state,
            actor="orchestrator",
            event_type="plan",
            node_name="spawn_agents",
            payload={
                "agents": list(layout.agent_ids),
                "agent_type_by_agent": dict(agent_type_by_agent),
                "context_policies": {
                    node.agent_id: {
                        "visible_kinds": list(node.context.visible_kinds),
                        "visible_from": list(node.context.visible_from),
                        "share_scope": node.context.share_scope,
                        "evidence_access": node.context.evidence_access,
                    }
                    for node in topo_spec.agents
                },
            },
        )
        self._record_context_state(
            state,
            topo_spec,
            layout,
            reason="spawn",
            turn_index=None,
        )

        # 3-6. EXECUTE / AUDIT / CONTROL / MUTATE loop.
        context = SharedContextController(topo_spec)
        executor = TurnExecutor(self._stage, context)
        max_turns = int(self.se_config.max_turns)
        repair_budget = max(0, int(self.se_config.repair_budget))
        # A full retrieval repair re-runs every searcher over large contexts and may
        # spawn more agents, which previously caused OOM termination. Permit one repair
        # opportunity, but contract that repair to one evidence-recovery specialist
        # below. This spends depth only after the auditor observes a concrete failure
        # while keeping peak fan-out bounded.
        if is_retrieval:
            max_turns = min(max_turns, 2)
        turn_results: list[TurnResult] = []
        decision: TerminationDecision = {}

        for turn_index in range(max_turns):
            result = executor.run_turn(state, topo_spec, turn_index=turn_index)
            turn_results.append(result)
            apply_updates(state, self._stage._descriptor_monitor_node(state))

            audit_report = self._audit_turn(state, topo_spec, turn_index=turn_index)
            if audit_report is not None:
                audit_reports.append(audit_report)

            mutations_used = len(spec_versions) - 1
            repeated_decision = self._repair_repeated_decision(turn_results)
            # Under commit_once nothing is committed before finalize, so repair stays open.
            transaction_committed = bool(mutation_tool_names) and not commit_once and any(
                successful_mutation_record(record, mutation_tool_names)
                for record in state.get("tool_records_log", [])
                if isinstance(record, dict)
            )
            repair_available = (
                mutations_used < repair_budget
                and turn_index + 1 < max_turns
                and self._repair_recommended(audit_report)
                and not repeated_decision
                and not transaction_committed
            )
            decision = self._meta_termination(
                state,
                turn_index=turn_index,
                result=result,
                previous_result=turn_results[-2] if len(turn_results) > 1 else None,
                repair_available=repair_available,
                audit_report=audit_report,
            )
            if transaction_committed:
                decision = {
                    **decision,
                    "should_stop": True,
                    "next_step": "finalize",
                    "reason": "transaction_committed",
                    "reason_detail": (
                        "The designated committer completed a state-changing tool call; "
                        "further turns are suppressed to preserve exactly-once effects."
                    ),
                }
            if repeated_decision and self._repair_recommended(audit_report):
                decision = {
                    **decision,
                    "should_stop": True,
                    "next_step": "finalize",
                    "reason": "no_meaningful_change",
                    "reason_detail": (
                        "The semantic decision signature did not change after the prior "
                        "topology mutation; suppressing further agent growth and retaining "
                        "the best temporal incumbent."
                    ),
                    "repair_suppressed": "repeated_decision_signature",
                }
            state["termination_decision"] = decision
            apply_updates(
                state,
                {
                    "termination_history": [dict(decision)],
                    "trace_payloads": [
                        self._stage._termination_event(state, "meta_termination", decision)
                    ],
                },
            )
            self._record_short_term_playbook_turn(
                state=state,
                short_term_playbook=short_term_playbook,
                topo_spec=topo_spec,
                task=task,
                audit_report=audit_report,
                termination_decision=decision,
                turn_index=turn_index,
            )
            if bool(decision.get("should_stop", True)):
                break

            # Trace-backed repair (one mutation per turn, capped by repair_budget).
            if is_retrieval:
                mutated = self._apply_retrieval_rescue_mutation(
                    state,
                    topo_spec,
                    audit_report,
                )
            else:
                mutated = self._apply_mutation(
                    state,
                    topo_spec,
                    audit_report,
                    task=task,
                    agent_types=agent_types,
                    short_term_playbook=short_term_playbook,
                )
            if mutated is None:
                # The continue decision is void without a usable mutation;
                # record the corrected stop so the trace stays coherent.
                decision = {
                    **decision,
                    "should_stop": True,
                    "next_step": "finalize",
                    "reason": "max_rounds_reached",
                    "reason_detail": (
                        "Repair was recommended but the planner produced no valid "
                        "mutation; finalizing on the current topology."
                    ),
                }
                state["termination_decision"] = decision
                apply_updates(
                    state,
                    {
                        "termination_history": [dict(decision)],
                        "trace_payloads": [
                            self._stage._termination_event(state, "meta_termination", decision)
                        ],
                    },
                )
                break
            topo_spec, mutation_payload = mutated
            if mutation_tool_names and not commit_once:
                topo_spec = self._transactional_star_spec(
                    topo_spec, minimum_agents=min(3, max(2, num_agents))
                )
                mutation_payload = {
                    **mutation_payload,
                    "transaction_protocol_reapplied": True,
                    "committer_id": topo_spec.group(topo_spec.root_group_id).leader_id,
                }
            mutation_payloads.append(mutation_payload)
            state["self_evolved_repair_directive"] = self._repair_directive(audit_report)
            layout = topo_spec.to_layout()
            state["layout"] = layout
            context.set_spec(topo_spec)
            spec_versions.append(topo_spec.to_payload())
            self._record_context_state(
                state,
                topo_spec,
                layout,
                reason="mutation",
                turn_index=turn_index,
            )

        # 7. FINALIZE.
        final_result = turn_results[-1] if turn_results else TurnResult(output_artifact=None)
        final_answer = self._finalize(
            state,
            final_result,
            decision,
            turn_results=turn_results,
        )
        if commit_once:
            source_artifact_id, proposals = select_commit_proposals(
                list(state.get("artifacts", [])),
                str(state.get("selected_artifact_id", "")),
                mutation_tool_names,
            )
            commit_payload = {
                "policy": "commit_once",
                "source_artifact_id": source_artifact_id,
                "proposals_in_run": sum(
                    1
                    for record in state.get("tool_records_log", [])
                    if isinstance(record, dict)
                    and successful_mutation_record(record, mutation_tool_names)
                ),
                "committed": commit_proposals(proposals, commit_handlers),
            }
            state["self_evolved_commit"] = commit_payload
            self._emit_meta_event(
                state,
                actor="transaction_committer",
                event_type="act",
                node_name="self_evolved_commit",
                payload=commit_payload,
            )

        # 8. RECORD — long-term playbook candidate; runs never write the
        # persistent playbook file (see scripts/update_topology_playbook.py).
        playbook_key = task_key(str(spec.benchmark_name or ""), task, tools_available=bool(tools))
        playbook_candidate = PlaybookMaintainer.build_update_candidate(
            key=playbook_key,
            benchmark_name=str(spec.benchmark_name or ""),
            spec_versions=spec_versions,
            audit_reports=audit_reports,
            mutation_payload=mutation_payload,
            mutation_payloads=mutation_payloads,
            termination_decision=dict(decision or {}),
            state_changing=bool(mutation_tool_names),
        )
        self._emit_meta_event(
            state,
            actor="playbook_maintainer",
            event_type="act",
            node_name="playbook_candidate",
            payload={"scope": "long_term", "update_candidate": playbook_candidate},
        )

        trace_events = self._stage._materialize_trace_events(state.get("trace_payloads", []))
        run_metadata = self._build_run_metadata(
            state=state,
            spec=spec,
            layout=layout,
            run_index=run_index,
            seed=seed,
            workflow_definition=workflow_definition,
            role_assignment_payload=role_assignment_payload,
            spec_versions=spec_versions,
            audit_reports=audit_reports,
            mutation_payload=mutation_payload,
            mutation_payloads=mutation_payloads,
            plan_payload=plan_payload,
            playbook_candidate=playbook_candidate,
        )
        return LangGraphRunResult(
            final_answer=final_answer,
            trace_events=trace_events,
            run_metadata=run_metadata,
        )

    # -- meta-agent hooks (phases 2-4 replace the stubs) ---------------------

    @staticmethod
    def _transactional_star_spec(spec: TopologySpec, *, minimum_agents: int = 2) -> TopologySpec:
        """Contract stateful work into readers followed by one commit stage."""

        source_nodes = {node.agent_id: node for node in spec.agents}
        agent_ids = list(source_nodes)
        next_index = 0
        while len(agent_ids) < max(2, int(minimum_agents)):
            candidate = f"agent_{next_index}"
            next_index += 1
            if candidate not in source_nodes and candidate not in agent_ids:
                agent_ids.append(candidate)
        leader_id = "agent_0" if "agent_0" in agent_ids else agent_ids[0]
        root_group_id = "g_transaction"
        template = spec.agents[0]
        agents = tuple(
            AgentNode(
                agent_id=agent_id,
                group_id=root_group_id,
                structural_role="coordinator" if agent_id == leader_id else "worker",
                stage_role=(
                    "worker"
                    if agent_id == leader_id
                    else source_nodes.get(agent_id, template).stage_role
                ),
                persona_hint=source_nodes.get(agent_id, template).persona_hint,
                allowed_tools=source_nodes.get(agent_id, template).allowed_tools,
                context=source_nodes.get(agent_id, template).context,
            )
            for agent_id in agent_ids
        )
        return TopologySpec(
            version=spec.version,
            agents=agents,
            groups=(
                GroupSpec(
                    group_id=root_group_id,
                    pattern="star",
                    member_ids=tuple(agent_ids),
                    leader_id=leader_id,
                ),
            ),
            root_group_id=root_group_id,
            rationale=(
                f"{spec.rationale} Capability-aware transaction contraction: independent "
                "readers report to one deterministic committer."
            ).strip(),
        )

    def _propose_initial_spec(
        self, task: Any, num_agents: int, benchmark_name: str, tools_available: bool
    ) -> tuple[TopologySpec, dict[str, Any]]:
        if self.se_config.initial_planner_mode == "fixed":
            fixed = self.planner.fallback_initial_spec(num_agents)
            return fixed, {
                "rationale": fixed.rationale,
                "used_fallback": True,
                "fallback_reason": "fixed_initial_topology_ablation",
                "prompt_messages": [],
                "response": "",
                "llm": {},
                "playbook_keys": [],
            }
        proposal = self.planner.propose_initial(
            task=task,
            benchmark_name=benchmark_name,
            num_agents=num_agents,
            playbook_entries=self._playbook_entries(task, benchmark_name, tools_available),
            principles=self._playbook_principles(),
            skill_text=self._skill_text(),
        )
        return proposal.spec, proposal.to_payload()

    def _skill_text(self) -> str:
        """The agent-maintained markdown skill (primary long-term memory), if present."""

        if not self.se_config.playbook_read:
            return ""
        if not self._skill_loaded:
            self._skill_loaded = True
            from .skill import TopologySkill

            self._skill_cache = TopologySkill.load(self.se_config.skill_path).prompt_section()
        return self._skill_cache or ""

    def reload_skill(self) -> None:
        """Drop the cached skill so the next plan re-reads it from disk.

        Called after the online skill updater rewrites the skill file mid-experiment
        so subsequent runs plan against the revised skill (the engine instance is
        reused across all runs in a `run` command)."""

        self._skill_loaded = False
        self._skill_cache = None

    def _playbook_principles(self) -> list[str]:
        """Benchmark-agnostic priors injected into every planner prompt."""

        if not self.se_config.playbook_read:
            return []
        playbook = self._load_playbook()
        if playbook is None:
            return []
        return playbook.global_principles()

    def _playbook_entries(
        self, task: Any, benchmark_name: str, tools_available: bool
    ) -> list[dict[str, Any]] | None:
        """Long-term playbook priors for the planner; runs never write the file."""

        if not self.se_config.playbook_read:
            return None
        playbook = self._load_playbook()
        if playbook is None or not playbook.entries:
            return None
        key = task_key(benchmark_name, task, tools_available=tools_available)
        entries = playbook.lookup(benchmark_name, key)
        views = [TopologyPlaybook.planner_view(entry) for entry in entries]
        return views or None

    def _load_playbook(self) -> TopologyPlaybook | None:
        if not self._playbook_loaded:
            self._playbook_loaded = True
            self._playbook = TopologyPlaybook.load(self.se_config.playbook_path)
        return self._playbook

    def _audit_turn(
        self, state: dict[str, Any], topo_spec: TopologySpec, *, turn_index: int
    ) -> dict[str, Any] | None:
        audit_started = time.perf_counter()
        report = self.auditor.audit(state, topo_spec, turn_index=turn_index)
        audit_latency_ms = max((time.perf_counter() - audit_started) * 1000.0, 1.0)
        llm_payload = report.get("llm", {}) if isinstance(report.get("llm"), dict) else {}
        self._emit_meta_event(
            state,
            actor="trace_auditor",
            event_type="verify",
            node_name=f"audit_turn_{turn_index}",
            payload={"audit": report},
            token_in=int(llm_payload.get("token_in", 0)),
            token_out=int(llm_payload.get("token_out", 0)),
            latency_ms=audit_latency_ms,
            cost_usd=float(llm_payload.get("cost_usd", 0.0)),
        )
        return report

    @staticmethod
    def _repair_recommended(audit_report: dict[str, Any] | None) -> bool:
        if not audit_report:
            return False
        return bool(audit_report.get("repair_recommended", False))

    @staticmethod
    def _repair_repeated_decision(turn_results: list[TurnResult]) -> bool:
        """Stop spending repair budget when one mutation leaves the decision unchanged."""

        if len(turn_results) < 2:
            return False
        signatures = []
        for result in turn_results[-2:]:
            artifact = result.output_artifact
            if artifact is None:
                return False
            signature = TurnExecutor._decision_answer_signature(str(artifact.get("answer", "")))
            if not signature:
                return False
            signatures.append(signature)
        return signatures[0] == signatures[1]

    @staticmethod
    def _repair_directive(audit_report: dict[str, Any] | None) -> str:
        if not audit_report:
            return ""
        findings = []
        for mode in audit_report.get("detected_modes", [])[:5]:
            findings.append(
                f"{mode.get('mode', 'unknown')}: {str(mode.get('detail', '')).strip()[:400]}"
            )
        recommendation = str(audit_report.get("recommendation", "")).strip()[:600]
        return " | ".join([*findings, f"recommended check: {recommendation}"])[-1800:]

    def _apply_mutation(
        self,
        state: dict[str, Any],
        topo_spec: TopologySpec,
        audit_report: dict[str, Any] | None,
        *,
        task: Any,
        agent_types: list[str],
        short_term_playbook: ShortTermTopologyPlaybook,
    ) -> tuple[TopologySpec, dict[str, Any]] | None:
        playbook_entries = (
            self._playbook_entries(
                task,
                str(state.get("benchmark_name", "")),
                bool(state.get("tools")),
            )
            or []
        )
        playbook_entries = [*short_term_playbook.planner_entries(), *playbook_entries]
        proposal_started = time.perf_counter()
        proposal = self.planner.propose_mutation(
            task=task,
            spec=topo_spec,
            audit_report=audit_report or {},
            playbook_entries=playbook_entries,
            principles=self._playbook_principles(),
            skill_text=self._skill_text(),
        )
        proposal_latency_ms = max((time.perf_counter() - proposal_started) * 1000.0, 1.0)
        proposal_payload = proposal.to_payload()
        state.setdefault("self_evolved_mutation_proposals", []).append(proposal_payload)
        llm_payload = proposal_payload.get("llm", {})
        self._emit_meta_event(
            state,
            actor="topology_planner",
            event_type="revise",
            node_name="mutation_proposal",
            payload={"proposal": proposal_payload},
            token_in=int(llm_payload.get("token_in", 0)),
            token_out=int(llm_payload.get("token_out", 0)),
            latency_ms=proposal_latency_ms,
            cost_usd=float(llm_payload.get("cost_usd", 0.0)),
        )
        mutation = proposal.mutation
        if mutation is None:
            self._emit_meta_event(
                state,
                actor="orchestrator",
                event_type="revise",
                node_name="mutation_skipped",
                payload={
                    "reason": proposal.fallback_reason or "no_mutation_proposed",
                    "used_fallback": bool(proposal.used_fallback),
                },
            )
            return None

        mutated_spec = mutation.apply(topo_spec, max_agents=int(self.se_config.max_total_agents))
        mutated_layout = mutated_spec.to_layout()
        new_agents = [
            agent_id
            for agent_id in mutated_layout.agent_ids
            if agent_id not in state.get("agent_type_by_agent", {})
        ]
        self._register_new_agents(state, new_agents, agent_types, mutated_layout)

        mutation_payload = {
            **proposal.to_payload(),
            "added_agents": list(new_agents),
            "spec_version": int(mutated_spec.version),
        }
        self._emit_meta_event(
            state,
            actor="orchestrator",
            event_type="revise",
            node_name="apply_mutation",
            payload={
                "mutation": mutation.to_payload(),
                "added_agents": list(new_agents),
                "topology_spec": mutated_layout.to_payload(),
            },
        )
        return mutated_spec, mutation_payload

    def _apply_retrieval_rescue_mutation(
        self,
        state: dict[str, Any],
        topo_spec: TopologySpec,
        audit_report: dict[str, Any] | None,
    ) -> tuple[TopologySpec, dict[str, Any]]:
        """Contract a failed retrieval turn to one bounded evidence specialist.

        Retrieval failures need a different query/read path, not another wide copy of
        the original team.  Reusing one existing agent avoids new model state and keeps
        the OOM guard structural rather than disabling repair altogether.
        """

        search_tools = {"search", "constraint_search", "get_document"}
        call_counts: dict[str, int] = {}
        for record in state.get("tool_records_log", []):
            if str(record.get("tool_name", "")) not in search_tools:
                continue
            agent_id = str(record.get("agent_id", ""))
            if agent_id:
                call_counts[agent_id] = call_counts.get(agent_id, 0) + 1

        ordered_ids = topo_spec.ordered_agent_ids()
        eligible = [
            node.agent_id for node in topo_spec.agents if node.structural_role != "coordinator"
        ] or ordered_ids
        chosen = max(
            eligible,
            key=lambda agent_id: (call_counts.get(agent_id, 0), -ordered_ids.index(agent_id)),
        )
        previous = topo_spec.agent(chosen)
        rescue_group = "g_retrieval_rescue"
        rescue_spec = TopologySpec(
            version=int(topo_spec.version) + 1,
            agents=(
                AgentNode(
                    agent_id=chosen,
                    group_id=rescue_group,
                    structural_role="worker",
                    stage_role="worker",
                    persona_hint=(
                        f"{previous.persona_hint} "
                        "Act as the sole evidence-recovery specialist: use constraint_search, "
                        "open decisive documents, test the prior answer against every clue, and "
                        "return the exact requested answer granularity."
                    ).strip(),
                    allowed_tools=previous.allowed_tools,
                    context=ContextPolicy(
                        share_scope="global",
                        evidence_access="global",
                        summary_only=True,
                    ),
                ),
            ),
            groups=(
                GroupSpec(
                    group_id=rescue_group,
                    pattern="singleton",
                    member_ids=(chosen,),
                ),
            ),
            root_group_id=rescue_group,
            rationale=(
                "Audit-triggered resource-adaptive contraction: one specialist explores a "
                "different retrieval/read path without repeating the wide fan-out."
            ),
        )
        rescue_spec.validate(max_agents=int(self.se_config.max_total_agents))
        payload = {
            "source": "resource_adaptive_retrieval_rescue",
            "used_fallback": False,
            "fallback_reason": "",
            "mutation": {
                "type": "contract_to_retrieval_rescue",
                "agent_id": chosen,
                "audit_modes": [
                    str(mode.get("mode", ""))
                    for mode in (audit_report or {}).get("detected_modes", [])
                ],
            },
            "added_agents": [],
            "spec_version": int(rescue_spec.version),
        }
        state.setdefault("self_evolved_mutation_proposals", []).append(payload)
        self._emit_meta_event(
            state,
            actor="resource_guard",
            event_type="revise",
            node_name="mutation_proposal",
            payload={"proposal": payload},
        )
        self._emit_meta_event(
            state,
            actor="orchestrator",
            event_type="revise",
            node_name="apply_mutation",
            payload={
                "mutation": payload["mutation"],
                "added_agents": [],
                "topology_spec": rescue_spec.to_layout().to_payload(),
            },
        )
        return rescue_spec, payload

    def _register_new_agents(
        self,
        state: dict[str, Any],
        new_agents: list[str],
        agent_types: list[str],
        layout: Any,
    ) -> None:
        """Spawn bookkeeping for agents created by a mutation: model types,
        message budgets, and deterministic personas (no extra LLM round-trip)."""

        if not new_agents:
            return
        agent_type_by_agent = dict(state.get("agent_type_by_agent", {}))
        message_budget = dict(state.get("message_budget", {}))
        sent_counts = dict(state.get("sent_counts", {}))
        budget_sent_counts = dict(state.get("budget_sent_counts", {}))
        budget = int(state.get("communication_budget_per_agent", 0))
        for idx, agent_id in enumerate(new_agents):
            offset = len(agent_type_by_agent) + idx
            agent_type_by_agent[agent_id] = agent_types[offset % len(agent_types)]
            message_budget.setdefault(agent_id, budget)
            sent_counts.setdefault(agent_id, 0)
            budget_sent_counts.setdefault(agent_id, 0)
        state["agent_type_by_agent"] = agent_type_by_agent
        state["message_budget"] = message_budget
        state["sent_counts"] = sent_counts
        state["budget_sent_counts"] = budget_sent_counts

        benchmark_name = str(state.get("benchmark_name", ""))
        if benchmark_name:
            from ..role_assigner import assign_domain_roles_deterministic

            assignments = assign_domain_roles_deterministic(
                benchmark_name=benchmark_name, layout=layout
            )
            personas = dict(state.get("domain_personas", {}))
            for agent_id in new_agents:
                info = assignments.get(agent_id)
                if info is not None and agent_id not in personas:
                    personas[agent_id] = {
                        "role_name": info.role_name,
                        "persona": info.persona,
                    }
            state["domain_personas"] = personas

    def _record_short_term_playbook_turn(
        self,
        *,
        state: dict[str, Any],
        short_term_playbook: ShortTermTopologyPlaybook,
        topo_spec: TopologySpec,
        task: Any,
        audit_report: dict[str, Any] | None,
        termination_decision: dict[str, Any],
        turn_index: int,
    ) -> None:
        key = task_key(
            str(state.get("benchmark_name", "")),
            task,
            tools_available=bool(state.get("tools")),
        )
        entry = short_term_playbook.record_turn(
            key=key,
            benchmark_name=str(state.get("benchmark_name", "")),
            turn_index=turn_index,
            spec_payload=topo_spec.to_payload(),
            audit_report=audit_report,
            termination_decision=termination_decision,
        )
        state.setdefault("short_term_playbook_entries", []).append(entry)
        self._emit_meta_event(
            state,
            actor="playbook_maintainer",
            event_type="revise",
            node_name=f"short_term_playbook_turn_{turn_index}",
            payload={"scope": "short_term", "entry": entry},
        )

    def _record_context_state(
        self,
        state: dict[str, Any],
        topo_spec: TopologySpec,
        layout: Any,
        *,
        reason: str,
        turn_index: int | None,
    ) -> None:
        snapshot = self._context_state_snapshot(
            topo_spec,
            layout,
            reason=reason,
            turn_index=turn_index,
        )
        state.setdefault("context_state_versions", []).append(snapshot)
        self._emit_meta_event(
            state,
            actor="orchestrator",
            event_type="revise",
            node_name=f"context_state_{reason}",
            payload={"context_state": snapshot},
        )

    @staticmethod
    def _context_state_snapshot(
        topo_spec: TopologySpec,
        layout: Any,
        *,
        reason: str,
        turn_index: int | None,
    ) -> dict[str, Any]:
        return {
            "reason": reason,
            "turn_index": turn_index,
            "spec_version": int(topo_spec.version),
            "agents": list(layout.agent_ids),
            "adjacency": {
                agent_id: list(peers) for agent_id, peers in dict(layout.adjacency).items()
            },
            "groups": [
                {
                    "group_id": group.group_id,
                    "pattern": group.pattern,
                    "member_ids": list(group.member_ids),
                    "parent_agent_id": group.parent_agent_id,
                    "leader_id": group.leader_id,
                }
                for group in topo_spec.groups
            ],
            "context_policies": {
                node.agent_id: {
                    "visible_kinds": list(node.context.visible_kinds),
                    "visible_from": list(node.context.visible_from),
                    "share_scope": node.context.share_scope,
                    "evidence_access": node.context.evidence_access,
                    "summary_only": bool(node.context.summary_only),
                    "max_packet_chars": int(node.context.max_packet_chars),
                }
                for node in topo_spec.agents
            },
        }

    # -- control -------------------------------------------------------------

    def _meta_termination(
        self,
        state: dict[str, Any],
        *,
        turn_index: int,
        result: TurnResult,
        previous_result: TurnResult | None,
        repair_available: bool,
        audit_report: dict[str, Any] | None = None,
    ) -> TerminationDecision:
        candidate = [result.output_artifact] if result.output_artifact else []
        previous: list[ArtifactRecord] = []
        if previous_result is not None and previous_result.output_artifact:
            previous = [previous_result.output_artifact]
        consensus = result.member_artifacts or candidate

        decision = self._stage._termination_decision(
            state,
            stage_name="meta_termination",
            round_index=turn_index,
            discussion_index=int(state.get("discussion_index", 0)),
            minimum_round_index=turn_index,
            minimum_required_rounds=0,
            minimum_round_label="turn",
            candidate_artifacts=candidate or consensus,
            previous_candidate_artifacts=previous,
            consensus_artifacts=consensus,
            expected_count=max(1, int(result.expected_member_count)),
            max_reached=not repair_available,
            continue_next_step="apply_mutation",
            stop_next_step="finalize",
        )
        # Decision-grade agreement normally stops the loop. A grounded, high-risk
        # open-set audit may challenge it once a concrete repair is available; this
        # catches correlated or incomplete consensus that confidence alone cannot.
        # The controller still owns the decision and the normal turn/repair ceilings
        # apply, so an auditor can never create an unbounded loop.
        if (
            bool(decision.get("should_stop", True))
            and str(decision.get("reason", "")) == "consensus_reached"
            and repair_available
            and bool((audit_report or {}).get("challenge_consensus", False))
        ):
            challenged_modes = [
                str(mode.get("mode", ""))
                for mode in (audit_report or {}).get("detected_modes", [])
                if mode.get("mode")
            ]
            decision = {
                **decision,
                "should_stop": False,
                "next_step": "apply_mutation",
                "reason": "audit_challenge",
                "reason_detail": (
                    "A grounded high-risk trace finding challenged otherwise "
                    "decision-grade consensus: " + ", ".join(challenged_modes[:5])
                ),
            }
        if (
            bool(decision.get("should_stop", True))
            and str(decision.get("reason")) == "max_rounds_reached"
            and not repair_available
        ):
            decision = {
                **decision,
                "reason_detail": (
                    "No trace-backed repair was available "
                    "(audit found no repair or the repair budget was spent)."
                ),
            }
        return decision

    # -- finalize -------------------------------------------------------------

    def _finalize(
        self,
        state: dict[str, Any],
        result: TurnResult,
        decision: TerminationDecision,
        *,
        turn_results: list[TurnResult] | None = None,
    ) -> str:
        candidates: list[ArtifactRecord] = []
        history = list(turn_results or [result])
        temporal_outputs = [
            turn.output_artifact
            for turn in history
            if turn.output_artifact is not None
            and has_substantive_answer(str(turn.output_artifact.get("answer", "")))
        ]
        # Preserve one aggregate/incumbent per turn. Repairs are experiments, not
        # destructive replacements: the final judge may keep an earlier answer when a
        # later topology regresses. Artifact ids make the temporal ensemble replayable.
        seen_ids: set[str] = set()
        for artifact in temporal_outputs:
            artifact_id = str(artifact.get("artifact_id", ""))
            if artifact_id and artifact_id in seen_ids:
                continue
            candidates.append(artifact)
            if artifact_id:
                seen_ids.add(artifact_id)

        output = result.output_artifact
        if not candidates:
            if result.member_artifacts:
                candidates = list(result.member_artifacts)
            elif output is not None:
                candidates = [output]
            else:
                candidates = list(state.get("artifacts", []))

        state["self_evolved_temporal_candidates"] = [
            {
                "artifact_id": str(artifact.get("artifact_id", "")),
                "agent_id": str(artifact.get("agent_id", "")),
                "round_index": int(artifact.get("round_index", 0)),
                "confidence": float(artifact.get("confidence", 0.5)),
            }
            for artifact in candidates
        ]

        vote_result = self._stage._select_final_answer(
            state=state, stage_name="finalize", artifacts=candidates
        )
        final_answer = self._stage._safe_vote_answer_or_fallback(state, candidates, vote_result)
        synthesis_artifact = self._maybe_synthesize_final_answer(
            state=state,
            final_answer=final_answer,
            selected_artifact_id=str(vote_result.get("selected_artifact_id", "")),
            vote_result=vote_result,
        )
        if synthesis_artifact is not None:
            candidates = [synthesis_artifact]
            final_answer = str(synthesis_artifact.get("answer", "") or final_answer)
            vote_result = {
                "selected_artifact_id": synthesis_artifact.get("artifact_id", ""),
                "selected_agent_id": synthesis_artifact.get("agent_id", ""),
                "selected_source_artifact_ids": synthesis_artifact.get("source_artifact_ids", []),
                "source": "self_evolved_final_synthesis",
                "tally": {final_answer.lower(): 1} if final_answer else {},
            }

        contract_artifact = self._maybe_repair_output_contract(
            state=state,
            final_answer=final_answer,
            selected_artifact_id=str(vote_result.get("selected_artifact_id", "")),
        )
        if contract_artifact is not None:
            candidates = [contract_artifact]
            final_answer = str(contract_artifact.get("answer", "") or final_answer)
            vote_result = {
                "selected_artifact_id": contract_artifact.get("artifact_id", ""),
                "selected_agent_id": contract_artifact.get("agent_id", ""),
                "selected_source_artifact_ids": contract_artifact.get("source_artifact_ids", []),
                "source": "self_evolved_output_contract",
                "tally": {final_answer.lower(): 1} if final_answer else {},
            }

        canonical_answer = self._canonicalize_explicit_math_answer(final_answer)
        canonical_answer = self._canonicalize_structured_identity_answer(
            canonical_answer, task_prompt=state.get("task_prompt", "")
        )
        canonical_answer = self._append_verified_entity_aliases(canonical_answer, state=state)
        if canonical_answer != final_answer:
            self._emit_meta_event(
                state,
                actor="output_canonicalizer",
                event_type="revise",
                node_name="self_evolved_output_canonicalization",
                payload={
                    "source_artifact_id": str(vote_result.get("selected_artifact_id", "")),
                    "before": final_answer[:1000],
                    "after": canonical_answer[:1000],
                },
            )
            final_answer = canonical_answer
            vote_result["tally"] = {final_answer.lower(): 1}

        state["final_answer"] = final_answer
        state["final_reason"] = (
            f"{TOPOLOGY_SELF_EVOLVED}:{decision.get('reason', 'finalize')}"
            if decision
            else f"{TOPOLOGY_SELF_EVOLVED}:finalize"
        )
        state["vote_tally"] = dict(vote_result.get("tally", {}))
        state["final_vote_source"] = str(vote_result.get("source", ""))
        state["selected_artifact_id"] = str(vote_result.get("selected_artifact_id", ""))
        state["selected_agent_id"] = str(vote_result.get("selected_agent_id", ""))
        state["selected_source_artifact_ids"] = list(
            vote_result.get("selected_source_artifact_ids", [])
        )
        apply_updates(state, self._stage._finalize_node(state))
        return str(state.get("final_answer", final_answer))

    @staticmethod
    def _canonicalize_explicit_math_answer(answer: str) -> str:
        if not re.search(r"\\(?:boxed|fbox)\s*\{", answer):
            return answer
        return re.sub(
            r"(?<!\\frac\{)(\\(?:pi|theta|alpha|beta|gamma)|-?\d+)\s*/\s*(\d+)",
            r"\\frac{\1}{\2}",
            answer,
        )

    @staticmethod
    def _canonicalize_structured_identity_answer(answer: str, *, task_prompt: Any) -> str:
        """Flatten an unsolicited name object without changing requested schemas.

        Models sometimes return ``{"first_name": ..., "last_name": ...}`` to an
        ordinary identification question.  That object and the plain full name carry
        the same claim, but exact-answer tasks reasonably reject the wrapper.  Keep it
        intact when the task explicitly asks for JSON, an object, or named fields.
        """

        text = str(answer or "").strip()
        if not text.startswith("{") or not text.endswith("}"):
            return answer
        prompt = str(task_prompt or "").lower()
        if any(
            marker in prompt
            for marker in (
                "json",
                "structured object",
                "first_name",
                "middle_name",
                "last_name",
                "given_name",
                "family_name",
            )
        ):
            return answer
        try:
            payload = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            return answer
        if not isinstance(payload, dict):
            return answer

        allowed = {
            "title",
            "prefix",
            "first_name",
            "given_name",
            "middle_name",
            "last_name",
            "family_name",
            "surname",
            "suffix",
        }
        lowered = {str(key).strip().lower(): value for key, value in payload.items()}
        if not lowered or not set(lowered) <= allowed:
            return answer

        def component(*keys: str) -> str:
            for key in keys:
                value = lowered.get(key)
                if isinstance(value, str) and value.strip():
                    return re.sub(r"\s+", " ", value).strip()
            return ""

        first = component("first_name", "given_name")
        last = component("last_name", "family_name", "surname")
        if not first or not last:
            return answer
        parts = [
            component("title", "prefix"),
            first,
            component("middle_name"),
            last,
            component("suffix"),
        ]
        return " ".join(part for part in parts if part)

    @classmethod
    def _append_verified_entity_aliases(cls, answer: str, *, state: dict[str, Any]) -> str:
        """Retain close spelling/transliteration variants surfaced by tools.

        Named entities often have multiple legitimate Latin-script renderings.  A
        finalizer can silently choose one even when another retrieved source uses a
        one-character variant.  For a short entity answer, append only near-identical
        title/name variants copied from successful tool output.  This never invents an
        alias and does not rewrite structured or sentence-shaped answers.
        """

        text = re.sub(r"\s+", " ", str(answer or "")).strip()
        if (
            not text
            or len(text) > 160
            or len(text.split()) > 12
            or text[0] in "{["
            or "\n" in str(answer)
            or re.search(r"[.!?](?:\s|$)", text)
        ):
            return answer

        def comparable(value: str) -> str:
            base = re.sub(r"\s*\([^)]{1,80}\)\s*$", "", value).strip()
            return re.sub(r"[^a-z0-9]+", "", base.casefold())

        answer_key = comparable(text)
        if len(answer_key) < 5:
            return answer
        answer_words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'’.-]*", text)
        task_prompt = str(state.get("task_prompt", "")).casefold()
        formal_entity_requested = any(
            marker in task_prompt
            for marker in (
                "institute",
                "institution",
                "college",
                "university",
                "organization",
                "organisation",
                "company",
                "agency",
                "school",
                "hospital",
                "mosque",
            )
        )

        surfaced: list[str] = []
        formal_extensions: list[str] = []

        def collect(value: Any, *, field: str = "") -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    collect(item, field=str(key).strip().lower())
                return
            if isinstance(value, list):
                for item in value:
                    collect(item, field=field)
                return
            if not isinstance(value, str):
                return
            if field in {"title", "name", "official_name", "full_name"}:
                candidate = re.sub(r"\s+", " ", value).strip(" -\t")
                if candidate:
                    surfaced.append(candidate)
            if field in {"snippet", "text", "output_preview"}:
                if formal_entity_requested:
                    extension_pattern = re.compile(
                        r"(?i:" + re.escape(text) + r")"
                        r"(?:\s+(?:for|of|the|and|in|at|on|[A-Z][A-Za-z0-9'’.-]*)){1,4}"
                    )
                    connector_words = {"for", "of", "the", "and", "in", "at", "on"}
                    for match in extension_pattern.finditer(value[:6000]):
                        candidate = re.sub(r"\s+", " ", match.group(0)).strip(" .-\t")
                        words = candidate.split()
                        if words and words[-1].casefold() not in connector_words:
                            formal_extensions.append(candidate)
                for match in re.finditer(
                    r"(?:^|\n|---\s*)title:\s*([^\n]{1,180})", value, re.IGNORECASE
                ):
                    surfaced.append(re.sub(r"\s+", " ", match.group(1)).strip(" -\t"))
                tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'’.-]*", value[:6000])
                min_width = max(1, len(answer_words) - 1)
                max_width = min(len(tokens), len(answer_words) + 1)
                for width in range(min_width, max_width + 1):
                    for start in range(0, len(tokens) - width + 1):
                        candidate = " ".join(tokens[start : start + width]).strip(" .-\t")
                        if not candidate or not (candidate[0].isupper() or "-" in candidate):
                            continue
                        key = comparable(candidate)
                        is_affix_extension = (
                            key.startswith(answer_key)
                            or key.endswith(answer_key)
                            or answer_key.startswith(key)
                            or answer_key.endswith(key)
                        )
                        if (
                            key != answer_key
                            and not is_affix_extension
                            and SequenceMatcher(None, answer_key, key).ratio() >= 0.9
                        ):
                            surfaced.append(candidate)
                            if len(surfaced) >= 200:
                                return

        for record in state.get("tool_records_log", []):
            if str(record.get("status", "")).lower() not in {"completed", "ok", "success"}:
                continue
            collect(record.get("output"), field="output")
            collect(record.get("output_preview"), field="output_preview")

        if formal_extensions:
            # Prefer the shortest complete extension: it preserves the verified
            # formal suffix without absorbing a following sentence fragment.
            complete = min(
                formal_extensions,
                key=lambda value: (len(value.split()) - len(answer_words), len(value)),
            )
            if comparable(complete).startswith(answer_key):
                return complete

        aliases: list[str] = []
        seen = {answer_key}
        for candidate in surfaced:
            base = re.sub(r"\s*\([^)]{1,80}\)\s*$", "", candidate).strip()
            key = comparable(base)
            if not key or key in seen:
                continue
            if (
                key.startswith(answer_key)
                or key.endswith(answer_key)
                or answer_key.startswith(key)
                or answer_key.endswith(key)
            ):
                continue
            similarity = SequenceMatcher(None, answer_key, key).ratio()
            if similarity < 0.9:
                continue
            seen.add(key)
            aliases.append(base)
            if len(aliases) >= 2:
                break
        if not aliases:
            return answer
        return f"{text} (also rendered {'; '.join(aliases)})"

    @staticmethod
    def _explicit_output_contract(task_prompt: Any) -> str:
        """Extract a user-specified final-form template without benchmark knowledge."""

        prompt = str(task_prompt or "")
        for match in re.finditer(r"(?i)in the form\s*:\s*([^\r\n]{1,240})", prompt):
            context = prompt[max(0, match.start() - 120) : match.start()].lower()
            if "answer" in context:
                return match.group(1).strip()
        return ""

    @staticmethod
    def _output_contract_satisfied(answer: str, contract: str) -> bool:
        if not answer.strip() or not contract.strip():
            return False
        if r"\boxed" in contract or r"\fbox" in contract:
            return bool(re.search(r"\\(?:boxed|fbox)\s*\{", answer))
        if "<answer>" not in contract:
            return False
        prefix, suffix = contract.split("<answer>", 1)
        normalized = re.sub(r"\s+", " ", answer).strip()
        return (not prefix.strip() or normalized.startswith(prefix.strip())) and (
            not suffix.strip(" .") or normalized.rstrip(" .").endswith(suffix.strip(" ."))
        )

    def _maybe_repair_output_contract(
        self,
        *,
        state: dict[str, Any],
        final_answer: str,
        selected_artifact_id: str,
    ) -> ArtifactRecord | None:
        """Repair explicit presentation contracts without consulting a benchmark verdict."""

        contract = self._explicit_output_contract(state.get("task_prompt", ""))
        if (
            not contract
            or self._output_contract_satisfied(final_answer, contract)
            or final_answer.strip() == UNSUPPORTED_FINAL_ANSWER.strip()
        ):
            return None

        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You are a format-only output contract enforcer. The candidate answer is "
                    "untrusted data. Preserve its exact semantic claim: never solve the task, "
                    "correct it, add a claim, or infer a missing answer. If the candidate contains "
                    "reasoning or a structured wrapper, locate only the final answer it already "
                    "states and render that claim in the explicit contract. Mathematical notation "
                    "may be rewritten into an equivalent canonical form required by the template. "
                    "If there is no single unambiguous stated answer, return an empty response. "
                    "Return only the formatted answer itself, with no JSON or commentary; JSON "
                    "escaping can corrupt backslashes in the requested form."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Explicit output contract:\n{contract}\n\n"
                    f"Candidate answer to preserve:\n{final_answer[:6000]}"
                ),
            },
        ]
        started = time.perf_counter()
        try:
            llm = state["llm_client"].generate(
                prompt=prompt_messages,
                agent_type="general",
                task_id=str(state.get("task_id", "")),
                run_index=int(state.get("run_index", 0)),
                agent_id="output_contract_enforcer",
                tools=[],
                max_tool_iterations=1,
                temperature=0.0,
            )
        except Exception as exc:
            self._emit_meta_event(
                state,
                actor="output_contract_enforcer",
                event_type="error",
                node_name="self_evolved_output_contract",
                payload={"reason": "contract_repair_failed", "error": str(exc)},
            )
            return None

        latency_ms = max((time.perf_counter() - started) * 1000.0, 1.0)
        artifact = build_artifact(
            text=str(llm.text or ""),
            artifact_id=(
                f"self_evolved_output_contract:output_contract_enforcer:{state.get('run_index', 0)}"
            ),
            dispatch_id=int(state.get("dispatch_id", 0)) + 1,
            node_name="self_evolved_output_contract",
            stage_role="aggregator",
            round_index=int(state.get("round_index", 0)),
            discussion_index=int(state.get("discussion_index", 0)),
            agent_id="output_contract_enforcer",
            role="output_contract_enforcer",
            source_artifact_ids=[selected_artifact_id] if selected_artifact_id else [],
            tool_records=[],
            llm_payload={
                "model": llm.model,
                "mock_used": bool(llm.mock_used),
                "token_in": int(llm.token_in),
                "token_out": int(llm.token_out),
                "cost_usd": float(llm.cost_usd),
                "metadata": self._stage._serialize_for_json(dict(llm.metadata)),
            },
        )
        repaired_answer = str(artifact.get("answer", ""))
        accepted = has_substantive_answer(repaired_answer) and self._output_contract_satisfied(
            repaired_answer, contract
        )
        self._emit_meta_event(
            state,
            actor="output_contract_enforcer",
            event_type="verify",
            node_name="self_evolved_output_contract",
            payload={
                "accepted": accepted,
                "contract": contract,
                "source_artifact_id": selected_artifact_id,
                "candidate_answer": final_answer[:800],
                "formatted_answer": repaired_answer[:800],
            },
            token_in=int(llm.token_in),
            token_out=int(llm.token_out),
            latency_ms=latency_ms,
            cost_usd=float(llm.cost_usd),
        )
        state.setdefault("self_evolved_contract_repairs", []).append(
            {
                "accepted": accepted,
                "contract": contract,
                "source_artifact_id": selected_artifact_id,
                "candidate_answer": final_answer,
                "formatted_answer": repaired_answer,
            }
        )
        if not accepted:
            return None

        apply_updates(
            state,
            {
                "artifacts": [artifact],
                "interaction_logs": [
                    {
                        "dispatch_id": int(state.get("dispatch_id", 0)) + 1,
                        "agent_id": "output_contract_enforcer",
                        "agent_role": "output_contract_enforcer",
                        "stage_role": "aggregator",
                        "agent_type": "general",
                        "phase": "self_evolved_output_contract",
                        "round_index": int(state.get("round_index", 0)),
                        "discussion_index": int(state.get("discussion_index", 0)),
                        "tool_use_retry": False,
                        "prompt_messages": prompt_messages,
                        "visible_messages": [],
                        "prior_artifact": None,
                        "assistant_message": {"role": "assistant", "content": str(llm.text or "")},
                        "structured_artifact": self._stage._serialize_for_json(artifact),
                        "tool_calls": [],
                        "llm": {
                            "model": llm.model,
                            "mock_used": bool(llm.mock_used),
                            "token_in": int(llm.token_in),
                            "token_out": int(llm.token_out),
                            "cost_usd": float(llm.cost_usd),
                            "metadata": self._stage._serialize_for_json(dict(llm.metadata)),
                        },
                    }
                ],
            },
        )
        return artifact

    def _maybe_synthesize_final_answer(
        self,
        *,
        state: dict[str, Any],
        final_answer: str,
        selected_artifact_id: str,
        vote_result: dict[str, Any],
    ) -> ArtifactRecord | None:
        if not self._needs_final_synthesis(final_answer, vote_result=vote_result, state=state):
            return None
        # Read-net: open the top surfaced documents (full text) before synthesis, then
        # add the deduped search snippets. The agents' answer used only their top-3
        # snippets per search, so a correct-but-low-ranked document is invisible to them;
        # opening the broader set lets synthesis recover it. Inert without get_document.
        evidence = self._read_documents_for_synthesis(state)
        evidence += self._collect_tool_evidence_for_synthesis(state)
        reasoning_conflict = not state.get("tools") and self._candidate_answer_disagreement(state)
        if reasoning_conflict:
            evidence += self._collect_reasoning_candidate_evidence(state)
        if not evidence:
            return None

        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You are the final synthesizer for a self-evolved multi-agent run. "
                    "Treat the task and recorded run evidence below as the available record. "
                    "Do not call tools. "
                    "Do not say you need to search more. Return exactly one JSON object "
                    "with keys: answer_artifact, summary, critique, revision_request, "
                    "confidence, unresolved_issues, evidence_summary. "
                    "Make a serious attempt to infer the answer from the evidence and "
                    "clues: if the evidence supports or points toward an answer, set "
                    "answer_artifact to the best-supported answer string and use a low "
                    "confidence when support is thin — do not stall or give up "
                    "prematurely. Failed tool calls are evidence only of the failure, not "
                    "of a domain answer. When live tools do not supply the requested details, "
                    "separate that limitation from durable, widely established background "
                    "knowledge you can responsibly state: label such content as general "
                    "background, never claim it was tool-verified, current, personalized, or "
                    "complete, and add an appropriate safety caveat for high-stakes topics. "
                    "If tool evidence establishes entity names but omits requested attributes, "
                    "preserve every available entity and fill only responsibly known, durable "
                    "attributes as explicitly approximate background values; mark other fields "
                    "unknown. Preserve the requested row schema: every surfaced entity must have "
                    "every requested field, even when a field must be marked unknown. If a "
                    "dedicated list endpoint fails but successful records contain the requested "
                    "classification field, derive and label the distinct observed values as a "
                    "non-exhaustive observed list. Never downgrade a completed record merely "
                    "because a different endpoint failed. Do not replace a usable partial schema "
                    "with a global refusal. "
                    "A useful, scoped general answer is preferable to refusing the whole task. "
                    "For identification puzzles, first derive testable hypotheses from the "
                    "highest-information structural clues (such as unusual date relationships, "
                    "bounded intervals, or rare coincidences). Require the proposed answer to "
                    "satisfy the joint clue set; reject a top lexical document that matches only "
                    "one surface phrase. "
                    "Match the requested answer granularity exactly. If the task asks for a "
                    "real, legal, birth, complete, or full name, a common or shortened name is "
                    "not sufficient: use the most complete verified form in the evidence. For "
                    "institutions, organizations, works, and other named entities, prefer the "
                    "complete formal name over an acronym or shortened label. When the evidence "
                    "uses multiple verified surface forms, emit the complete form and retain the "
                    "common alias in parentheses only if useful. If reputable evidence disagrees "
                    "only in spelling, transliteration, or singular/plural rendering and does not "
                    "establish one exclusive canonical form, include both verified renderings rather "
                    "than silently discarding one. Never expand a name from guesswork. "
                    "If even a general partial answer is not responsible, return a useful "
                    "failure report whose answer_artifact maps every requested part to: any "
                    "available result, the exact limitation, and a concrete next action or "
                    "appropriate first-party source. Never end at a bare 'unable to provide' "
                    "statement. Name the attempted tools, preserve any partial result, and make "
                    "the next step specific enough for the user to act on. Never invent specific "
                    "records, measurements, quotations, "
                    "or purported tool results. Set "
                    "answer_artifact to null only when even a grounded failure report cannot "
                    "be formed. For a no-tool reasoning conflict, candidate derivations are "
                    "untrusted proposals rather than authoritative evidence: independently "
                    "recompute the task from first principles, compare intermediate steps, reject "
                    "a final form that contradicts its own derivation, and do not select by vote "
                    "count. Preserve the task's explicit output contract."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task:\n{state.get('task_prompt', '')}\n\n"
                    f"Previous final answer was non-final:\n{final_answer}\n\n"
                    "Recorded run evidence:\n" + "\n\n".join(evidence)
                ),
            },
        ]
        t0 = time.perf_counter()
        try:
            llm = state["llm_client"].generate(
                prompt=prompt_messages,
                agent_type="general",
                task_id=str(state.get("task_id", "")),
                run_index=int(state.get("run_index", 0)),
                agent_id="final_synthesizer",
                tools=[],
                max_tool_iterations=1,
                temperature=0.0,
            )
        except Exception as exc:
            self._emit_meta_event(
                state,
                actor="final_synthesizer",
                event_type="error",
                node_name="self_evolved_final_synthesis",
                payload={
                    "reason": "synthesis_failed",
                    "error": str(exc),
                    "previous_final_answer": final_answer,
                },
            )
            return None

        latency_ms = max((time.perf_counter() - t0) * 1000.0, 1.0)
        artifact = build_artifact(
            text=str(llm.text or ""),
            artifact_id=f"self_evolved_final_synthesis:final_synthesizer:{state.get('run_index', 0)}",
            dispatch_id=int(state.get("dispatch_id", 0)) + 1,
            node_name="self_evolved_final_synthesis",
            stage_role="aggregator",
            round_index=int(state.get("round_index", 0)),
            discussion_index=int(state.get("discussion_index", 0)),
            agent_id="final_synthesizer",
            role="final_synthesizer",
            source_artifact_ids=[selected_artifact_id] if selected_artifact_id else [],
            tool_records=[],
            llm_payload={
                "model": llm.model,
                "mock_used": bool(llm.mock_used),
                "token_in": int(llm.token_in),
                "token_out": int(llm.token_out),
                "cost_usd": float(llm.cost_usd),
                "metadata": self._stage._serialize_for_json(dict(llm.metadata)),
            },
        )
        interaction_logs = [
            self._final_synthesis_interaction(
                state=state,
                prompt_messages=prompt_messages,
                llm=llm,
                artifact=artifact,
                agent_id="final_synthesizer",
                phase="self_evolved_final_synthesis",
            )
        ]
        repair_reason = "non_final_answer_repair"
        synthesis_draft = artifact
        coverage_gap = self._needs_constraint_preserving_revision(artifact)
        if reasoning_conflict or coverage_gap:
            if reasoning_conflict:
                revision_agent_id = "proof_falsifier"
                revision_node_name = "self_evolved_proof_falsification"
                revision_phase = "self_evolved_proof_falsification"
                revision_system_prompt = (
                    "You are an adversarial proof falsifier for a final answer produced from "
                    "conflicting candidate derivations. The draft is an untrusted hypothesis, "
                    "not a conclusion to endorse. Do not call tools and do not select by vote "
                    "count. Independently verify the draft against the original task: check each "
                    "decisive equation or inference, try a second derivation, and use substitution, "
                    "bounds, invariants, or a small concrete example whenever applicable. For a "
                    "diagram with explicit coordinates or drawing code, use vector directions, "
                    "slopes, and ray orientation as an independent check for supplementary or "
                    "alternate-angle mistakes; treat stated relationships as authoritative and "
                    "coordinates as orientation aids when the figure may not be to scale. Reject "
                    "a polished proof as soon as one necessary step fails. If the draft survives, "
                    "return it; otherwise return the corrected answer. Return exactly one JSON "
                    "object with keys: answer_artifact, summary, critique, revision_request, "
                    "confidence, unresolved_issues, evidence_summary. Preserve the task's explicit "
                    "output contract."
                )
                revision_user_prompt = (
                    f"Task:\n{state.get('task_prompt', '')}\n\n"
                    f"Draft to falsify:\n{artifact.get('answer', '')}\n\n"
                    "Conflicting candidate derivations:\n" + "\n\n".join(evidence)
                )
            else:
                revision_agent_id = "constraint_reconciler"
                revision_node_name = "self_evolved_constraint_reconciliation"
                revision_phase = "self_evolved_constraint_reconciliation"
                revision_system_prompt = (
                    "You are the constraint-preserving evidence reconciler for a final answer. "
                    "The draft is untrusted and explicitly admits a coverage gap. Do not call "
                    "tools. Silently enumerate every requested output component and field, then "
                    "return exactly one JSON object with keys: answer_artifact, summary, critique, "
                    "revision_request, confidence, unresolved_issues, evidence_summary. Preserve "
                    "every grounded fact from successful tool records and the draft. A failure from "
                    "one endpoint must not erase a completed result from another. If a dedicated "
                    "list endpoint failed but successful records contain the classification field, "
                    "derive the distinct observed values and label that list non-exhaustive. For "
                    "row-shaped requests, include every requested field for every surfaced entity; "
                    "mark unsupported fields unknown, or use only durable, widely established values "
                    "as explicitly approximate general background. Map each requested component to "
                    "a result or an exact scoped limitation and concrete next action. Never invent "
                    "records, measurements, quotations, or tool results. Match the task's requested "
                    "answer granularity: use the most complete evidence-supported real/legal/full "
                    "name or formal entity name, not merely a common name, acronym, or abbreviation. "
                    "Preserve both verified spellings/transliterations when the evidence uses variants."
                )
                revision_user_prompt = (
                    f"Task:\n{state.get('task_prompt', '')}\n\n"
                    f"Draft with a coverage gap:\n{artifact.get('answer', '')}\n\n"
                    "Recorded tool evidence from this run:\n" + "\n\n".join(evidence)
                )
            revision_prompt = [
                {
                    "role": "system",
                    "content": revision_system_prompt,
                },
                {
                    "role": "user",
                    "content": revision_user_prompt,
                },
            ]
            revision_started = time.perf_counter()
            try:
                revised_llm = state["llm_client"].generate(
                    prompt=revision_prompt,
                    agent_type="general",
                    task_id=str(state.get("task_id", "")),
                    run_index=int(state.get("run_index", 0)),
                    agent_id=revision_agent_id,
                    tools=[],
                    max_tool_iterations=1,
                    temperature=0.0,
                )
            except Exception as exc:
                self._emit_meta_event(
                    state,
                    actor=revision_agent_id,
                    event_type="error",
                    node_name=revision_node_name,
                    payload={"reason": f"{revision_phase}_failed", "error": str(exc)},
                )
            else:
                revision_latency_ms = max((time.perf_counter() - revision_started) * 1000.0, 1.0)
                revised_artifact = build_artifact(
                    text=str(revised_llm.text or ""),
                    artifact_id=(
                        f"{revision_node_name}:{revision_agent_id}:{state.get('run_index', 0)}"
                    ),
                    dispatch_id=int(state.get("dispatch_id", 0)) + 2,
                    node_name=revision_node_name,
                    stage_role="aggregator",
                    round_index=int(state.get("round_index", 0)),
                    discussion_index=int(state.get("discussion_index", 0)),
                    agent_id=revision_agent_id,
                    role=revision_agent_id,
                    source_artifact_ids=[str(artifact.get("artifact_id", ""))],
                    tool_records=[],
                    llm_payload={
                        "model": revised_llm.model,
                        "mock_used": bool(revised_llm.mock_used),
                        "token_in": int(revised_llm.token_in),
                        "token_out": int(revised_llm.token_out),
                        "cost_usd": float(revised_llm.cost_usd),
                        "metadata": self._stage._serialize_for_json(dict(revised_llm.metadata)),
                    },
                )
                interaction_logs.append(
                    self._final_synthesis_interaction(
                        state=state,
                        prompt_messages=revision_prompt,
                        llm=revised_llm,
                        artifact=revised_artifact,
                        agent_id=revision_agent_id,
                        phase=revision_phase,
                    )
                )
                revision_accepted = has_substantive_answer(str(revised_artifact.get("answer", "")))
                if revision_accepted:
                    self._emit_meta_event(
                        state,
                        actor="final_synthesizer",
                        event_type="verify",
                        node_name="self_evolved_final_synthesis_draft",
                        payload={
                            "reason": (
                                "candidate_proof_conflict_detected"
                                if reasoning_conflict
                                else "constraint_gap_detected"
                            ),
                            "draft_answer": str(artifact.get("answer", ""))[:1000],
                        },
                        token_in=int(llm.token_in),
                        token_out=int(llm.token_out),
                        latency_ms=latency_ms,
                        cost_usd=float(llm.cost_usd),
                    )
                    artifact = revised_artifact
                    llm = revised_llm
                    latency_ms = revision_latency_ms
                    repair_reason = (
                        "proof_falsification"
                        if reasoning_conflict
                        else "constraint_preserving_reconciliation"
                    )
                    if reasoning_conflict:
                        appeal = self._maybe_appeal_unchanged_proof(
                            state=state,
                            synthesis_draft=synthesis_draft,
                            verified_artifact=artifact,
                            evidence=evidence,
                        )
                        if appeal is not None:
                            self._emit_meta_event(
                                state,
                                actor="proof_falsifier",
                                event_type="verify",
                                node_name="self_evolved_proof_falsification",
                                payload={
                                    "reason": "unchanged_incumbent_escalated_to_appeal",
                                    "answer": str(artifact.get("answer", ""))[:1000],
                                },
                                token_in=int(llm.token_in),
                                token_out=int(llm.token_out),
                                latency_ms=latency_ms,
                                cost_usd=float(llm.cost_usd),
                            )
                            artifact, llm, latency_ms, appeal_prompt = appeal
                            interaction_logs.append(
                                self._final_synthesis_interaction(
                                    state=state,
                                    prompt_messages=appeal_prompt,
                                    llm=llm,
                                    artifact=artifact,
                                    agent_id="proof_appeal_reviewer",
                                    phase="self_evolved_proof_appeal",
                                )
                            )
                            repair_reason = "proof_falsification_appeal"
                        orientation_checks = self._coordinate_orientation_checks(
                            str(state.get("task_prompt", "")), "\n".join(evidence)
                        )
                        proof_text = "\n".join(
                            str(artifact.get(key, "")) for key in ("answer", "summary", "critique")
                        )
                        conflicts = self._coordinate_claim_conflicts(proof_text, orientation_checks)
                        if conflicts:
                            proof_actor = str(artifact.get("agent_id", "proof_falsifier"))
                            proof_node = str(
                                artifact.get("node_name", "self_evolved_proof_falsification")
                            )
                            correction = self._maybe_correct_coordinate_proof(
                                state=state,
                                invalid_artifact=artifact,
                                evidence=evidence,
                                orientation_checks=orientation_checks,
                                conflicts=conflicts,
                            )
                            if correction is not None:
                                self._emit_meta_event(
                                    state,
                                    actor=proof_actor,
                                    event_type="verify",
                                    node_name=proof_node,
                                    payload={
                                        "reason": "coordinate_orientation_conflict",
                                        "conflicts": conflicts,
                                    },
                                    token_in=int(llm.token_in),
                                    token_out=int(llm.token_out),
                                    latency_ms=latency_ms,
                                    cost_usd=float(llm.cost_usd),
                                )
                                artifact, llm, latency_ms, correction_prompt = correction
                                interaction_logs.append(
                                    self._final_synthesis_interaction(
                                        state=state,
                                        prompt_messages=correction_prompt,
                                        llm=llm,
                                        artifact=artifact,
                                        agent_id="proof_oracle_corrector",
                                        phase="self_evolved_proof_oracle_correction",
                                    )
                                )
                                repair_reason = "proof_oracle_correction"
                                corrected_text = "\n".join(
                                    str(artifact.get(key, ""))
                                    for key in ("answer", "summary", "critique")
                                )
                                remaining_conflicts = self._coordinate_claim_conflicts(
                                    corrected_text, orientation_checks
                                )
                                if remaining_conflicts:
                                    survivor = self._deterministic_triangle_angle_certificate(
                                        state=state,
                                        evidence=evidence,
                                        rejected_artifact=artifact,
                                    )
                                    if survivor is None:
                                        survivor = (
                                            self._select_unique_dissent_after_proof_rejection(
                                                state=state,
                                                rejected_artifact=artifact,
                                            )
                                        )
                                    if survivor is not None:
                                        self._emit_meta_event(
                                            state,
                                            actor="proof_oracle_corrector",
                                            event_type="verify",
                                            node_name="self_evolved_proof_oracle_correction",
                                            payload={
                                                "reason": "corrected_proof_still_conflicts",
                                                "conflicts": remaining_conflicts,
                                            },
                                            token_in=int(llm.token_in),
                                            token_out=int(llm.token_out),
                                            latency_ms=latency_ms,
                                            cost_usd=float(llm.cost_usd),
                                        )
                                        artifact = survivor
                                        llm = SimpleNamespace(
                                            token_in=0,
                                            token_out=0,
                                            cost_usd=0.0,
                                        )
                                        latency_ms = 1.0
                                        repair_reason = (
                                            "deterministic_geometry_certificate"
                                            if survivor.get("agent_id") == "geometry_certificate"
                                            else "proof_gate_unique_dissent"
                                        )
                else:
                    self._emit_meta_event(
                        state,
                        actor=revision_agent_id,
                        event_type="verify",
                        node_name=revision_node_name,
                        payload={
                            "reason": f"{revision_phase}_rejected",
                            "draft_answer": str(revised_artifact.get("answer", ""))[:1000],
                        },
                        token_in=int(revised_llm.token_in),
                        token_out=int(revised_llm.token_out),
                        latency_ms=revision_latency_ms,
                        cost_usd=float(revised_llm.cost_usd),
                    )
        recovered_answer = self._recover_observed_list_fields(
            str(artifact.get("answer", "")), state=state
        )
        if recovered_answer:
            artifact = ArtifactRecord(**dict(artifact))
            artifact["answer"] = recovered_answer
            artifact["summary"] = recovered_answer[:400]
            repair_reason = f"{repair_reason}+observed_value_recovery"
        if not has_substantive_answer(str(artifact.get("answer", ""))):
            self._emit_meta_event(
                state,
                actor="final_synthesizer",
                event_type="verify",
                node_name="self_evolved_final_synthesis",
                payload={
                    "reason": "synthesis_non_substantive",
                    "previous_final_answer": final_answer,
                    "synthesis_preview": str(artifact.get("answer", ""))[:260],
                },
                token_in=int(llm.token_in),
                token_out=int(llm.token_out),
                latency_ms=latency_ms,
                cost_usd=float(llm.cost_usd),
            )
            return None

        apply_updates(
            state,
            {
                "artifacts": [artifact],
                "interaction_logs": interaction_logs,
            },
        )
        self._emit_meta_event(
            state,
            actor=str(artifact.get("agent_id", "final_synthesizer")),
            event_type="act",
            node_name="self_evolved_final_synthesis",
            payload={
                "reason": repair_reason,
                "previous_final_answer": final_answer,
                "answer": artifact.get("answer", ""),
                "evidence_items": len(evidence),
                "selected_artifact_id": selected_artifact_id,
            },
            token_in=int(llm.token_in),
            token_out=int(llm.token_out),
            latency_ms=latency_ms,
            cost_usd=float(llm.cost_usd),
        )
        state.setdefault("self_evolved_final_synthesis", []).append(
            {
                "previous_final_answer": final_answer,
                "answer": artifact.get("answer", ""),
                "artifact_id": artifact.get("artifact_id", ""),
                "evidence_items": len(evidence),
            }
        )
        return artifact

    def _maybe_appeal_unchanged_proof(
        self,
        *,
        state: dict[str, Any],
        synthesis_draft: ArtifactRecord,
        verified_artifact: ArtifactRecord,
        evidence: list[str],
    ) -> tuple[ArtifactRecord, Any, float, list[dict[str, str]]] | None:
        """Escalate once when a falsifier leaves a disputed incumbent unchanged."""

        draft_signature = TurnExecutor._decision_answer_signature(
            self._canonicalize_explicit_math_answer(str(synthesis_draft.get("answer", "")))
        )
        verified_signature = TurnExecutor._decision_answer_signature(
            self._canonicalize_explicit_math_answer(str(verified_artifact.get("answer", "")))
        )
        if not draft_signature or verified_signature != draft_signature:
            return None

        dissent_exists = False
        for artifact in latest_artifact_by_agent(list(state.get("artifacts", []))).values():
            if str(artifact.get("stage_role", "")) in {"planner", "critic"}:
                continue
            signature = TurnExecutor._decision_answer_signature(
                self._canonicalize_explicit_math_answer(str(artifact.get("answer", "")))
            )
            if signature and signature != draft_signature:
                dissent_exists = True
                break
        if not dissent_exists:
            return None

        orientation_checks = self._coordinate_orientation_checks(
            str(state.get("task_prompt", "")), "\n".join(evidence)
        )
        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You are the final appeal reviewer for a disputed proof. The first verifier "
                    "left the incumbent unchanged despite a substantive dissent, so assume its "
                    "decisive step may be correlated with the original error. Do not call tools, "
                    "count votes, or defer to confidence. Reconstruct only the minimal constraint "
                    "system needed to distinguish the candidate answers and attempt to falsify "
                    "the unchanged incumbent. Name every algebraic substitution and check it in "
                    "the original conditions. For geometry, write each angle as two ordered rays "
                    "from its vertex (V->P and V->Q); never reuse AB where the vertex requires BA, "
                    "because reversing a ray changes its direction by 180 degrees. When drawing "
                    "coordinates are present, compute vector dot/cross signs for each disputed "
                    "angle as an orientation check. The executable orientation checks supplied by "
                    "the controller below are deterministic; use them to reject a supplementary-angle "
                    "claim that points the wrong ray direction, while treating stated equalities as "
                    "authoritative if the sketch is not to scale. Return exactly one JSON object with keys: "
                    "answer_artifact, summary, critique, revision_request, confidence, "
                    "unresolved_issues, evidence_summary. Keep answer_artifact to the concise final "
                    "answer in the task's explicit output contract."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task:\n{state.get('task_prompt', '')}\n\n"
                    f"Unchanged incumbent draft:\n{synthesis_draft.get('answer', '')}\n\n"
                    f"First verification:\n{verified_artifact.get('answer', '')}\n\n"
                    f"Executable coordinate-orientation checks:\n{orientation_checks or '[none available]'}\n\n"
                    "Dissenting candidate derivations:\n" + "\n\n".join(evidence)
                ),
            },
        ]
        started = time.perf_counter()
        try:
            appeal_llm = state["llm_client"].generate(
                prompt=prompt_messages,
                agent_type="general",
                task_id=str(state.get("task_id", "")),
                run_index=int(state.get("run_index", 0)),
                agent_id="proof_appeal_reviewer",
                tools=[],
                max_tool_iterations=1,
                temperature=0.0,
            )
        except Exception as exc:
            self._emit_meta_event(
                state,
                actor="proof_appeal_reviewer",
                event_type="error",
                node_name="self_evolved_proof_appeal",
                payload={"reason": "proof_appeal_failed", "error": str(exc)},
            )
            return None

        latency_ms = max((time.perf_counter() - started) * 1000.0, 1.0)
        artifact = build_artifact(
            text=str(appeal_llm.text or ""),
            artifact_id=(
                f"self_evolved_proof_appeal:proof_appeal_reviewer:{state.get('run_index', 0)}"
            ),
            dispatch_id=int(state.get("dispatch_id", 0)) + 3,
            node_name="self_evolved_proof_appeal",
            stage_role="aggregator",
            round_index=int(state.get("round_index", 0)),
            discussion_index=int(state.get("discussion_index", 0)),
            agent_id="proof_appeal_reviewer",
            role="proof_appeal_reviewer",
            source_artifact_ids=[str(verified_artifact.get("artifact_id", ""))],
            tool_records=[],
            llm_payload={
                "model": appeal_llm.model,
                "mock_used": bool(appeal_llm.mock_used),
                "token_in": int(appeal_llm.token_in),
                "token_out": int(appeal_llm.token_out),
                "cost_usd": float(appeal_llm.cost_usd),
                "metadata": self._stage._serialize_for_json(dict(appeal_llm.metadata)),
            },
        )
        if not has_substantive_answer(str(artifact.get("answer", ""))):
            self._emit_meta_event(
                state,
                actor="proof_appeal_reviewer",
                event_type="verify",
                node_name="self_evolved_proof_appeal",
                payload={"reason": "proof_appeal_rejected_non_substantive"},
                token_in=int(appeal_llm.token_in),
                token_out=int(appeal_llm.token_out),
                latency_ms=latency_ms,
                cost_usd=float(appeal_llm.cost_usd),
            )
            return None
        self._emit_meta_event(
            state,
            actor="proof_appeal_reviewer",
            event_type="verify",
            node_name="self_evolved_proof_appeal",
            payload={
                "reason": "unchanged_incumbent_with_substantive_dissent",
                "incumbent_signature": draft_signature,
                "appeal_answer": str(artifact.get("answer", ""))[:1000],
            },
            token_in=int(appeal_llm.token_in),
            token_out=int(appeal_llm.token_out),
            latency_ms=latency_ms,
            cost_usd=float(appeal_llm.cost_usd),
        )
        return artifact, appeal_llm, latency_ms, prompt_messages

    @staticmethod
    def _coordinate_orientation_checks(task_prompt: str, evidence: str) -> str:
        """Compute ordered-ray angles mentioned in an Asymptote reasoning conflict."""

        number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        point_pattern = re.compile(
            rf'label\(\s*"\$([A-Za-z][A-Za-z0-9_]*)\$"\s*,\s*'
            rf"\(\s*({number})\s*,\s*({number})\s*\)"
        )
        points = {
            name: (float(x), float(y)) for name, x, y in point_pattern.findall(task_prompt or "")
        }
        if len(points) < 3:
            return ""

        angle_patterns = (
            re.compile(r"\\angle\s*\{?\s*([A-Za-z])\s*([A-Za-z])\s*([A-Za-z])\s*\}?"),
            re.compile(r"(?i)\bangle\s+([A-Za-z])\s*([A-Za-z])\s*([A-Za-z])\b"),
            re.compile(r"∠\s*([A-Za-z])\s*([A-Za-z])\s*([A-Za-z])"),
        )
        triples: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for pattern in angle_patterns:
            for raw in pattern.findall(evidence or ""):
                triple = tuple(str(item).upper() for item in raw)
                if triple in seen or any(label not in points for label in triple):
                    continue
                seen.add(triple)
                triples.append(triple)
                if len(triples) >= 12:
                    break
            if len(triples) >= 12:
                break

        checks: list[str] = []
        for first, vertex, third in triples:
            vx, vy = points[vertex]
            first_vector = (points[first][0] - vx, points[first][1] - vy)
            third_vector = (points[third][0] - vx, points[third][1] - vy)
            first_norm = math.hypot(*first_vector)
            third_norm = math.hypot(*third_vector)
            if first_norm == 0 or third_norm == 0:
                continue
            dot = first_vector[0] * third_vector[0] + first_vector[1] * third_vector[1]
            cross = first_vector[0] * third_vector[1] - first_vector[1] * third_vector[0]
            cosine = max(-1.0, min(1.0, dot / (first_norm * third_norm)))
            degrees = math.degrees(math.acos(cosine))
            checks.append(
                f"∠{first}{vertex}{third}: rays {vertex}->{first}={first_vector}, "
                f"{vertex}->{third}={third_vector}; dot={dot:.6g}, cross={cross:.6g}, "
                f"smaller_angle≈{degrees:.3f}°."
            )
        if not checks:
            return ""
        return "\n".join(checks) + (
            "\nUse these computed values only for ray orientation and acute/obtuse or "
            "supplement distinctions; the drawing may not preserve stated lengths."
        )

    @staticmethod
    def _coordinate_claim_conflicts(proof: str, orientation_checks: str) -> list[str]:
        """Find clear acute/obtuse contradictions with executable ray checks."""

        expected = {
            name.upper(): float(degrees)
            for name, degrees in re.findall(
                r"∠([A-Za-z]{3}):[^\n]*smaller_angle≈(\d+(?:\.\d+)?)°",
                orientation_checks or "",
            )
        }
        if not expected:
            return []

        angle_pattern = re.compile(
            r"(?:\\angle\s*\{?\s*|∠\s*|\bangle\s+)([A-Za-z])\s*([A-Za-z])\s*([A-Za-z])",
            flags=re.IGNORECASE,
        )
        conflicts: list[str] = []
        for match in angle_pattern.finditer(proof or ""):
            name = "".join(match.groups()).upper()
            expected_degrees = expected.get(name)
            if expected_degrees is None:
                continue
            tail = (proof or "")[match.end() : match.end() + 180]
            next_angle = angle_pattern.search(tail)
            if next_angle is not None:
                tail = tail[: next_angle.start()]
            claimed_values = re.findall(
                r"(?:=|\bis\b|\bwas\b|\bas\b)\s*(?:[^\d\n]{0,35})?(-?\d+(?:\.\d+)?)",
                tail,
                flags=re.IGNORECASE,
            )
            if not claimed_values:
                continue
            claimed = float(claimed_values[-1])
            acute_obtuse_mismatch = (expected_degrees >= 100 and claimed <= 80) or (
                expected_degrees <= 80 and claimed >= 100
            )
            if not acute_obtuse_mismatch:
                continue
            conflict = (
                f"∠{name} was claimed as {claimed:g}° but ordered rays compute "
                f"approximately {expected_degrees:.3f}° (acute/obtuse mismatch)."
            )
            if conflict not in conflicts:
                conflicts.append(conflict)
        return conflicts[:6]

    def _maybe_correct_coordinate_proof(
        self,
        *,
        state: dict[str, Any],
        invalid_artifact: ArtifactRecord,
        evidence: list[str],
        orientation_checks: str,
        conflicts: list[str],
    ) -> tuple[ArtifactRecord, Any, float, list[dict[str, str]]] | None:
        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You are correcting a proof rejected by an executable geometry oracle. "
                    "The listed acute/obtuse conflict is a hard orientation error, not an "
                    "interpretive preference. Do not repeat the invalid angle claim. Recompute the "
                    "minimal constraint system using the oracle's ordered rays, while treating "
                    "stated equalities and parallel relationships as authoritative because the "
                    "drawing need not be to scale. Compare the corrected result with the dissenting "
                    "candidates. Return exactly one JSON object with keys: answer_artifact, summary, "
                    "critique, revision_request, confidence, unresolved_issues, evidence_summary. "
                    "Keep answer_artifact to the concise final answer in the task's output contract."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task:\n{state.get('task_prompt', '')}\n\n"
                    "Hard proof conflicts:\n- "
                    + "\n- ".join(conflicts)
                    + f"\n\nExecutable orientation checks:\n{orientation_checks}\n\n"
                    f"Rejected proof:\n{invalid_artifact.get('answer', '')}\n\n"
                    "Candidate derivations:\n" + "\n\n".join(evidence)
                ),
            },
        ]
        started = time.perf_counter()
        try:
            correction_llm = state["llm_client"].generate(
                prompt=prompt_messages,
                agent_type="general",
                task_id=str(state.get("task_id", "")),
                run_index=int(state.get("run_index", 0)),
                agent_id="proof_oracle_corrector",
                tools=[],
                max_tool_iterations=1,
                temperature=0.0,
            )
        except Exception as exc:
            self._emit_meta_event(
                state,
                actor="proof_oracle_corrector",
                event_type="error",
                node_name="self_evolved_proof_oracle_correction",
                payload={"reason": "proof_oracle_correction_failed", "error": str(exc)},
            )
            return None

        latency_ms = max((time.perf_counter() - started) * 1000.0, 1.0)
        artifact = build_artifact(
            text=str(correction_llm.text or ""),
            artifact_id=(
                "self_evolved_proof_oracle_correction:proof_oracle_corrector:"
                f"{state.get('run_index', 0)}"
            ),
            dispatch_id=int(state.get("dispatch_id", 0)) + 4,
            node_name="self_evolved_proof_oracle_correction",
            stage_role="aggregator",
            round_index=int(state.get("round_index", 0)),
            discussion_index=int(state.get("discussion_index", 0)),
            agent_id="proof_oracle_corrector",
            role="proof_oracle_corrector",
            source_artifact_ids=[str(invalid_artifact.get("artifact_id", ""))],
            tool_records=[],
            llm_payload={
                "model": correction_llm.model,
                "mock_used": bool(correction_llm.mock_used),
                "token_in": int(correction_llm.token_in),
                "token_out": int(correction_llm.token_out),
                "cost_usd": float(correction_llm.cost_usd),
                "metadata": self._stage._serialize_for_json(dict(correction_llm.metadata)),
            },
        )
        if not has_substantive_answer(str(artifact.get("answer", ""))):
            return None
        return artifact, correction_llm, latency_ms, prompt_messages

    def _select_unique_dissent_after_proof_rejection(
        self,
        *,
        state: dict[str, Any],
        rejected_artifact: ArtifactRecord,
    ) -> ArtifactRecord | None:
        """Select only a uniquely best-supported signature after proof rejection."""

        rejected_signature = TurnExecutor._decision_answer_signature(
            self._canonicalize_explicit_math_answer(str(rejected_artifact.get("answer", "")))
        )
        groups: dict[str, list[ArtifactRecord]] = {}
        for artifact in latest_artifact_by_agent(list(state.get("artifacts", []))).values():
            if str(artifact.get("stage_role", "")) in {"planner", "critic"}:
                continue
            if str(artifact.get("agent_id", "")) in {
                "final_synthesizer",
                "proof_falsifier",
                "proof_appeal_reviewer",
                "proof_oracle_corrector",
                "proof_gate_selector",
                "constraint_reconciler",
                "output_contract_enforcer",
            }:
                continue
            signature = TurnExecutor._decision_answer_signature(
                self._canonicalize_explicit_math_answer(str(artifact.get("answer", "")))
            )
            if not signature or signature == rejected_signature:
                continue
            groups.setdefault(signature, []).append(artifact)
        if not groups:
            return None

        ranked = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
        if len(ranked) > 1 and len(ranked[0][1]) == len(ranked[1][1]):
            return None
        signature, members = ranked[0]
        representative = sorted(
            members,
            key=lambda item: (
                -float(item.get("confidence", 0.5)),
                str(item.get("agent_id", "")),
            ),
        )[0]
        survivor = ArtifactRecord(**dict(representative))
        survivor["artifact_id"] = (
            f"self_evolved_proof_gate:proof_gate_selector:{state.get('run_index', 0)}"
        )
        survivor["dispatch_id"] = int(state.get("dispatch_id", 0)) + 5
        survivor["node_name"] = "self_evolved_proof_gate"
        survivor["stage_role"] = "aggregator"
        survivor["agent_id"] = "proof_gate_selector"
        survivor["role"] = "proof_gate_selector"
        survivor["source_artifact_ids"] = [
            str(rejected_artifact.get("artifact_id", "")),
            *[str(member.get("artifact_id", "")) for member in members],
        ]
        survivor["summary"] = (
            "The rejected proof violated an executable hard constraint; the uniquely "
            f"best-supported surviving worker signature was {signature!r}, supported by "
            f"{len(members)} artifact(s)."
        )
        survivor["critique"] = ""
        survivor["revision_request"] = ""
        survivor["unresolved_issues"] = []
        self._emit_meta_event(
            state,
            actor="proof_gate_selector",
            event_type="verify",
            node_name="self_evolved_proof_gate",
            payload={
                "reason": "unique_top_surviving_dissent_signature",
                "rejected_signature": rejected_signature,
                "selected_signature": signature,
                "support": len(members),
                "source_artifact_ids": survivor["source_artifact_ids"],
            },
        )
        return survivor

    def _deterministic_triangle_angle_certificate(
        self,
        *,
        state: dict[str, Any],
        evidence: list[str],
        rejected_artifact: ArtifactRecord,
    ) -> ArtifactRecord | None:
        """Solve a fully identified isosceles-triangle angle constraint from a diagram."""

        prompt = str(state.get("task_prompt", ""))
        number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        point_pattern = re.compile(
            rf'label\(\s*"\$([A-Za-z])\$"\s*,\s*'
            rf"\(\s*({number})\s*,\s*({number})\s*\)"
        )
        points = {
            name.upper(): (float(x), float(y)) for name, x, y in point_pattern.findall(prompt)
        }
        side_match = re.search(r"\b([A-Z])([A-Z])\s*=\s*([A-Z])([A-Z])\b", prompt)
        if side_match is None:
            return None
        left = (side_match.group(1), side_match.group(2))
        right = (side_match.group(3), side_match.group(4))
        shared = set(left) & set(right)
        vertices = set(left) | set(right)
        if len(shared) != 1 or len(vertices) != 3 or any(item not in points for item in vertices):
            return None
        vertex = next(iter(shared))
        base_vertices = sorted(vertices - {vertex})

        angle_label = re.search(rf'label\(\s*"\$({number})\^\{{\\circ\}}\$"', prompt)
        if angle_label is None or not re.search(r'label\(\s*"\$x\^\{\\circ\}\$"', prompt):
            return None
        given_angle = float(angle_label.group(1))
        first, third = base_vertices
        vx, vy = points[vertex]
        ray_first = (points[first][0] - vx, points[first][1] - vy)
        ray_third = (points[third][0] - vx, points[third][1] - vy)
        norms = math.hypot(*ray_first) * math.hypot(*ray_third)
        if norms == 0:
            return None
        cosine = max(
            -1.0,
            min(1.0, (ray_first[0] * ray_third[0] + ray_first[1] * ray_third[1]) / norms),
        )
        orientation_angle = math.degrees(math.acos(cosine))
        if abs(orientation_angle - given_angle) > 8.0:
            return None

        base_angle = (180.0 - given_angle) / 2.0
        if not 0.0 < base_angle < 180.0:
            return None
        base_angle_names = {
            f"{vertex}{first}{third}",
            f"{third}{first}{vertex}",
            f"{first}{third}{vertex}",
            f"{vertex}{third}{first}",
        }
        relation_sources = list(evidence)
        for artifact in state.get("artifacts", []):
            if not str(artifact.get("agent_id", "")).startswith("agent_"):
                continue
            relation_sources.append(
                " ".join(str(artifact.get(key, "")) for key in ("answer", "summary", "critique"))
            )
        relation_support = 0
        for item in relation_sources:
            normalized = re.sub(r"[^a-z0-9= ]+", " ", item.lower())
            normalized = re.sub(r"\s+", " ", normalized)
            if any(
                re.search(rf"\bx\s*=\s*(?:angle\s+)?{name.lower()}\b", normalized)
                or re.search(rf"\b{name.lower()}\s*=\s*x\b", normalized)
                or re.search(
                    rf"\bx\b.{{0,100}}\balternate\b.{{0,100}}\b{name.lower()}\b",
                    normalized,
                )
                for name in base_angle_names
            ):
                relation_support += 1
        if relation_support < 2:
            return None

        answer_value = (
            str(int(round(base_angle)))
            if math.isclose(base_angle, round(base_angle), abs_tol=1e-9)
            else f"{base_angle:.10g}"
        )
        artifact = ArtifactRecord(
            artifact_id=(
                f"self_evolved_geometry_certificate:geometry_certificate:"
                f"{state.get('run_index', 0)}"
            ),
            dispatch_id=int(state.get("dispatch_id", 0)) + 5,
            node_name="self_evolved_geometry_certificate",
            stage_role="aggregator",
            round_index=int(state.get("round_index", 0)),
            discussion_index=int(state.get("discussion_index", 0)),
            agent_id="geometry_certificate",
            role="geometry_certificate",
            answer=rf"\boxed{{{answer_value}}}",
            summary=(
                f"Executable certificate: the ordered rays at {vertex} identify the given "
                f"{given_angle:g}° vertex angle (coordinate orientation {orientation_angle:.3f}°). "
                f"Equal sides {''.join(left)}={''.join(right)} make the other two angles equal, "
                f"so each is (180-{given_angle:g})/2={answer_value}°. {relation_support} "
                "independent candidate records map x to one of those equal base angles."
            ),
            critique="",
            revision_request="",
            confidence=1.0,
            unresolved_issues=[],
            evidence_summary=[],
            source_artifact_ids=[str(rejected_artifact.get("artifact_id", ""))],
            status="ok",
            raw_response="",
            tool_records=[],
            llm={
                "model": "deterministic_geometry_certificate",
                "mock_used": False,
                "token_in": 0,
                "token_out": 0,
                "cost_usd": 0.0,
                "metadata": {},
            },
        )
        self._emit_meta_event(
            state,
            actor="geometry_certificate",
            event_type="verify",
            node_name="self_evolved_geometry_certificate",
            payload={
                "reason": "executable_triangle_angle_certificate",
                "given_vertex": vertex,
                "given_angle": given_angle,
                "coordinate_orientation_angle": orientation_angle,
                "equal_sides": ["".join(left), "".join(right)],
                "relation_support": relation_support,
                "answer": answer_value,
            },
        )
        return artifact

    @staticmethod
    def _needs_constraint_preserving_revision(artifact: ArtifactRecord) -> bool:
        if artifact.get("unresolved_issues"):
            return True
        answer = re.sub(r"\s+", " ", str(artifact.get("answer", ""))).lower()
        gap_patterns = (
            r"\b(?:could|can) not be (?:retrieved|provided|found|determined)\b",
            r"\b(?:was|were|is|are) not (?:provided|included|available|retrieved)\b",
            r"\bfailed to (?:return|provide|retrieve|supply)\b",
            r"\b(?:data|answer|result|record)s? (?:is|are|was|were) incomplete\b",
            r"\bunable to provide\b",
        )
        return any(re.search(pattern, answer) for pattern in gap_patterns)

    @classmethod
    def _recover_observed_list_fields(cls, answer: str, *, state: dict[str, Any]) -> str:
        """Recover non-exhaustive list values from completed structured records.

        Example: a categories endpoint fails, while successful product records each
        contain ``category``. The recovered list is explicitly observed/non-exhaustive
        and retains the original endpoint limitation.
        """

        try:
            payload = json.loads(answer)
        except (TypeError, ValueError, json.JSONDecodeError):
            return ""
        if not isinstance(payload, dict):
            return ""

        observed: dict[str, list[Any]] = {}
        for record in state.get("tool_records_log", []):
            if str(record.get("status", "")).lower() not in {"completed", "ok", "success"}:
                continue
            for field, values in cls._structured_scalar_fields(record.get("output")).items():
                bucket = observed.setdefault(field.lower(), [])
                for value in values:
                    normalized = str(value).casefold()
                    if all(str(existing).casefold() != normalized for existing in bucket):
                        bucket.append(value)

        changed = False
        for output_key, output_value in list(payload.items()):
            if not isinstance(output_value, str) or not re.search(
                r"(?i)\b(?:failed|not (?:provide|return|include|retrieve)|unavailable|unable)\b",
                output_value,
            ):
                continue
            for token in reversed(re.findall(r"[a-z]+", output_key.lower())):
                singular = cls._singular_field_name(token)
                values = observed.get(singular, [])
                if not values:
                    continue
                payload[output_key] = {
                    "observed_values": values[:20],
                    "scope": "non-exhaustive values observed in successful records",
                    "endpoint_limitation": output_value,
                }
                changed = True
                break
        if not changed:
            return ""
        return json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True)

    @staticmethod
    def _singular_field_name(value: str) -> str:
        if value.endswith("ies") and len(value) > 3:
            return f"{value[:-3]}y"
        if value.endswith("ses") and len(value) > 3:
            return value[:-2]
        if value.endswith("s") and len(value) > 1:
            return value[:-1]
        return value

    def _final_synthesis_interaction(
        self,
        *,
        state: dict[str, Any],
        prompt_messages: list[dict[str, str]],
        llm: Any,
        artifact: ArtifactRecord,
        agent_id: str,
        phase: str,
    ) -> dict[str, Any]:
        return {
            "dispatch_id": int(artifact.get("dispatch_id", state.get("dispatch_id", 0))),
            "agent_id": agent_id,
            "agent_role": agent_id,
            "stage_role": "aggregator",
            "agent_type": "general",
            "phase": phase,
            "round_index": int(state.get("round_index", 0)),
            "discussion_index": int(state.get("discussion_index", 0)),
            "tool_use_retry": False,
            "prompt_messages": prompt_messages,
            "visible_messages": [],
            "prior_artifact": None,
            "assistant_message": {"role": "assistant", "content": str(llm.text or "")},
            "structured_artifact": self._stage._serialize_for_json(artifact),
            "tool_calls": [],
            "llm": {
                "model": llm.model,
                "mock_used": bool(llm.mock_used),
                "token_in": int(llm.token_in),
                "token_out": int(llm.token_out),
                "cost_usd": float(llm.cost_usd),
                "metadata": self._stage._serialize_for_json(dict(llm.metadata)),
            },
        }

    def _needs_final_synthesis(
        self,
        final_answer: str,
        *,
        vote_result: dict[str, Any],
        state: dict[str, Any],
    ) -> bool:
        text = str(final_answer or "").strip()
        if not text:
            return True

        # The finalize fallback emits a fixed "unsupported answer" sentinel when no
        # admissible candidate exists. It reads as a full sentence, so
        # has_substantive_answer() treats it as a real answer and the checks below
        # would skip the read-net — even on retrieval runs that already surfaced the
        # gold document. Treat the sentinel as "no answer" and always re-synthesize.
        if text == UNSUPPORTED_FINAL_ANSWER.strip():
            return True

        # Retrieval runs whose agents never opened documents answered from truncated
        # top-3 snippets and are unreliable however confident they look — re-derive from
        # the full text of the top surfaced documents (the read-net). Gated on the run
        # exposing get_document and the agents having read < 2 documents, so runs that
        # did read, and all non-retrieval runs, are untouched.
        run_tools = state.get("tools") or []
        if any(
            isinstance(tool, dict) and str(tool.get("name", "")) == "get_document"
            for tool in run_tools
        ):
            reads = sum(
                1
                for record in state.get("tool_records_log", [])
                if str(record.get("tool_name", "")) == "get_document"
            )
            if reads < 2:
                return True

        lowered = text.lower()
        planning_markers = (
            "let me search",
            "let me investigate",
            "now let me search",
            "i found a promising lead",
            "looks very promising",
            "need to search",
            "search for the specific evidence",
        )
        if (
            not has_substantive_answer(text)
            or self._stage._answer_mode(text) in {"plan", "blocked", "empty"}
            or any(marker in lowered for marker in planning_markers)
        ):
            return True

        if not state.get("tools") and self._candidate_answer_disagreement(state):
            return True

        # A substantive answer can still be a weak pick worth re-synthesizing
        # from evidence: an unbroken vote tie, low winner confidence, or open
        # issues on the selected artifact. Synthesis still no-ops when no tool
        # evidence exists, so confident, agreed answers are never disturbed.
        tally = vote_result.get("tally", {}) or {}
        counts = sorted((int(value) for value in tally.values()), reverse=True)
        if len(counts) >= 2 and counts[0] == counts[1]:
            return True
        selected_id = str(vote_result.get("selected_artifact_id", "") or "")
        if selected_id:
            selected = artifacts_by_id(list(state.get("artifacts", []))).get(selected_id)
            if selected is not None:
                if float(selected.get("confidence", 0.5)) < 0.5:
                    return True
                if selected.get("unresolved_issues"):
                    return True
        return False

    @staticmethod
    def _candidate_answer_disagreement(state: dict[str, Any]) -> bool:
        latest = latest_artifact_by_agent(list(state.get("artifacts", [])))
        signatures: set[str] = set()
        for agent_id, artifact in latest.items():
            if agent_id in {
                "final_synthesizer",
                "constraint_reconciler",
                "proof_falsifier",
                "proof_appeal_reviewer",
                "proof_oracle_corrector",
                "proof_gate_selector",
                "output_contract_enforcer",
            }:
                continue
            if str(artifact.get("stage_role", "")) in {"planner", "critic"}:
                continue
            answer = str(artifact.get("answer", ""))
            if not has_substantive_answer(answer):
                continue
            signature = TurnExecutor._decision_answer_signature(
                SelfEvolvedEngine._canonicalize_explicit_math_answer(answer)
            )
            if signature:
                signatures.add(signature)
            if len(signatures) >= 2:
                return True
        return False

    def _collect_reasoning_candidate_evidence(self, state: dict[str, Any]) -> list[str]:
        latest = latest_artifact_by_agent(list(state.get("artifacts", [])))
        evidence: list[str] = []
        for agent_id, artifact in sorted(latest.items()):
            if str(artifact.get("stage_role", "")) in {"planner", "critic"}:
                continue
            answer = re.sub(r"\s+", " ", str(artifact.get("answer", ""))).strip()
            if not answer:
                continue
            summary = re.sub(r"\s+", " ", str(artifact.get("summary", ""))).strip()
            critique = re.sub(r"\s+", " ", str(artifact.get("critique", ""))).strip()
            evidence.append(
                f"- candidate={agent_id}; answer={answer[:3600]}; "
                f"summary={summary[:1400]}; critique={critique[:1400]}"
            )
            if len(evidence) >= 8:
                break
        return evidence

    def _read_documents_for_synthesis(
        self, state: dict[str, Any], *, max_docs: int = 24, max_chars: int = 1500
    ) -> list[str]:
        """Deterministically open the top surfaced search documents for synthesis.

        Weak models often search broadly but never call ``get_document``, then answer
        from the top-3 truncated snippets per search — so a correct-but-low-ranked
        document is never read (e.g. the gold doc ranked 23rd by BM25). This opens the
        highest-scoring surfaced docids (deduped across all searches) via the run's own
        ``get_document`` handler and returns their full text, so synthesis can recover the
        answer the agents missed. Inert for runs without a ``get_document`` tool.
        """

        get_document = None
        for tool in state.get("tools") or []:
            if (
                isinstance(tool, dict)
                and str(tool.get("name", "")) == "get_document"
                and callable(tool.get("handler"))
            ):
                get_document = tool["handler"]
                break
        if get_document is None:
            return []

        best_score: dict[str, float] = {}
        for record in state.get("tool_records_log", []):
            tool_name = str(record.get("tool_name", ""))
            if not tool_name.endswith("search"):
                continue
            output = record.get("output")
            if not isinstance(output, list):
                continue
            for rank, row in enumerate(output, start=1):
                if not isinstance(row, dict):
                    continue
                docid = str(row.get("docid", "")).strip()
                if not docid:
                    continue
                try:
                    if tool_name == "constraint_search":
                        # Its output order deliberately preserves one candidate per
                        # clue. Prefer that coverage-aware order over raw lexical score.
                        score = 100.0 - rank
                    else:
                        score = float(row.get("score", 0.0))
                except Exception:
                    score = 0.0
                best_score[docid] = max(best_score.get(docid, 0.0), score)
        if not best_score:
            return []

        ranked = sorted(best_score, key=lambda docid: best_score[docid], reverse=True)
        documents: list[str] = []
        for docid in ranked[:max_docs]:
            try:
                doc = get_document({"docid": docid})
            except Exception:
                continue
            if not isinstance(doc, dict):
                continue
            text = doc.get("text") or doc.get("snippet") or ""
            text = re.sub(r"\s+", " ", str(text)).strip()[:max_chars]
            if text:
                documents.append(f"- docid={docid}: {text}")
        return documents

    def _collect_tool_evidence_for_synthesis(self, state: dict[str, Any]) -> list[str]:
        evidence: list[str] = []
        seen: set[str] = set()
        for record in state.get("tool_records_log", []):
            tool_name = str(record.get("tool_name", ""))
            if not tool_name.endswith("search"):
                continue
            arguments = record.get("arguments")
            query = ""
            if isinstance(arguments, dict):
                query = str(arguments.get("query") or arguments.get("queries") or "")
            output = record.get("output")
            rows = output if isinstance(output, list) else []
            for row in rows[:3]:
                if not isinstance(row, dict):
                    continue
                docid = str(row.get("docid", "")).strip()
                snippet = str(row.get("snippet", "")).strip()
                if not snippet:
                    continue
                key = f"{docid}:{snippet[:120]}"
                if key in seen:
                    continue
                seen.add(key)
                evidence.append(
                    f"- tool={tool_name}; query={query!r}; docid={docid}; snippet={snippet[:900]}"
                )
                if len(evidence) >= 12:
                    return evidence

        # Non-search tools may return structured results, partial data, API
        # descriptions, cache misses, or transport failures. All are useful to a weak
        # finalizer, provided failures are represented as failures rather than promoted
        # to factual answers. Keep one bounded record per distinct attempt.
        for record in state.get("tool_records_log", []):
            tool_name = str(record.get("tool_name", ""))
            if not tool_name or tool_name.endswith("search") or tool_name == "inter_agent_send":
                continue
            arguments = record.get("arguments", {})
            status = str(record.get("status", "unknown"))
            output = record.get("output", record.get("error", ""))
            try:
                args_text = json.dumps(arguments, ensure_ascii=False, default=str, sort_keys=True)
            except (TypeError, ValueError):
                args_text = str(arguments)
            try:
                output_text = json.dumps(output, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                output_text = str(output)
            output_text = re.sub(r"\s+", " ", output_text).strip()[:1000]
            projection = self._structured_field_projection(output)
            key = f"{tool_name}:{args_text}:{status}:{output_text}"
            if key in seen:
                continue
            seen.add(key)
            evidence.append(
                f"- tool={tool_name}; status={status}; args={args_text[:500]}; "
                f"observed_fields={projection or '(none)'}; output={output_text or '(empty)'}"
            )
            if len(evidence) >= 12:
                break
        return evidence

    @staticmethod
    def _structured_field_projection(
        value: Any, *, max_fields: int = 12, max_values_per_field: int = 10
    ) -> str:
        """Preserve distinct scalar fields across a large structured tool result.

        Prefix truncation can retain the first record while losing later entities or
        classification values. This bounded projection is descriptive only: it does not
        infer missing fields or promote failed tool output to evidence.
        """

        fields = SelfEvolvedEngine._structured_scalar_fields(
            value, max_values_per_field=max_values_per_field
        )

        bounded = dict(list(fields.items())[:max_fields])
        if not bounded:
            return ""
        return json.dumps(bounded, ensure_ascii=False, default=str, sort_keys=True)[:900]

    @staticmethod
    def _structured_scalar_fields(
        value: Any, *, max_values_per_field: int = 20
    ) -> dict[str, list[Any]]:
        fields: dict[str, list[Any]] = {}

        def visit(item: Any, field_name: str = "") -> None:
            if isinstance(item, dict):
                for key, child in item.items():
                    visit(child, str(key))
                return
            if isinstance(item, list):
                for child in item:
                    visit(child, field_name)
                return
            if not field_name or item is None:
                return
            bucket = fields.setdefault(field_name, [])
            if len(bucket) >= max_values_per_field:
                return
            normalized = str(item)
            if all(str(existing) != normalized for existing in bucket):
                bucket.append(item)

        visit(value)
        return fields

    # -- state / metadata ------------------------------------------------------

    def _assign_personas(
        self,
        *,
        task: Any,
        spec: ExperimentSpec,
        layout: Any,
    ) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
        domain_personas: dict[str, dict[str, str]] = {}
        payload: dict[str, Any] = {
            "enabled": bool(spec.enable_dynamic_roles),
            "benchmark_name": str(spec.benchmark_name or ""),
            "used_fallback": False,
            "fallback_reason": "",
            "prompt_messages": [],
            "response": "",
            "llm": {},
            "assignments": {},
        }
        if not spec.benchmark_name:
            payload["fallback_reason"] = "missing_benchmark_name"
            return domain_personas, payload
        if not spec.enable_dynamic_roles:
            payload["fallback_reason"] = "disabled"
            return domain_personas, payload

        from ..role_assigner import (
            RoleAssignmentResult,
            assign_domain_roles,
            assign_domain_roles_deterministic,
        )

        try:
            assignment_result = assign_domain_roles(
                benchmark_name=spec.benchmark_name,
                layout=layout,
                task_prompt=getattr(task, "prompt", ""),
                llm_client=self.llm_client,
            )
        except Exception:
            assignments = assign_domain_roles_deterministic(
                benchmark_name=spec.benchmark_name, layout=layout
            )
            assignment_result = RoleAssignmentResult(
                assignments=assignments,
                prompt_messages=[],
                response_text="",
                llm={},
                used_fallback=True,
                fallback_reason="engine_role_assignment_error",
            )
        domain_personas = {
            agent_id: {"role_name": info.role_name, "persona": info.persona}
            for agent_id, info in assignment_result.assignments.items()
        }
        payload.update(
            {
                "used_fallback": bool(assignment_result.used_fallback),
                "fallback_reason": str(assignment_result.fallback_reason),
                "prompt_messages": self._stage._serialize_for_json(
                    list(assignment_result.prompt_messages)
                ),
                "response": str(assignment_result.response_text),
                "llm": self._stage._serialize_for_json(dict(assignment_result.llm)),
                "assignments": {
                    agent_id: {"role_name": info.role_name, "persona": info.persona}
                    for agent_id, info in assignment_result.assignments.items()
                },
            }
        )
        return domain_personas, payload

    def _initial_state(
        self,
        *,
        task: Any,
        run_index: int,
        seed: int,
        spec: ExperimentSpec,
        layout: Any,
        agent_type_by_agent: dict[str, str],
        tools: list[dict[str, Any]],
        max_tool_iterations: int,
        descriptor: DescriptorHook,
        domain_personas: dict[str, dict[str, str]],
        role_assignment_payload: dict[str, Any],
        workflow_definition: dict[str, Any],
    ) -> dict[str, Any]:
        # Full fidelity by default; an explicit peer_artifact_max_chars (or the
        # optional self_evolved default_packet_max_chars knob) sets a generous
        # structural-compaction budget rather than a blunt truncation floor.
        peer_artifact_max_chars = int(spec.peer_artifact_max_chars)
        if peer_artifact_max_chars <= 0 and int(self.se_config.default_packet_max_chars) > 0:
            peer_artifact_max_chars = int(self.se_config.default_packet_max_chars)
        return {
            "task_id": str(getattr(task, "task_id", "task")),
            "task_prompt": getattr(task, "prompt", ""),
            "benchmark_name": str(spec.benchmark_name or ""),
            "reference_answer": str(getattr(task, "reference_answer", "")),
            "task_metadata": dict(getattr(task, "metadata", {}) or {}),
            "run_index": int(run_index),
            "seed": int(seed),
            "topology": TOPOLOGY_SELF_EVOLVED,
            "layout": layout,
            "workflow_definition": workflow_definition,
            "rounds": int(self.se_config.max_turns),
            "discussion_rounds": int(spec.discussion_rounds),
            "minimum_discussion_rounds": int(spec.minimum_discussion_rounds),
            "termination_consensus_mode": str(spec.termination_consensus_mode),
            "final_vote_mode": str(spec.final_vote_mode),
            "peer_artifact_max_chars": peer_artifact_max_chars,
            "round_index": 0,
            "discussion_index": 0,
            "phase": "init",
            "dispatch_id": -1,
            "active_agents": [],
            "next_step": "",
            "done": False,
            "llm_client": self.llm_client,
            "agent_type_by_agent": agent_type_by_agent,
            "tools": list(tools),
            "max_tool_iterations": max(1, int(max_tool_iterations)),
            "descriptor": descriptor,
            "messages": [],
            "message_views": [],
            "artifacts": [],
            "trace_payloads": [],
            "phase_history": [],
            "descriptor_records": [],
            "interaction_logs": [],
            "tool_records_log": [],
            "termination_history": [],
            "transcript_summary": {},
            "transcript_compaction_history": [],
            "context_state_versions": [],
            "short_term_playbook_entries": [],
            "communication_budget_per_agent": int(spec.communication_budget_per_agent),
            "message_budget": {
                agent_id: int(spec.communication_budget_per_agent) for agent_id in layout.agent_ids
            },
            "sent_counts": {agent_id: 0 for agent_id in layout.agent_ids},
            "budget_sent_counts": {agent_id: 0 for agent_id in layout.agent_ids},
            "message_seq": 0,
            "final_answer": "",
            "final_reason": "",
            "vote_tally": {},
            "final_vote_source": "",
            "selected_artifact_id": "",
            "selected_agent_id": "",
            "selected_source_artifact_ids": [],
            "termination_decision": {},
            "descriptor_summary": {},
            "domain_personas": domain_personas,
            "role_assignment": role_assignment_payload,
        }

    def _emit_meta_event(
        self,
        state: dict[str, Any],
        *,
        actor: str,
        event_type: str,
        node_name: str,
        payload: dict[str, Any],
        token_in: int = 0,
        token_out: int = 0,
        latency_ms: float = 1.0,
        cost_usd: float = 0.0,
    ) -> None:
        dispatch_id = int(state.get("dispatch_id", -1)) + 1
        state["dispatch_id"] = dispatch_id
        state["phase"] = node_name
        event = self._stage._draft_event(
            dispatch_id=dispatch_id,
            node_order=0,
            agent_order=-1,
            event_order=0,
            actor=actor,
            event_type=event_type,
            payload={"node": node_name, **self._stage._serialize_for_json(payload)},
            token_in=token_in,
            token_out=token_out,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            state_id=f"run_{state['run_index']}_{node_name}",
        )
        apply_updates(state, {"trace_payloads": [event]})

    @staticmethod
    def _workflow_definition() -> WorkflowDocumentation:
        return WorkflowDocumentation(
            topology=TOPOLOGY_SELF_EVOLVED,
            nodes={
                "topology_plan": "Topology Planner proposes the initial TopologySpec",
                "spawn_agents": "Orchestrator instantiates agents, personas, context policies",
                "execute_turn": "TurnExecutor interprets the current TopologySpec",
                "audit_turn": "Trace Auditor inspects the turn for failure modes",
                "meta_termination": "Ordered code-level stop/repair decision",
                "short_term_playbook_turn": "Playbook Maintainer records turn-level process memory",
                "apply_mutation": "Orchestrator applies the planner mutation",
                "finalize": "Vote over one preserved candidate from every turn",
            },
            edges=[
                "START -> topology_plan",
                "topology_plan -> spawn_agents",
                "spawn_agents -> execute_turn",
                "execute_turn -> audit_turn",
                "audit_turn -> meta_termination",
                "meta_termination -> short_term_playbook_turn",
                "apply_mutation -> execute_turn",
                "finalize -> END",
            ],
            conditional_edges=[
                "meta_termination -> apply_mutation (repair available)",
                "meta_termination -> finalize (stop)",
            ],
            dispatch_logic=(
                "Group-tree interpretation: leader plan -> member contributions "
                "(expansions recurse) -> relay -> aggregate."
            ),
            aggregation_logic=(
                "Root output preferred when substantive; otherwise vote over member artifacts."
            ),
            stopping_criteria=[
                "invalid_or_failed_branch",
                "consensus_reached",
                "no_meaningful_change",
                "max_rounds_reached (no repair available or mutation budget spent)",
            ],
            state_fields=[
                "topology_spec_versions",
                "context_state_versions",
                "short_term_playbook_entries",
                "audit_reports",
                "mutation",
            ],
            logging_outputs=["trace_events", "relay_messages", "interaction_logs"],
        )

    def _build_run_metadata(
        self,
        *,
        state: dict[str, Any],
        spec: ExperimentSpec,
        layout: Any,
        run_index: int,
        seed: int,
        workflow_definition: dict[str, Any],
        role_assignment_payload: dict[str, Any],
        spec_versions: list[dict[str, Any]],
        audit_reports: list[dict[str, Any]],
        mutation_payload: dict[str, Any] | None,
        mutation_payloads: list[dict[str, Any]],
        plan_payload: dict[str, Any],
        playbook_candidate: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        artifacts = list(state.get("artifacts", []))
        latest_outputs = {
            agent_id: str(artifact.get("answer", ""))
            for agent_id, artifact in latest_artifact_by_agent(artifacts).items()
        }
        tool_call_counts = self._stage._count_tool_calls(state)
        metadata = {
            "task_id": state.get("task_id"),
            "run_index": run_index,
            "seed": seed,
            "topology": TOPOLOGY_SELF_EVOLVED,
            "rounds_configured": int(self.se_config.max_turns),
            "discussion_rounds": spec.discussion_rounds,
            "termination_consensus_mode": str(spec.termination_consensus_mode),
            "final_vote_mode": str(spec.final_vote_mode),
            "peer_artifact_max_chars": int(state.get("peer_artifact_max_chars", 0)),
            "turns_executed": max(1, int(state.get("round_index", 0)) + 1),
            "messages_sent_total": sum(state.get("sent_counts", {}).values()),
            "messages_sent_by_agent": dict(state.get("sent_counts", {})),
            "budget_messages_sent_total": sum(state.get("budget_sent_counts", {}).values()),
            "budget_messages_sent_by_agent": dict(state.get("budget_sent_counts", {})),
            "tool_call_counts": tool_call_counts,
            "tool_calls_total": int(sum(tool_call_counts.values())),
            "remaining_message_budget": dict(state.get("message_budget", {})),
            "tool_definitions": self._stage._serialize_tools(list(state.get("tools", []))),
            "retrieved_docids": self._stage._collect_retrieved_docids(state),
            "agent_outputs": latest_outputs,
            "vote_tally": dict(state.get("vote_tally", {})),
            "final_vote_source": str(state.get("final_vote_source", "")),
            "selected_artifact_id": str(state.get("selected_artifact_id", "")),
            "selected_agent_id": str(state.get("selected_agent_id", "")),
            "selected_source_artifact_ids": list(state.get("selected_source_artifact_ids", [])),
            "phase_history": list(state.get("phase_history", [])),
            "relay_messages": list(state.get("messages", [])),
            "message_views": list(state.get("message_views", [])),
            "interaction_logs": list(state.get("interaction_logs", [])),
            "descriptor_records": list(state.get("descriptor_records", [])),
            "descriptor_summary": dict(state.get("descriptor_summary", {})),
            "termination_history": list(state.get("termination_history", [])),
            "transcript_summary": dict(state.get("transcript_summary", {})),
            "transcript_compaction_history": list(state.get("transcript_compaction_history", [])),
            "domain_personas": dict(state.get("domain_personas", {})),
            "role_assignment": dict(state.get("role_assignment", role_assignment_payload)),
            "topology_layout": layout.to_payload(),
            "workflow_definition": workflow_definition,
            "artifact_records": artifacts,
            "final_reason": str(state.get("final_reason", "")),
            "self_evolved": {
                "harness_backend": str(self.se_config.harness_backend),
                "initial_planner_mode": str(self.se_config.initial_planner_mode),
                "topology_spec_versions": list(spec_versions),
                "context_state_versions": list(state.get("context_state_versions", [])),
                "mutation": mutation_payload,
                "mutations": list(mutation_payloads),
                "mutation_proposals": list(state.get("self_evolved_mutation_proposals", [])),
                "audit_reports": list(audit_reports),
                "planner": dict(plan_payload),
                "audit_mode": str(self.se_config.audit_mode),
                "playbook_memory": {
                    "short_term": "updated in memory after each audited turn",
                    "long_term": "persistent file updated post-hoc from eval-joined candidates",
                },
                "short_term_playbook_entries": list(state.get("short_term_playbook_entries", [])),
                "final_synthesis": list(state.get("self_evolved_final_synthesis", [])),
                "contract_repairs": list(state.get("self_evolved_contract_repairs", [])),
                "temporal_candidates": list(state.get("self_evolved_temporal_candidates", [])),
                "playbook_update_candidate": playbook_candidate,
                "write_commit": state.get("self_evolved_commit"),
            },
        }
        commit = state.get("self_evolved_commit")
        if isinstance(commit, dict):
            # Benchmarks that score external state read the committed writes from here.
            metadata["committed_actions"] = list(commit.get("committed", []))
        return metadata
