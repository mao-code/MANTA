"""Turn executor: interprets a TopologySpec for one collaboration turn.

Instead of compiling a static LangGraph per topology, the executor walks the
group tree directly. Every agent stage still runs through
``LangGraphMASEngine._execute_agent_stage`` so prompts, artifact coercion,
tool retry, and trace events are identical to the fixed-topology systems.

Pattern semantics (uniform across nesting levels):
- singleton: the member runs its contribution stage.
- star:      leader plans -> task packets -> members contribute -> reports
             relayed to leader -> leader aggregates.
- chain:     members contribute sequentially, each seeing the predecessor's
             bounded report packet.
- debate:    members contribute, exchange peer summaries along group edges,
             then run one critic revision round; output by deterministic vote.
- voting:    members contribute independently; output by deterministic vote.

An agent with an attached subgroup contributes by expansion: it plans, the
subgroup executes its own pattern, reports relay back, and the agent
aggregates — its aggregate artifact is its report to the enclosing group.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..artifacts import ArtifactRecord, answer_signature, packet_payload_from_artifact
from ..langgraph_engine import LangGraphMASEngine
from .context import SharedContextController
from .spec import GroupSpec, TopologySpec

# State keys accumulated across stages (mirrors the LangGraph reducers).
APPEND_STATE_KEYS = frozenset(
    {
        "messages",
        "message_views",
        "artifacts",
        "trace_payloads",
        "phase_history",
        "descriptor_records",
        "interaction_logs",
        "tool_records_log",
        "termination_history",
        "transcript_compaction_history",
        "context_state_versions",
        "short_term_playbook_entries",
    }
)

TASK_PACKET_KINDS = {"task_package", "orchestrator_feedback"}

# Self-evolved retrieval directive. When a run exposes a get_document tool
# (retrieval benchmarks such as browsecomp), search results are only pointers;
# the dominant failure mode is agents that search repeatedly but never open a
# document and then give up with an "insufficient evidence" answer. Every
# self-evolved agent stage is told to read before concluding. Inert for runs
# without get_document (plancraft / workbench / stabletoolbench / mock mode).
_READ_FIRST_DIRECTIVE = (
    " This task provides a get_document tool. Search returns only short snippets, which "
    "are NOT sufficient to answer — you MUST call get_document to read the full text of "
    "the most relevant documents. As soon as a search returns a relevant docid, "
    "immediately call get_document on the top 1-3 docids; do NOT run more than two rounds "
    "of search before opening a document. Answer only from documents you have opened with "
    "get_document — never answer from snippets alone, and never return an 'insufficient "
    "evidence' or blocked answer without having opened at least one document. For a task with "
    "several independent clues, call constraint_search first with 3-6 narrow clue clauses; it "
    "tests small bounded numeric intervals one value at a time and ranks cross-clause support. "
    "Copy rare modifiers from the task into those clauses verbatim; dropping an unusual word "
    "can erase the decisive lexical hit. "
    "Otherwise treat search as literal lexical retrieval: it does not understand wildcards. "
    "Start with a high-information conjunction of several rare constraints rather than a "
    "generic category query. Search your assigned facet; do not "
    "repeat a peer's query or adopt its candidate until you have tested a distinct constraint. "
    "When a plausible entity appears, pivot to entity-name queries and verify all constraints."
)


# Self-evolved tool-chaining directive for domain-tool tasks (stabletoolbench,
# workbench) that expose API tools but no get_document. Failure mode observed: an
# agent runs one entity "search", the backend returns a cache miss for a non-canonical
# argument (e.g. name="Messi"), the search circuit breaker trips, and the agent answers
# from incomplete data. The fix is behavioral: use canonical arguments and chain the
# follow-up tools to gather the specific facts the question asks for.
_TOOL_CHAIN_DIRECTIVE = (
    " Use full, official, canonical names and identifiers in tool arguments (e.g. "
    "'Lionel Messi', not 'Messi'); a non-canonical argument often returns nothing. One "
    "lookup rarely answers the question: after locating the relevant entity, chain the "
    "follow-up tools (details / profile / related lookups) to gather the specific facts "
    "the question asks for before answering. If a tool returns no result, retry once with "
    "a more canonical argument instead of abandoning the tool."
)

_COMMIT_ONCE_DIRECTIVE = (
    " WRITE PROTOCOL: State-changing tools change nothing during the run. Each call is "
    "checked and recorded as a proposal. After the run, the proposals of the agent "
    "closest to the selected final answer are executed, exactly once. Gather the facts "
    "with read tools first. If the task's condition requires a change, call the "
    "state-changing tool once with the exact final arguments. Do not stop at a plan or "
    "claim success without recording the proposal. If the condition does not hold, call "
    "no state-changing tool and state the evidence. If an earlier agent's proposal is "
    "correct, do not repeat it. If it is wrong, call the tool again with the corrected "
    "arguments, because your proposals replace the earlier ones. "
)


_INDEPENDENT_ROUTES = (
    "derive the answer from first principles and expose the critical intermediate checks",
    "use a materially different representation or decomposition from the most obvious route",
    "try to falsify the leading conclusion with counterexamples, boundary cases, or unmet constraints",
    "recompute the decisive steps backward from the proposed answer and check consistency",
)


def _run_has_get_document(state: dict[str, Any]) -> bool:
    return any(
        isinstance(tool, dict) and str(tool.get("name", "")) == "get_document"
        for tool in (state.get("tools") or [])
    )


def _run_has_tools(state: dict[str, Any]) -> bool:
    return bool(state.get("tools"))


_MUTATION_VERBS = frozenset(
    {
        "add",
        "book",
        "cancel",
        "create",
        "delete",
        "modify",
        "move",
        "remove",
        "rename",
        "schedule",
        "send",
        "set",
        "submit",
        "update",
        "write",
    }
)


def state_changing_tool_names(tools: list[dict[str, Any]]) -> set[str]:
    """Conservatively identify externally mutating tools from their names."""

    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = str(tool.get("name", "")).strip()
        declared = tool.get("side_effect")
        if isinstance(declared, bool):
            # The benchmark adapter knows which tools change its environment.
            if declared:
                names.add(name)
            continue
        leaf = name.rsplit(".", 1)[-1].casefold()
        tokens = set(re.findall(r"[a-z]+", leaf))
        if tokens & _MUTATION_VERBS:
            names.add(name)
    return names


def apply_updates(state: dict[str, Any], updates: dict[str, Any]) -> None:
    """Merge stage-node updates into the plain-dict state."""

    for key, value in updates.items():
        if key in APPEND_STATE_KEYS and isinstance(value, list):
            state.setdefault(key, []).extend(value)
        else:
            state[key] = value


@dataclass
class TurnResult:
    output_artifact: ArtifactRecord | None
    member_artifacts: list[ArtifactRecord] = field(default_factory=list)
    expected_member_count: int = 0


class TurnExecutor:
    def __init__(self, stage_engine: LangGraphMASEngine, context: SharedContextController) -> None:
        self._engine = stage_engine
        self._context = context
        # Per-run ledger of executed tool-call signatures. Guarantees an identical
        # (tool, args) call is recorded once even if the planner spawns parallel
        # workers that each invoke it — without this, side-effecting tools (e.g.
        # calendar.create_event) get replayed and corrupt the final state. Scoped
        # to one run (the executor is created per run) and shared across turns.
        self._executed_signatures: set[str] = set()

    def run_turn(self, state: dict[str, Any], spec: TopologySpec, *, turn_index: int) -> TurnResult:
        state["round_index"] = int(turn_index)
        state["discussion_index"] = 0
        root = spec.group(spec.root_group_id)
        return self._execute_root_group(state, spec, root)

    # -- group execution ----------------------------------------------------

    def _execute_root_group(
        self,
        state: dict[str, Any],
        spec: TopologySpec,
        group: GroupSpec,
    ) -> TurnResult:
        if group.pattern == "singleton":
            artifact = self._contribute(state, spec, group.member_ids[0], group)
            members = [artifact] if artifact else []
            return TurnResult(
                output_artifact=artifact, member_artifacts=members, expected_member_count=1
            )

        if group.pattern == "star":
            leader = str(group.leader_id)
            workers = [member for member in group.member_ids if member != leader]
            plan_artifact = self._run_leader_stage(
                state,
                spec,
                group,
                agent_id=leader,
                stage_role="planner",
                node_suffix="plan",
                directive=(
                    "Produce a concise plan and a bounded task package for each group member."
                ),
            )
            self._emit_task_packets(
                state, sender=leader, recipients=workers, artifact=plan_artifact
            )
            member_artifacts = self._run_member_contributions(state, spec, group, workers)
            self._relay_reports(
                state,
                group=group,
                artifacts=member_artifacts,
                recipients=[leader],
                kind="specialist_report",
            )
            merge_artifact = self._run_leader_stage(
                state,
                spec,
                group,
                agent_id=leader,
                stage_role="aggregator",
                node_suffix="merge",
                directive=(
                    "Merge the member reports into one best answer. "
                    "Preserve unresolved issues explicitly."
                ),
                visible_kinds={"specialist_report"},
                # Final synthesis always sees the global evidence digest.
                evidence_scope="global",
            )
            return TurnResult(
                output_artifact=merge_artifact,
                member_artifacts=member_artifacts,
                expected_member_count=len(workers),
            )

        if group.pattern == "chain":
            member_artifacts = self._run_member_contributions(
                state, spec, group, list(group.member_ids), sequential_relay=True
            )
            output = member_artifacts[-1] if member_artifacts else None
            return TurnResult(
                output_artifact=output,
                member_artifacts=member_artifacts,
                expected_member_count=len(group.member_ids),
            )

        # debate / voting roots
        member_artifacts = self._run_member_contributions(
            state, spec, group, list(group.member_ids)
        )
        if group.pattern == "debate" and member_artifacts:
            member_artifacts = self._run_debate_revision(state, spec, group, member_artifacts)
        output = self._select_group_output(member_artifacts)
        return TurnResult(
            output_artifact=output,
            member_artifacts=member_artifacts,
            expected_member_count=len(group.member_ids),
        )

    def _execute_expansion(
        self,
        state: dict[str, Any],
        spec: TopologySpec,
        *,
        parent_agent_id: str,
        subgroup: GroupSpec,
    ) -> ArtifactRecord | None:
        """Run an attached subgroup as the parent agent's contribution."""

        plan_artifact = self._run_leader_stage(
            state,
            spec,
            subgroup,
            agent_id=parent_agent_id,
            stage_role="planner",
            node_suffix="plan",
            directive=(
                "Decompose your assigned subtask into bounded task packages "
                "for your subgroup members."
            ),
            visible_kinds=TASK_PACKET_KINDS | {"peer_summary"},
        )
        self._emit_task_packets(
            state,
            sender=parent_agent_id,
            recipients=list(subgroup.member_ids),
            artifact=plan_artifact,
        )
        member_artifacts = self._run_member_contributions(
            state,
            spec,
            subgroup,
            list(subgroup.member_ids),
            sequential_relay=subgroup.pattern == "chain",
        )
        if subgroup.pattern == "debate" and member_artifacts:
            member_artifacts = self._run_debate_revision(state, spec, subgroup, member_artifacts)
        self._relay_reports(
            state,
            group=subgroup,
            artifacts=member_artifacts,
            recipients=[parent_agent_id],
            kind="child_report",
        )
        parent_scope = (
            "global"
            if spec.agent(parent_agent_id).context.evidence_access == "global"
            else "branch"
        )
        return self._run_leader_stage(
            state,
            spec,
            subgroup,
            agent_id=parent_agent_id,
            stage_role="aggregator",
            node_suffix="merge",
            directive=(
                "Synthesize your subgroup reports into one supported answer for "
                "your branch; this artifact is your report to the enclosing group."
            ),
            visible_kinds={"child_report"},
            # Branch aggregators see at least their own branch's evidence.
            evidence_scope=parent_scope,
        )

    # -- member contributions -----------------------------------------------

    def _run_member_contributions(
        self,
        state: dict[str, Any],
        spec: TopologySpec,
        group: GroupSpec,
        members: list[str],
        *,
        sequential_relay: bool = False,
    ) -> list[ArtifactRecord]:
        if not members:
            return []
        self._dispatch(state, node_name=f"se_{group.group_id}_work_dispatch", active_agents=members)
        artifacts: list[ArtifactRecord] = []
        previous: ArtifactRecord | None = None
        for member in members:
            if sequential_relay and previous is not None:
                self._relay_reports(
                    state,
                    group=group,
                    artifacts=[previous],
                    recipients=[member],
                    kind="peer_summary",
                )
            artifact = self._contribute(state, spec, member, group)
            if artifact is not None:
                artifacts.append(artifact)
                previous = artifact
        return artifacts

    def _contribute(
        self,
        state: dict[str, Any],
        spec: TopologySpec,
        agent_id: str,
        group: GroupSpec,
    ) -> ArtifactRecord | None:
        subgroup = spec.subgroup_of(agent_id)
        if subgroup is not None:
            return self._execute_expansion(
                state,
                spec,
                parent_agent_id=agent_id,
                subgroup=subgroup,
            )

        node = spec.agent(agent_id)
        directive = (
            "Work only from the task and the visible packets. "
            "Gather or apply evidence, then state the best supported answer."
            if node.stage_role == "worker"
            else "Verify the visible claims against evidence and revise toward "
            "the best supported answer."
        )
        directive += self._independent_route_directive(group, agent_id)
        visible = self._context.visible_packets(
            state,
            agent_id=agent_id,
            kinds=TASK_PACKET_KINDS | {"peer_summary"},
            round_index=int(state.get("round_index", 0)),
        )
        digest = self._context.evidence_digest_packet(state, agent_id=agent_id)
        if digest is not None:
            visible = [digest, *visible]
        return self._run_stage(
            state,
            agent_id=agent_id,
            node_name=f"se_{group.group_id}_work",
            stage_role=node.stage_role,
            directive=directive,
            visible_messages=visible,
        )

    @staticmethod
    def _independent_route_directive(group: GroupSpec, agent_id: str) -> str:
        """Assign epistemically distinct work to otherwise independent members."""

        if group.pattern not in {"debate", "voting"} or len(group.member_ids) < 2:
            return ""
        try:
            member_index = list(group.member_ids).index(agent_id)
        except ValueError:
            return ""
        route = _INDEPENDENT_ROUTES[member_index % len(_INDEPENDENT_ROUTES)]
        return (
            f" Independent route {member_index + 1}/{len(group.member_ids)}: {route}. "
            "Do not treat likely peer agreement as evidence; leave a checkable derivation "
            "that another route could disprove."
        )

    def _run_debate_revision(
        self,
        state: dict[str, Any],
        spec: TopologySpec,
        group: GroupSpec,
        member_artifacts: list[ArtifactRecord],
    ) -> list[ArtifactRecord]:
        state["discussion_index"] = int(state.get("discussion_index", 0)) + 1
        members = list(group.member_ids)
        peers_by_sender = {
            str(artifact.get("agent_id", "")): artifact for artifact in member_artifacts
        }
        packet_specs: list[dict[str, Any]] = []
        for sender, artifact in peers_by_sender.items():
            recipients = [member for member in members if member != sender]
            payload = packet_payload_from_artifact(
                artifact, max_chars=self._packet_max_chars(state)
            )
            packet_specs.append(
                {
                    "sender": sender,
                    "recipients": recipients,
                    "kind": "peer_summary",
                    "round": int(state.get("round_index", 0)),
                    "discussion_index": int(state.get("discussion_index", 0)),
                    "artifact_id": artifact.get("artifact_id"),
                    "payload": payload,
                }
            )
        self._dispatch(
            state,
            node_name=f"se_{group.group_id}_debate_dispatch",
            active_agents=members,
            packet_specs=packet_specs,
        )

        revised: list[ArtifactRecord] = []
        for member in members:
            visible = self._context.visible_packets(
                state,
                agent_id=member,
                kinds={"peer_summary"},
                round_index=int(state.get("round_index", 0)),
                discussion_index=int(state.get("discussion_index", 0)),
            )
            digest = self._context.evidence_digest_packet(state, agent_id=member)
            if digest is not None:
                visible = [digest, *visible]
            artifact = self._run_stage(
                state,
                agent_id=member,
                node_name=f"se_{group.group_id}_debate",
                stage_role="critic",
                directive=(
                    "Challenge weak peer claims using the visible debate packets and "
                    "revise toward the best supported answer."
                ),
                visible_messages=visible,
            )
            if artifact is not None:
                revised.append(artifact)
        return revised

    # -- shared stage / packet helpers ---------------------------------------

    def _run_leader_stage(
        self,
        state: dict[str, Any],
        spec: TopologySpec,
        group: GroupSpec,
        *,
        agent_id: str,
        stage_role: str,
        node_suffix: str,
        directive: str,
        visible_kinds: set[str] | None = None,
        evidence_scope: str | None = None,
    ) -> ArtifactRecord | None:
        self._dispatch(
            state,
            node_name=f"se_{group.group_id}_{node_suffix}_dispatch",
            active_agents=[agent_id],
        )
        visible: list[Any] = []
        if visible_kinds:
            visible = self._context.visible_packets(
                state,
                agent_id=agent_id,
                kinds=visible_kinds,
                round_index=int(state.get("round_index", 0)),
            )
        if evidence_scope is not None:
            digest = self._context.evidence_digest_packet(
                state, agent_id=agent_id, scope=evidence_scope
            )
            if digest is not None:
                visible = [digest, *visible]
        return self._run_stage(
            state,
            agent_id=agent_id,
            node_name=f"se_{group.group_id}_{node_suffix}",
            stage_role=stage_role,
            directive=directive,
            visible_messages=visible,
        )

    def _run_stage(
        self,
        state: dict[str, Any],
        *,
        agent_id: str,
        node_name: str,
        stage_role: str,
        directive: str,
        visible_messages: list[Any],
    ) -> ArtifactRecord | None:
        all_tools = list(state.get("tools", []))
        mutation_names = set(state.get("self_evolved_mutation_tool_names", []))
        committer_id = str(state.get("self_evolved_committer_id", ""))
        is_committer_stage = agent_id == committer_id and stage_role == "aggregator"
        stage_tools = (
            all_tools
            if not mutation_names or is_committer_stage
            else [tool for tool in all_tools if str(tool.get("name", "")) not in mutation_names]
        )
        if mutation_names:
            tool_names = {str(tool.get("name", "")) for tool in all_tools}
            calendar_transaction = {
                "calendar.search_events",
                "calendar.create_event",
            } <= tool_names
            calendar_mode = str(
                state.get("self_evolved_calendar_scheduling_mode", "")
            ).casefold()
            availability_verifier_available = (
                "calendar.find_first_available_slot" in tool_names
            )
            if is_committer_stage:
                directive = (
                    f"{directive} TRANSACTION COMMITTER CONTRACT: You are the only stage allowed "
                    "to change external state. Reconcile the reader reports, verify every "
                    "precondition with read tools, and perform exactly the requested mutation "
                    "once. The mutation is conditional: if the task's trigger condition does not "
                    "hold (for example the entity was already contacted, the record already "
                    "exists, or the requested change is already in place), perform NO mutation "
                    "and state that verified conclusion as the final answer. Do not "
                    "stop at a plan or claim success until the mutation tool succeeds or you "
                    "have verified that no mutation is required. "
                )
                if (
                    calendar_transaction
                    and calendar_mode == "flexible"
                    and availability_verifier_available
                ):
                    directive += (
                        "For availability, call calendar.find_first_available_slot (duration is "
                        "whole minutes, e.g. '30') and use its event_start exactly; do not "
                        "manually infer the gap or re-fetch individual event fields. If and only "
                        "if the booking is required, call calendar.create_event at that verified "
                        "start time. "
                    )
                elif calendar_transaction and calendar_mode == "fixed":
                    directive += (
                        "This request supplies a fixed calendar time. Do not add an availability "
                        "precondition or substitute another time. If the task's other conditions "
                        "hold, call calendar.create_event with the exact event name, participant "
                        "email, start time, and duration requested by the user. "
                    )
            else:
                directive = (
                    f"{directive} READ-ONLY TRANSACTION CONTRACT: State-changing tools are "
                    "intentionally unavailable in this stage. Verify preconditions and produce "
                    "one exact proposed mutation, including canonical identifiers, arguments, "
                    "and interval arithmetic, for the designated committer — or, if the "
                    "verified preconditions show the task's trigger condition does not hold, "
                    "report that no mutation is needed and cite the evidence. Do not claim that "
                    "the external action has already happened. "
                )
                if (
                    calendar_transaction
                    and calendar_mode == "flexible"
                    and availability_verifier_available
                ):
                    directive += (
                        "Call calendar.find_first_available_slot and report its event_start "
                        "exactly; do not manually infer availability or re-fetch individual "
                        "event fields. "
                    )
                elif calendar_transaction and calendar_mode == "fixed":
                    directive += (
                        "The user supplied a fixed calendar time. Preserve it exactly and do not "
                        "invent an availability precondition; report the complete proposed "
                        "calendar.create_event arguments to the committer. "
                    )
        if str(state.get("self_evolved_write_protocol", "")) == "commit_once":
            directive = f"{directive}{_COMMIT_ONCE_DIRECTIVE}"
            tool_names = {str(tool.get("name", "")) for tool in all_tools}
            calendar_mode = str(
                state.get("self_evolved_calendar_scheduling_mode", "")
            ).casefold()
            if (
                calendar_mode == "flexible"
                and "calendar.find_first_available_slot" in tool_names
            ):
                directive += (
                    "For availability, call calendar.find_first_available_slot (duration is "
                    "whole minutes, e.g. '30') and use its event_start exactly. "
                )
            elif calendar_mode == "fixed":
                directive += (
                    "The user supplied a fixed calendar time. Use it exactly and do not add "
                    "an availability precondition. "
                )
        repair_directive = str(state.get("self_evolved_repair_directive", "")).strip()
        if int(state.get("round_index", 0)) > 0 and repair_directive:
            directive = (
                f"{directive} TRACE-BACKED REPAIR DIAGNOSIS (untrusted; verify it against "
                f"the task before acting): {repair_directive} Re-evaluate the prior answer "
                "specifically against this diagnosis and produce a corrected direct answer."
            )
        stage_tool_state = {"tools": stage_tools}
        if _run_has_get_document(stage_tool_state):
            directive = f"{directive}{_READ_FIRST_DIRECTIVE}"
        elif _run_has_tools(stage_tool_state):
            directive = f"{directive}{_TOOL_CHAIN_DIRECTIVE}"
        original_tools = state.get("tools", [])
        try:
            state["tools"] = stage_tools
            updates = self._engine._execute_agent_stage(
                state,
                agent_id=agent_id,
                node_name=node_name,
                stage_role=stage_role,
                directive=directive,
                visible_messages=visible_messages,
            )
        except RuntimeError as exc:
            # A flaky tool tripped the per-agent circuit breaker (the shared stage
            # raises fatal_tool_failure). Static topologies finalize from their other
            # agents; the dynamic orchestrator must not let one agent's tool flakiness
            # kill the whole task. Skip this agent's contribution and continue — the
            # remaining agents / finalize synthesis still produce an answer.
            if "fatal_tool_failure" not in str(exc):
                raise
            print(
                f"[self_evolved] RECOVERED node={node_name} agent_id={agent_id} "
                f"reason=fatal_tool_failure recovery=skip_agent_continue: {exc}",
                flush=True,
            )
            return None
        finally:
            state["tools"] = original_tools
        artifacts = updates.get("artifacts") or []
        apply_updates(state, updates)
        artifact = artifacts[0] if artifacts else None
        self._dedupe_tool_records(artifact)
        self._context.record_evidence(state, artifact, agent_id=agent_id)
        return artifact

    def _dedupe_tool_records(self, artifact: ArtifactRecord | None) -> None:
        """Drop tool records whose (tool, args) already ran earlier this run.

        The artifact dict (and its ``tool_records`` list) is shared by reference
        with ``state['artifacts']`` and the run metadata the benchmark evaluates,
        so stripping the redundant record in place ensures each external action is
        replayed exactly once. Distinct calls (different args) are always kept, so
        diverse parallel search is unaffected — only exact duplicates collapse.
        """

        if not isinstance(artifact, dict):
            return
        records = artifact.get("tool_records")
        if not isinstance(records, list) or not records:
            return
        kept: list[Any] = []
        for record in records:
            if not isinstance(record, dict):
                kept.append(record)
                continue
            name = str(record.get("tool_name", "")).strip()
            if not name or name == "inter_agent_send":
                kept.append(record)
                continue
            arguments = record.get("arguments")
            try:
                arg_sig = json.dumps(arguments, sort_keys=True, default=str)
            except Exception:
                arg_sig = str(arguments)
            signature = f"{name}({arg_sig})"
            if signature in self._executed_signatures:
                continue
            self._executed_signatures.add(signature)
            kept.append(record)
        artifact["tool_records"] = kept

    def _dispatch(
        self,
        state: dict[str, Any],
        *,
        node_name: str,
        active_agents: list[str],
        packet_specs: list[dict[str, Any]] | None = None,
    ) -> None:
        dispatch_id = int(state.get("dispatch_id", -1)) + 1
        state["dispatch_id"] = dispatch_id
        state["active_agents"] = list(active_agents)
        state["phase"] = node_name
        trace = [
            self._engine._draft_event(
                dispatch_id=dispatch_id,
                node_order=0,
                agent_order=-1,
                event_order=0,
                actor="orchestrator",
                event_type="revise",
                payload={
                    "node": node_name,
                    "phase": node_name,
                    "round_index": int(state.get("round_index", 0)),
                    "discussion_index": int(state.get("discussion_index", 0)),
                    "active_agents": list(active_agents),
                },
                token_in=0,
                token_out=0,
                latency_ms=1.0,
                cost_usd=0.0,
                state_id=f"run_{state['run_index']}_dispatch_{dispatch_id}_{node_name}",
            )
        ]
        apply_updates(
            state,
            {
                "trace_payloads": trace,
                "phase_history": [
                    {
                        "phase": node_name,
                        "round_index": int(state.get("round_index", 0)),
                        "discussion_index": int(state.get("discussion_index", 0)),
                        "dispatch_id": dispatch_id,
                        "active_agents": list(active_agents),
                    }
                ],
            },
        )
        if packet_specs:
            emitted = self._engine._emit_packets(
                state, dispatch_id=dispatch_id, phase=node_name, packet_specs=packet_specs
            )
            apply_updates(state, emitted)

    def _emit_task_packets(
        self,
        state: dict[str, Any],
        *,
        sender: str,
        recipients: list[str],
        artifact: ArtifactRecord | None,
    ) -> None:
        if artifact is None or not recipients:
            return
        kind = "task_package" if int(state.get("round_index", 0)) == 0 else "orchestrator_feedback"
        payload = packet_payload_from_artifact(artifact, max_chars=self._packet_max_chars(state))
        packet_specs = [
            {
                "sender": sender,
                "recipients": [recipient],
                "kind": kind,
                "round": int(state.get("round_index", 0)),
                "discussion_index": 0,
                "artifact_id": artifact.get("artifact_id"),
                "payload": payload,
            }
            for recipient in recipients
        ]
        emitted = self._engine._emit_packets(
            state,
            dispatch_id=int(state.get("dispatch_id", -1)),
            phase=str(state.get("phase", "se_task_packets")),
            packet_specs=packet_specs,
        )
        apply_updates(state, emitted)

    def _relay_reports(
        self,
        state: dict[str, Any],
        *,
        group: GroupSpec,
        artifacts: list[ArtifactRecord],
        recipients: list[str],
        kind: str,
    ) -> None:
        if not artifacts or not recipients:
            return
        packet_specs = []
        for artifact in artifacts:
            payload = packet_payload_from_artifact(
                artifact, max_chars=self._packet_max_chars(state)
            )
            packet_specs.append(
                {
                    "sender": str(artifact.get("agent_id", "")),
                    "recipients": list(recipients),
                    "kind": kind,
                    "round": int(state.get("round_index", 0)),
                    "discussion_index": int(state.get("discussion_index", 0)),
                    "artifact_id": artifact.get("artifact_id"),
                    "payload": payload,
                }
            )
        emitted = self._engine._emit_packets(
            state,
            dispatch_id=int(state.get("dispatch_id", -1)),
            phase=f"se_{group.group_id}_relay",
            packet_specs=packet_specs,
        )
        apply_updates(state, emitted)

    # -- output selection -----------------------------------------------------

    @staticmethod
    def _select_group_output(artifacts: list[ArtifactRecord]) -> ArtifactRecord | None:
        """Deterministic group output: majority answer signature, then evidence,
        tool use, and confidence as tie-breakers."""

        if not artifacts:
            return None
        counts: dict[str, int] = {}
        for artifact in artifacts:
            signature = TurnExecutor._decision_answer_signature(str(artifact.get("answer", "")))
            if signature:
                counts[signature] = counts.get(signature, 0) + 1

        def rank(artifact: ArtifactRecord) -> tuple[int, int, float, str]:
            signature = TurnExecutor._decision_answer_signature(str(artifact.get("answer", "")))
            evidence = artifact.get("evidence_summary", [])
            evidence_count = len(evidence) if isinstance(evidence, list) else 0
            used_tools = bool(artifact.get("tool_records"))
            return (
                counts.get(signature, 0),
                evidence_count + int(used_tools),
                float(artifact.get("confidence", 0.5)),
                str(artifact.get("agent_id", "")),
            )

        selected = sorted(artifacts, key=rank, reverse=True)[0]
        selected_text = str(selected.get("answer", ""))
        explicit = TurnExecutor._last_balanced_final_form(
            selected_text, commands=(r"\boxed", r"\fbox")
        )
        if explicit is not None and selected_text.lstrip().startswith(("{", "```")):
            # Invalid JSON often leaves LaTeX in a raw wrapper. Project the already
            # selected explicit final form before a later parser can interpret
            # backslash escapes as control characters. This changes presentation only.
            projected = dict(selected)
            projected["answer"] = rf"\boxed{{{explicit}}}"
            return projected
        return selected

    @staticmethod
    def _decision_answer_signature(text: str) -> str:
        """Recover an explicit final form before comparing malformed wrappers."""

        boxed = TurnExecutor._last_balanced_final_form(text, commands=(r"\boxed", r"\fbox"))
        return answer_signature(boxed if boxed is not None else text)

    @staticmethod
    def _last_balanced_final_form(text: str, *, commands: tuple[str, ...]) -> str | None:
        starts = [(text.rfind(command), command) for command in commands]
        start, command = max(starts, key=lambda item: item[0])
        if start < 0:
            return None
        left = text.find("{", start + len(command))
        if left < 0:
            return None
        depth = 0
        for index in range(left, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[left + 1 : index]
        return None

    @staticmethod
    def _packet_max_chars(state: dict[str, Any]) -> int:
        return max(0, int(state.get("peer_artifact_max_chars", 0)))
