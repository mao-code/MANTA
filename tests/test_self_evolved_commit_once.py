"""Commit-once write protocol: planned topology kept, writes proposed then committed once."""

import csv
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmark.workbench import WorkBenchBenchmark, WorkBenchSandbox
from MAS.config import OpenRouterConfig, SelfEvolvedConfig
from MAS.langgraph_engine import ExperimentSpec
from MAS.llm import OpenRouterLLMClient
from MAS.self_evolved.engine import SelfEvolvedEngine
from MAS.self_evolved.executor import state_changing_tool_names
from MAS.self_evolved.transaction import (
    commit_proposals,
    select_commit_proposals,
    wrap_write_tools_as_proposals,
)

_TABLES = {
    "data/processed/calendar_events.csv": (
        ["event_id", "event_name", "participant_email", "event_start", "duration"],
        [["00000001", "Sprint Planning", "alex@company.com", "2023-11-30 10:00:00", "30"]],
    ),
    "data/processed/emails.csv": (
        ["email_id", "sender/recipient", "subject", "sent_datetime", "body", "status"],
        [["1", "alex@company.com", "Hello", "2023-11-29 09:00:00", "Test body", "inbox"]],
    ),
    "data/processed/analytics_data.csv": (
        ["visitor_id", "date_of_visit", "session_duration_seconds", "user_engaged",
         "traffic_source"],
        [["v1", "2023-11-29", "120", "True", "search"]],
    ),
    "data/processed/project_tasks.csv": (
        ["task_id", "task_name", "assigned_to_email", "list_name", "due_date", "board"],
        [["00000001", "API cleanup", "alex@company.com", "Backlog", "2023-12-10", "Back end"]],
    ),
    "data/processed/customer_relationship_manager_data.csv": (
        ["customer_id", "customer_name", "customer_email", "customer_phone",
         "last_contact_date", "product_interest", "status", "assigned_to_email", "notes",
         "follow_up_by"],
        [["00000001", "Taylor", "taylor@example.com", "123", "2023-11-20", "Software", "Lead",
          "alex@company.com", "Interested", "2023-12-05"]],
    ),
}


def _fixture(base: Path) -> Path:
    for relative, (header, rows) in _TABLES.items():
        path = base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
    directory = base / "data/raw/email_addresses.csv"
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.write_text("alex@company.com\nsam@company.com\n", encoding="utf-8")
    return base


@dataclass(frozen=True)
class _Task:
    task_id: str
    prompt: Any
    reference_answer: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _run_engine(tools: list[dict[str, Any]], write_protocol: str) -> Any:
    client = OpenRouterLLMClient(OpenRouterConfig(api_key=None), {"default": "test-model"})
    engine = SelfEvolvedEngine(client, SelfEvolvedConfig(write_protocol=write_protocol))
    spec = ExperimentSpec(
        topology="self_evolved",
        num_agents=3,
        rounds=2,
        discussion_rounds=1,
        communication_budget_per_agent=2,
        termination_consensus_mode="lexical",
        final_vote_mode="deterministic",
        benchmark_name="workbench",
        enable_dynamic_roles=False,
    )
    return engine.run(
        task=_Task(task_id="t0", prompt="Delete the sprint planning meeting."),
        run_index=0,
        seed=42,
        spec=spec,
        agent_types=["general"],
        tools=tools,
        max_tool_iterations=2,
    )


class TestCommitOnceProtocol(unittest.TestCase):
    def test_config_defaults_to_legacy_and_rejects_unknown_protocol(self) -> None:
        self.assertEqual(SelfEvolvedConfig().write_protocol, "transactional_star")
        with self.assertRaises(ValueError):
            SelfEvolvedConfig(write_protocol="anything").validate()

    def test_declared_side_effect_overrides_the_verb_guess(self) -> None:
        tools = [
            {"name": "email.reply_email", "side_effect": True},
            {"name": "analytics.create_report", "side_effect": False},
            {"name": "calendar.create_event"},
            {"name": "calendar.search_events"},
        ]
        self.assertEqual(
            state_changing_tool_names(tools), {"email.reply_email", "calendar.create_event"}
        )

    def test_proposed_write_is_validated_but_changes_no_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sandbox = WorkBenchSandbox(_fixture(Path(tmpdir)))
            benchmark = WorkBenchBenchmark({"domain": "calendar", "data_dir": tmpdir})
            tools = benchmark._build_tools(sandbox, ["calendar"])
            wrapped, handlers = wrap_write_tools_as_proposals(
                tools, state_changing_tool_names(tools)
            )
            create = next(tool for tool in wrapped if tool["name"] == "calendar.create_event")
            before = len(sandbox.calendar_events)

            result = create["handler"](
                {
                    "event_name": "Review",
                    "participant_email": "sam@company.com",
                    "event_start": "2023-12-01 11:00:00",
                    "duration": "30",
                }
            )

            self.assertEqual(result["status"], "proposed")
            self.assertEqual(len(sandbox.calendar_events), before)
            self.assertIn("calendar.create_event", handlers)
            with self.assertRaises(TypeError):
                create["handler"]({"bogus_argument": "x"})

    def test_commit_uses_the_nearest_proposing_artifact_in_the_selected_lineage(self) -> None:
        write = {"event_name": "Review", "participant_email": "sam@company.com"}
        proposal = {"status": "completed", "output": {"status": "proposed"}}
        artifacts = [
            {"artifact_id": "synthesis", "source_artifact_ids": ["aggregator"], "tool_records": []},
            {
                "artifact_id": "aggregator",
                "source_artifact_ids": ["member"],
                "tool_records": [
                    {"tool_name": "calendar.search_events", "arguments": {}, "status": "completed"},
                    {"tool_name": "calendar.create_event", "arguments": write, **proposal},
                    {"tool_name": "calendar.create_event", "arguments": write, **proposal},
                    {"tool_name": "calendar.delete_event", "arguments": {"event_id": "9"},
                     "status": "error", "error": "not found"},
                ],
            },
            {
                "artifact_id": "member",
                "source_artifact_ids": [],
                "tool_records": [
                    {"tool_name": "calendar.create_event", "arguments": {"event_name": "Other"},
                     **proposal},
                ],
            },
        ]
        mutations = {"calendar.create_event", "calendar.delete_event"}

        source, proposals = select_commit_proposals(artifacts, "synthesis", mutations)

        self.assertEqual(source, "aggregator")
        self.assertEqual(proposals, [{"tool_name": "calendar.create_event", "arguments": write}])

        # A final stage that only re-reads the data keeps the earlier proposal.
        artifacts[1]["tool_records"] = artifacts[1]["tool_records"][:1]
        source, proposals = select_commit_proposals(artifacts, "synthesis", mutations)

        self.assertEqual(source, "member")
        self.assertEqual(
            proposals, [{"tool_name": "calendar.create_event", "arguments": {"event_name": "Other"}}]
        )

    def test_commit_executes_each_proposal_once_and_records_errors(self) -> None:
        calls: list[dict[str, Any]] = []

        def create(arguments: dict[str, Any]) -> str:
            calls.append(arguments)
            return "00000002"

        def delete(arguments: dict[str, Any]) -> str:
            raise ValueError("no such event")

        committed = commit_proposals(
            [
                {"tool_name": "calendar.create_event", "arguments": {"event_name": "Review"}},
                {"tool_name": "calendar.delete_event", "arguments": {"event_id": "9"}},
            ],
            {"calendar.create_event": create, "calendar.delete_event": delete},
        )

        self.assertEqual(calls, [{"event_name": "Review"}])
        self.assertEqual([entry["status"] for entry in committed], ["completed", "error"])

    def test_workbench_scores_committed_writes_not_lineage_proposals(self) -> None:
        run_metadata = {
            "selected_artifact_id": "winner",
            "artifact_records": [
                {
                    "artifact_id": "winner",
                    "source_artifact_ids": [],
                    "tool_records": [
                        {"tool_name": "company_directory.find_email_address",
                         "arguments": {"name": "Sam"}},
                        {"tool_name": "calendar.create_event",
                         "arguments": {"event_name": "Proposal only"}},
                    ],
                }
            ],
            "committed_actions": [
                {"tool_name": "calendar.create_event", "arguments": {"event_name": "Review"},
                 "status": "completed"},
                {"tool_name": "calendar.delete_event", "arguments": {"event_id": "9"},
                 "status": "error"},
            ],
        }

        calls = WorkBenchBenchmark._extract_selected_function_calls(run_metadata, trace_events=[])

        self.assertEqual(
            calls,
            [
                'company_directory.find_email_address.func(name="Sam")',
                'calendar.create_event.func(event_name="Review")',
            ],
        )

    def test_engine_keeps_planned_topology_under_commit_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sandbox = WorkBenchSandbox(_fixture(Path(tmpdir)))
            tools = WorkBenchBenchmark({"domain": "calendar", "data_dir": tmpdir})._build_tools(
                sandbox, ["calendar"]
            )

            commit_once = _run_engine(tools, "commit_once").run_metadata
            legacy = _run_engine(tools, "transactional_star").run_metadata

        def groups(metadata: dict[str, Any]) -> set[str]:
            return {
                group["group_id"]
                for version in metadata["self_evolved"]["topology_spec_versions"]
                for group in version.get("groups", [])
            }

        self.assertNotIn("g_transaction", groups(commit_once))
        self.assertEqual(
            commit_once["self_evolved"]["planner"]["transaction_protocol"]["policy"], "commit_once"
        )
        self.assertIsInstance(commit_once["committed_actions"], list)
        self.assertIn("g_transaction", groups(legacy))
        self.assertNotIn("committed_actions", legacy)


if __name__ == "__main__":
    unittest.main()
