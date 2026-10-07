import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from MAS.config import OpenRouterConfig
from MAS.llm import OpenRouterLLMClient


def _make_completion(
    *,
    content: str = "",
    tool_calls: list | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "tools" in kwargs:
            tool_call = SimpleNamespace(
                id="call_1",
                type="function",
                function=SimpleNamespace(name="search_tool", arguments='{"query":"x"}'),
            )
            return _make_completion(tool_calls=[tool_call], prompt_tokens=10, completion_tokens=2)
        return _make_completion(content="FINAL ANSWER: done", prompt_tokens=5, completion_tokens=4)


class _FakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions())


class _SlowCompletions:
    def create(self, **kwargs):
        time.sleep(0.2)
        return _make_completion(content="late answer", prompt_tokens=3, completion_tokens=2)


class _SlowClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_SlowCompletions())


class _SlowOnceCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            time.sleep(0.2)
            return _make_completion(content="late answer", prompt_tokens=3, completion_tokens=2)
        return _make_completion(content="FINAL ANSWER: timeout retry recovered", prompt_tokens=5, completion_tokens=4)


class _SlowOnceClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_SlowOnceCompletions())


class _EmptyCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=3, completion_tokens=0))


class _EmptyClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_EmptyCompletions())


class _CompactingCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.tool_call_count = 0

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "tools" not in kwargs:
            return _make_completion(content="FINAL ANSWER: compacted", prompt_tokens=6, completion_tokens=4)
        self.tool_call_count += 1
        tool_call = SimpleNamespace(
            id=f"call_{self.tool_call_count}",
            type="function",
            function=SimpleNamespace(
                name="search_tool",
                arguments=json.dumps({"query": f"q{self.tool_call_count}"}),
            ),
        )
        return _make_completion(tool_calls=[tool_call], prompt_tokens=20, completion_tokens=2)


class _CompactingClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_CompactingCompletions())


class _StagnatingCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.tool_call_count = 0

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "tools" not in kwargs:
            return _make_completion(
                content="FINAL ANSWER: insufficient evidence",
                prompt_tokens=7,
                completion_tokens=5,
            )
        self.tool_call_count += 1
        tool_call = SimpleNamespace(
            id=f"call_{self.tool_call_count}",
            type="function",
            function=SimpleNamespace(
                name="search_tool",
                arguments=json.dumps({"query": f"q{self.tool_call_count}"}),
            ),
        )
        return _make_completion(tool_calls=[tool_call], prompt_tokens=18, completion_tokens=2)


class _StagnatingClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_StagnatingCompletions())


class _ReadGateCompletions:
    """Searches stagnate without ever reading; once the read-gate nudges, it opens a
    document via get_document and only then produces a real answer."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.search_count = 0
        self.read_count = 0

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "tools" not in kwargs:
            return _make_completion(content="FINAL ANSWER: forced", prompt_tokens=5, completion_tokens=4)
        joined = "\n".join(str(m.get("content", "")) for m in kwargs.get("messages", []))
        nudged = "have not opened" in joined
        if nudged and self.read_count == 0:
            self.read_count += 1
            tool_call = SimpleNamespace(
                id="read_1",
                type="function",
                function=SimpleNamespace(name="get_document", arguments=json.dumps({"docid": "59188"})),
            )
            return _make_completion(tool_calls=[tool_call], prompt_tokens=15, completion_tokens=2)
        if self.read_count >= 1:
            return _make_completion(
                content="FINAL ANSWER: the capital is Paris", prompt_tokens=8, completion_tokens=5
            )
        self.search_count += 1
        tool_call = SimpleNamespace(
            id=f"search_{self.search_count}",
            type="function",
            function=SimpleNamespace(name="search", arguments=json.dumps({"query": f"q{self.search_count}"})),
        )
        return _make_completion(tool_calls=[tool_call], prompt_tokens=18, completion_tokens=2)


class _ReadGateClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_ReadGateCompletions())


class _FailingSearchCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.tool_call_count = 0

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "tools" not in kwargs:
            return _make_completion(
                content="FINAL ANSWER: blocked by tool failure",
                prompt_tokens=7,
                completion_tokens=5,
            )
        self.tool_call_count += 1
        tool_call = SimpleNamespace(
            id=f"call_{self.tool_call_count}",
            type="function",
            function=SimpleNamespace(
                name="google_web_search",
                arguments=json.dumps({"search_query": f"q{self.tool_call_count}"}),
            ),
        )
        return _make_completion(tool_calls=[tool_call], prompt_tokens=18, completion_tokens=2)


class _FailingSearchClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_FailingSearchCompletions())


class _StructuredSearchCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            tool_call = SimpleNamespace(
                id="crm_search_1",
                type="function",
                function=SimpleNamespace(
                    name="customer_relationship_manager_search_customers",
                    arguments=json.dumps({"customer_name": "Taylor"}),
                ),
            )
            return _make_completion(tool_calls=[tool_call], prompt_tokens=12, completion_tokens=2)
        return _make_completion(
            content="FINAL ANSWER: customer found",
            prompt_tokens=8,
            completion_tokens=4,
        )


class _StructuredSearchClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_StructuredSearchCompletions())


class _SingleToolCallCompletions:
    def __init__(self, tool_name: str, arguments: dict) -> None:
        self.calls: list[dict] = []
        self.tool_name = tool_name
        self.arguments = arguments

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            tool_call = SimpleNamespace(
                id="single_call_1",
                type="function",
                function=SimpleNamespace(
                    name=self.tool_name,
                    arguments=json.dumps(self.arguments),
                ),
            )
            return _make_completion(tool_calls=[tool_call], prompt_tokens=12, completion_tokens=2)
        return _make_completion(
            content="FINAL ANSWER: done",
            prompt_tokens=8,
            completion_tokens=4,
        )


class _SingleToolCallClient:
    def __init__(self, tool_name: str, arguments: dict) -> None:
        self.chat = SimpleNamespace(completions=_SingleToolCallCompletions(tool_name, arguments))


class _RateLimitOnceCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            raise RuntimeError("Error code: 429 retry_after_seconds: 1")
        return _make_completion(content="FINAL ANSWER: recovered", prompt_tokens=5, completion_tokens=4)


class _RateLimitOnceClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_RateLimitOnceCompletions())


class _ForceFinalTimeoutCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "tools" not in kwargs:
            time.sleep(0.2)
            return _make_completion(content="late answer", prompt_tokens=5, completion_tokens=4)
        tool_call = SimpleNamespace(
            id="call_timeout",
            type="function",
            function=SimpleNamespace(name="search_tool", arguments='{"query":"q1"}'),
        )
        return _make_completion(tool_calls=[tool_call], prompt_tokens=12, completion_tokens=2)


class _ForceFinalTimeoutClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_ForceFinalTimeoutCompletions())


class _InitialToolTimeoutCompletions:
    def create(self, **kwargs):
        time.sleep(0.2)
        return _make_completion(content="late tool answer", prompt_tokens=3, completion_tokens=2)


class _InitialToolTimeoutClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_InitialToolTimeoutCompletions())


class _MidToolTimeoutCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.tool_call_count = 0

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "tools" not in kwargs:
            return _make_completion(content="FINAL ANSWER: should not be used", prompt_tokens=5, completion_tokens=4)
        self.tool_call_count += 1
        if self.tool_call_count == 1:
            tool_call = SimpleNamespace(
                id="call_1",
                type="function",
                function=SimpleNamespace(name="search_tool", arguments='{"query":"q1"}'),
            )
            return _make_completion(tool_calls=[tool_call], prompt_tokens=12, completion_tokens=2)
        time.sleep(0.2)
        return _make_completion(content="late tool answer", prompt_tokens=3, completion_tokens=2)


class _MidToolTimeoutClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_MidToolTimeoutCompletions())


class TestLLMToolLoop(unittest.TestCase):
    def test_repairs_provider_quoted_argument_keys(self) -> None:
        self.assertEqual(
            OpenRouterLLMClient._normalize_tool_arguments(
                {'"queries"': ["first", "second"], "query": "preferred"}
            ),
            {"queries": ["first", "second"], "query": "preferred"},
        )

    def test_repairs_provider_quoted_identifier_values_only(self) -> None:
        self.assertEqual(
            OpenRouterLLMClient._normalize_tool_arguments(
                {"docid": '"59188"', "query": '"quoted phrase"'}
            ),
            {"docid": "59188", "query": '"quoted phrase"'},
        )

    def test_resolves_provider_tool_name_with_trailing_punctuation(self) -> None:
        handler = lambda args: args

        self.assertEqual(
            OpenRouterLLMClient._resolve_emitted_tool_name(
                "constraint_search(", {"constraint_search": handler}
            ),
            "constraint_search",
        )
        self.assertEqual(
            OpenRouterLLMClient._resolve_emitted_tool_name("unknown(", {}),
            "unknown(",
        )

    def test_disable_live_override_forces_mock_even_with_api_key(self) -> None:
        with patch.dict("os.environ", {"MAS_DISABLE_LIVE_LLM": "1"}, clear=False):
            client = OpenRouterLLMClient(
                OpenRouterConfig(api_key="test"), {"default": "openai/gpt-4o-mini"}
            )

        self.assertIsNone(client.client)

        result = client.generate(
            prompt="solve this",
            agent_type="general",
            task_id="t1",
            run_index=0,
            agent_id="agent_0",
        )

        self.assertTrue(result.mock_used)

    def test_forces_final_answer_after_tool_limit(self) -> None:
        client = OpenRouterLLMClient(
            OpenRouterConfig(api_key="test"), {"default": "openai/gpt-4o-mini"}
        )
        client.client = _FakeClient()

        result = client.generate(
            prompt=[{"role": "user", "content": "solve this"}],
            agent_type="general",
            task_id="t1",
            run_index=0,
            agent_id="agent_0",
            tools=[
                {
                    "name": "search.tool",
                    "description": "Search tool",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                    "handler": lambda args: {"ok": True},
                }
            ],
            max_tool_iterations=1,
        )

        self.assertFalse(result.mock_used)
        self.assertEqual(result.text, "FINAL ANSWER: done")
        self.assertTrue(result.metadata.get("tool_loop_forced_final_answer"))
        self.assertEqual(result.tool_calls[0]["tool_name"], "search.tool")

    def test_generate_falls_back_to_mock_on_hard_timeout(self) -> None:
        client = OpenRouterLLMClient(
            OpenRouterConfig(api_key="test", timeout_s=0.05), {"default": "openai/gpt-4o-mini"}
        )
        client.client = _SlowClient()

        with patch.dict("os.environ", {"MAS_LLM_TIMEOUT_RETRY_ATTEMPTS": "1"}, clear=False):
            result = client.generate(
                prompt="solve this",
                agent_type="general",
                task_id="t1",
                run_index=0,
                agent_id="agent_0",
            )

        self.assertTrue(result.mock_used)
        self.assertIn("timeout", str(result.metadata.get("fallback_reason", "")).lower())

    def test_generate_retries_hard_timeout_when_enabled(self) -> None:
        client = OpenRouterLLMClient(
            OpenRouterConfig(api_key="test", timeout_s=0.05), {"default": "openai/gpt-4o-mini"}
        )
        fake = _SlowOnceClient()
        client.client = fake

        with patch.dict(
            "os.environ",
            {
                "MAS_LLM_TIMEOUT_RETRY_ATTEMPTS": "2",
                "MAS_LLM_RETRY_BACKOFF_S": "0",
                "MAS_LLM_RETRY_MAX_BACKOFF_S": "1",
                "MAS_OPENROUTER_PROVIDER_ORDER": "DeepInfra,Chutes",
                "MAS_OPENROUTER_SHUFFLE_PROVIDER_ORDER": "1",
            },
            clear=False,
        ):
            result = client.generate(
                prompt="solve this",
                agent_type="general",
                task_id="t1",
                run_index=0,
                agent_id="agent_0",
            )

        self.assertFalse(result.mock_used)
        self.assertIn("timeout retry recovered", result.text)
        self.assertEqual(len(fake.chat.completions.calls), 2)

    def test_compacts_prompt_facing_tool_output_and_summarizes_older_turns(self) -> None:
        client = OpenRouterLLMClient(
            OpenRouterConfig(api_key="test"), {"default": "openai/gpt-4o-mini"}
        )
        client.client = _CompactingClient()

        raw_outputs = {
            "q1": "FIRST_FULL_PAYLOAD::" + ("A" * 1200) + "::END1",
            "q2": "SECOND_FULL_PAYLOAD::" + ("B" * 1200) + "::END2",
            "q3": "THIRD_FULL_PAYLOAD::" + ("C" * 1200) + "::END3",
        }

        with patch.dict(
            "os.environ",
            {
                "MAS_TOOL_CONTEXT_RAW_TURNS": "1",
                "MAS_TOOL_CONTEXT_PREVIEW_CHARS": "160",
                "MAS_TOOL_CONTEXT_SUMMARY_MAX_CHARS": "1800",
            },
            clear=False,
        ):
            result = client.generate(
                prompt=[{"role": "user", "content": "solve this"}],
                agent_type="general",
                task_id="t1",
                run_index=0,
                agent_id="agent_0",
                tools=[
                    {
                        "name": "search.tool",
                        "description": "Search tool",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                        "handler": lambda args: [
                            {
                                "docid": f"doc-{args['query']}",
                                "score": 42.123456,
                                "snippet": raw_outputs[args["query"]],
                            }
                        ],
                    }
                ],
                max_tool_iterations=3,
            )

        completions = client.client.chat.completions
        third_tool_call_messages = completions.calls[2]["messages"]
        joined_content = "\n".join(str(message.get("content", "")) for message in third_tool_call_messages)

        self.assertEqual(result.text, "FINAL ANSWER: compacted")
        self.assertTrue(result.metadata.get("tool_context_compacted"))
        self.assertEqual(result.metadata.get("tool_context_summarized_turns"), 2)
        self.assertIn("Tool-memory summary from earlier tool iterations.", joined_content)
        self.assertIn('"docid": "doc-q2"', joined_content)
        self.assertIn('"snippet_preview"', joined_content)
        self.assertNotIn(raw_outputs["q1"], joined_content)
        self.assertNotIn(raw_outputs["q2"], joined_content)
        self.assertEqual(len(result.tool_calls), 3)

    def test_document_tool_prompt_compaction_keeps_text_preview(self) -> None:
        client = OpenRouterLLMClient(
            OpenRouterConfig(api_key="test"), {"default": "openai/gpt-4o-mini"}
        )
        document_text = "TITLE\n" + ("A" * 180) + " NEEDLE_FACT " + ("B" * 300)

        with patch.dict(
            "os.environ",
            {"MAS_TOOL_CONTEXT_DOCUMENT_CHARS": "240"},
            clear=False,
        ):
            compact = client._compact_tool_output_for_prompt(
                {
                    "docid": "doc-1",
                    "snippet": "TITLE",
                    "text": document_text,
                    "url": "https://example.test/doc-1",
                },
                preview_chars=40,
            )

        self.assertEqual(compact["docid"], "doc-1")
        self.assertEqual(compact["snippet_preview"], "TITLE")
        self.assertIn("NEEDLE_FACT", compact["text_preview"])
        self.assertIn("text_omitted_chars", compact)

    def test_small_structured_tool_output_keeps_fields(self) -> None:
        client = OpenRouterLLMClient(
            OpenRouterConfig(api_key="test"), {"default": "openai/gpt-4o-mini"}
        )

        compact = client._compact_tool_output_for_prompt(
            {
                "college": [
                    {
                        "amenity": "The American University in Cairo",
                        "address": "New Cairo, Cairo, Egypt",
                        "latitude": "30.018923",
                        "longitude": "31.499674",
                    },
                    {
                        "amenity": "Cairo University",
                        "address": "Giza, Cairo, Egypt",
                        "latitude": "30.022052",
                        "longitude": "31.208853",
                    },
                ]
            },
            preview_chars=80,
        )

        self.assertEqual(compact["college"][0]["latitude"], "30.018923")
        self.assertEqual(compact["college"][0]["longitude"], "31.499674")
        self.assertEqual(compact["college"][1]["amenity"], "Cairo University")

    def test_document_tool_prompt_compaction_samples_middle_for_long_text(self) -> None:
        client = OpenRouterLLMClient(
            OpenRouterConfig(api_key="test"), {"default": "openai/gpt-4o-mini"}
        )
        document_text = ("A" * 5000) + " SECRETARY_NEEDLE " + ("B" * 5000)

        with patch.dict(
            "os.environ",
            {"MAS_TOOL_CONTEXT_DOCUMENT_CHARS": "2400"},
            clear=False,
        ):
            compact = client._compact_tool_output_for_prompt(
                {
                    "docid": "doc-1",
                    "snippet": "TITLE",
                    "text": document_text,
                },
                preview_chars=120,
            )

        self.assertIn("SECRETARY_NEEDLE", compact["text_preview"])
        self.assertIn("middle of document omitted", compact["text_preview"])

    def test_numeric_search_results_are_not_tool_failures(self) -> None:
        client = OpenRouterLLMClient(
            OpenRouterConfig(api_key="test"), {"default": "openai/gpt-4o-mini"}
        )

        self.assertIsNone(
            client._tool_output_failure_reason(
                [
                    {
                        "docid": "429",
                        "score": 1.0,
                        "snippet": "A normal BrowseComp result mentioning 432 pages.",
                    }
                ]
            )
        )
        self.assertEqual(
            client._tool_output_failure_reason("error 429 rate limit from provider"),
            "tool returned 429/rate-limit failure",
        )

    def test_missing_required_mutation_output_is_a_tool_failure(self) -> None:
        client = OpenRouterLLMClient(
            OpenRouterConfig(api_key="test"), {"default": "openai/gpt-4o-mini"}
        )

        self.assertEqual(
            client._tool_output_failure_reason("Event name not provided."),
            "tool reported a missing required argument",
        )

    def test_bounds_older_tool_turn_summary_size(self) -> None:
        client = OpenRouterLLMClient(
            OpenRouterConfig(api_key="test"), {"default": "openai/gpt-4o-mini"}
        )
        client.client = _CompactingClient()

        with patch.dict(
            "os.environ",
            {
                "MAS_TOOL_CONTEXT_RAW_TURNS": "1",
                "MAS_TOOL_CONTEXT_PREVIEW_CHARS": "120",
                "MAS_TOOL_CONTEXT_SUMMARY_MAX_CHARS": "220",
            },
            clear=False,
        ):
            client.generate(
                prompt=[{"role": "user", "content": "solve this"}],
                agent_type="general",
                task_id="t1",
                run_index=0,
                agent_id="agent_0",
                tools=[
                    {
                        "name": "search.tool",
                        "description": "Search tool",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                        "handler": lambda args: {
                            "query": args["query"],
                            "payload": "PAYLOAD::" + ("X" * 800),
                        },
                    }
                ],
                max_tool_iterations=3,
            )

        force_final_messages = client.client.chat.completions.calls[3]["messages"]
        summary_messages = [
            str(message.get("content", ""))
            for message in force_final_messages
            if message.get("role") == "system"
            and "Tool-memory summary from earlier tool iterations." in str(message.get("content", ""))
        ]

        self.assertTrue(summary_messages)
        self.assertLessEqual(len(summary_messages[0]), 220)

    def test_recovers_when_force_final_times_out_after_tool_evidence(self) -> None:
        client = OpenRouterLLMClient(
            OpenRouterConfig(api_key="test", timeout_s=0.05), {"default": "openai/gpt-4o-mini"}
        )
        client.client = _ForceFinalTimeoutClient()

        result = client.generate(
            prompt=[{"role": "user", "content": "solve this"}],
            agent_type="general",
            task_id="t1",
            run_index=0,
            agent_id="agent_0",
            tools=[
                {
                    "name": "search.tool",
                    "description": "Search tool",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                    "handler": lambda args: [
                        {
                            "docid": "59188",
                            "score": 50.0,
                            "snippet": f"evidence-for-{args['query']}",
                        }
                    ],
                }
            ],
            max_tool_iterations=1,
        )

        self.assertFalse(result.mock_used)
        self.assertTrue(result.metadata.get("tool_loop_timeout_recovered"))
        self.assertEqual(result.metadata.get("tool_loop_timeout_stage"), "force_final")
        self.assertEqual(
            result.metadata.get("tool_loop_recovery_mode"),
            "best_effort_evidence_fallback",
        )
        self.assertTrue(result.metadata.get("tool_loop_forced_final_answer"))
        self.assertIn("Best-effort conclusion", result.text)
        self.assertIn("Evidence snapshot:", result.text)

    def test_tool_timeout_without_evidence_still_falls_back_to_mock(self) -> None:
        client = OpenRouterLLMClient(
            OpenRouterConfig(api_key="test", timeout_s=0.05), {"default": "openai/gpt-4o-mini"}
        )
        client.client = _InitialToolTimeoutClient()

        result = client.generate(
            prompt=[{"role": "user", "content": "solve this"}],
            agent_type="general",
            task_id="t1",
            run_index=0,
            agent_id="agent_0",
            tools=[
                {
                    "name": "search.tool",
                    "description": "Search tool",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                    "handler": lambda args: {"ok": True},
                }
            ],
            max_tool_iterations=1,
        )

        self.assertTrue(result.mock_used)
        self.assertIn("timeout", str(result.metadata.get("fallback_reason", "")).lower())

    def test_recovers_when_tool_iteration_times_out_after_evidence(self) -> None:
        with patch.dict("os.environ", {"MAS_REQUIRE_LIVE_LLM": "1"}, clear=False):
            client = OpenRouterLLMClient(
                OpenRouterConfig(api_key="test", timeout_s=0.05), {"default": "openai/gpt-4o-mini"}
            )
        client.client = _MidToolTimeoutClient()

        result = client.generate(
            prompt=[{"role": "user", "content": "solve this"}],
            agent_type="general",
            task_id="t1",
            run_index=0,
            agent_id="agent_0",
            tools=[
                {
                    "name": "search.tool",
                    "description": "Search tool",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                    "handler": lambda args: [
                        {
                            "docid": "59188",
                            "score": 50.0,
                            "snippet": f"evidence-for-{args['query']}",
                        }
                    ],
                }
            ],
            max_tool_iterations=3,
        )

        self.assertFalse(result.mock_used)
        self.assertTrue(result.metadata.get("tool_loop_timeout_recovered"))
        self.assertEqual(result.metadata.get("tool_loop_timeout_stage"), "tool_iteration")
        self.assertEqual(result.metadata.get("tool_loop_timeout_iteration"), 2)
        self.assertEqual(
            result.metadata.get("tool_loop_recovery_mode"),
            "best_effort_evidence_fallback",
        )
        self.assertIn("tool-loop iteration timed out", result.metadata.get("tool_loop_stopped_reason", "").lower())
        self.assertIn("Evidence snapshot:", result.text)
        self.assertIn("Best-effort conclusion", result.text)

    def test_stops_early_when_search_results_stagnate(self) -> None:
        client = OpenRouterLLMClient(
            OpenRouterConfig(api_key="test"), {"default": "openai/gpt-4o-mini"}
        )
        client.client = _StagnatingClient()

        with patch.dict("os.environ", {"MAS_SEARCH_STAGNATION_CONSECUTIVE": "1"}, clear=False):
            result = client.generate(
                prompt=[{"role": "user", "content": "solve this"}],
                agent_type="general",
                task_id="t1",
                run_index=0,
                agent_id="agent_0",
                tools=[
                    {
                        "name": "search.tool",
                        "description": "Search tool",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                        "handler": lambda args: [
                            {
                                "docid": "59188",
                                "score": 50.0,
                                "snippet": f"same-doc-for-{args['query']}",
                            }
                        ],
                    }
                ],
                max_tool_iterations=8,
            )

        completions = client.client.chat.completions
        force_final_messages = completions.calls[2]["messages"]
        joined_content = "\n".join(str(message.get("content", "")) for message in force_final_messages)

        self.assertEqual(result.text, "FINAL ANSWER: insufficient evidence")
        self.assertEqual(len(result.tool_calls), 2)
        self.assertIn("Search results stagnated", result.metadata.get("tool_loop_stopped_reason", ""))
        self.assertTrue(result.metadata.get("tool_loop_forced_final_answer"))
        self.assertIn("search results have stagnated", joined_content.lower())
        self.assertIn("insufficient-evidence", joined_content.lower())

    def test_read_gate_forces_get_document_before_blocked_finalize(self) -> None:
        client = OpenRouterLLMClient(
            OpenRouterConfig(api_key="test"), {"default": "openai/gpt-4o-mini"}
        )
        client.client = _ReadGateClient()

        with patch.dict("os.environ", {"MAS_SEARCH_STAGNATION_CONSECUTIVE": "1"}, clear=False):
            result = client.generate(
                prompt=[{"role": "user", "content": "what is the capital"}],
                agent_type="general",
                task_id="t1",
                run_index=0,
                agent_id="agent_0",
                tools=[
                    {
                        "name": "search",
                        "description": "Search tool",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                        "handler": lambda args: [
                            {"docid": "59188", "score": 50.0, "snippet": "same-doc"}
                        ],
                    },
                    {
                        "name": "get_document",
                        "description": "Read a document",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                        "handler": lambda args: {"docid": "59188", "text": "Paris is the capital."},
                    },
                ],
                max_tool_iterations=8,
            )

        tool_names = [c.get("tool_name") for c in result.tool_calls]
        # The read-gate must have redirected the stagnating search loop into a read,
        # and the post-read answer (not a blocked one) must win.
        self.assertIn("get_document", tool_names)
        self.assertEqual(result.text, "FINAL ANSWER: the capital is Paris")

    def test_read_gate_inert_without_get_document_tool(self) -> None:
        # No get_document tool available -> existing stagnation behavior is preserved
        # (stop early with a blocked answer; no read-gate nudge).
        client = OpenRouterLLMClient(
            OpenRouterConfig(api_key="test"), {"default": "openai/gpt-4o-mini"}
        )
        client.client = _StagnatingClient()

        with patch.dict("os.environ", {"MAS_SEARCH_STAGNATION_CONSECUTIVE": "1"}, clear=False):
            result = client.generate(
                prompt=[{"role": "user", "content": "solve this"}],
                agent_type="general",
                task_id="t1",
                run_index=0,
                agent_id="agent_0",
                tools=[
                    {
                        "name": "search.tool",
                        "description": "Search tool",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                        "handler": lambda args: [
                            {"docid": "59188", "score": 50.0, "snippet": f"same-doc-for-{args['query']}"}
                        ],
                    }
                ],
                max_tool_iterations=8,
            )

        self.assertEqual(result.text, "FINAL ANSWER: insufficient evidence")
        self.assertIn("search results stagnated", result.metadata.get("tool_loop_stopped_reason", "").lower())

    def test_tool_failure_circuit_breaker_stops_failed_search_query_loop(self) -> None:
        client = OpenRouterLLMClient(
            OpenRouterConfig(api_key="test"), {"default": "openai/gpt-4o-mini"}
        )
        client.client = _FailingSearchClient()

        with patch.dict("os.environ", {"MAS_TOOL_FAILURE_CIRCUIT_BREAKER": "2"}, clear=False):
            result = client.generate(
                prompt=[{"role": "user", "content": "solve this"}],
                agent_type="general",
                task_id="t1",
                run_index=0,
                agent_id="agent_0",
                tools=[
                    {
                        "name": "google_web_search",
                        "description": "Search tool",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                        "handler": lambda args: {
                            "success": False,
                            "result": "432, message='', url='https://api.tavily.com/search'",
                        },
                    }
                ],
                max_tool_iterations=8,
            )

        completions = client.client.chat.completions
        force_final_messages = completions.calls[2]["messages"]
        joined_content = "\n".join(str(message.get("content", "")) for message in force_final_messages)

        self.assertEqual(result.text, "FINAL ANSWER: blocked by tool failure")
        self.assertEqual(len(result.tool_calls), 2)
        self.assertTrue(result.metadata.get("tool_failure_circuit_breaker_triggered"))
        self.assertIn("Repeated tool failures", result.metadata.get("tool_loop_stopped_reason", ""))
        self.assertTrue(all(call["status"] == "error" for call in result.tool_calls))
        self.assertIn("tool failure circuit breaker", joined_content.lower())

    def test_structured_search_without_query_reaches_handler(self) -> None:
        client = OpenRouterLLMClient(
            OpenRouterConfig(api_key="test"), {"default": "openai/gpt-4o-mini"}
        )
        client.client = _StructuredSearchClient()
        received: list[dict] = []

        result = client.generate(
            prompt=[{"role": "user", "content": "find Taylor"}],
            agent_type="general",
            task_id="workbench:multi_domain_0",
            run_index=0,
            agent_id="agent_0",
            tools=[
                {
                    "name": "customer_relationship_manager.search_customers",
                    "description": "Search customer records by structured filters.",
                    "parameters": {
                        "type": "object",
                        "properties": {"customer_name": {"type": "string"}},
                        "required": [],
                    },
                    "handler": lambda args: received.append(args) or [{"customer_id": "00000001"}],
                }
            ],
            max_tool_iterations=2,
        )

        self.assertEqual(received, [{"customer_name": "Taylor"}])
        self.assertEqual(result.tool_calls[0]["status"], "completed")
        self.assertEqual(result.text, "FINAL ANSWER: customer found")

    def _run_single_search_call(
        self, *, arguments: dict, required: list[str]
    ) -> tuple[list[dict], object]:
        client = OpenRouterLLMClient(
            OpenRouterConfig(api_key="test"), {"default": "openai/gpt-4o-mini"}
        )
        client.client = _SingleToolCallClient("calendar_search_events", arguments)
        received: list[dict] = []
        result = client.generate(
            prompt=[{"role": "user", "content": "first event on December 11"}],
            agent_type="general",
            task_id="workbench:multi_domain_31",
            run_index=0,
            agent_id="agent_0",
            tools=[
                {
                    "name": "calendar.search_events",
                    "description": "Returns events matching the query and optional time bounds.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "time_min": {"type": "string"},
                            "time_max": {"type": "string"},
                        },
                        "required": required,
                    },
                    "handler": lambda args: received.append(args) or [{"event_id": "00000279"}],
                }
            ],
            max_tool_iterations=2,
        )
        return received, result

    def test_optional_query_search_with_date_range_reaches_handler(self) -> None:
        arguments = {"time_min": "2023-12-11 00:00:00", "time_max": "2023-12-11 23:59:59"}

        received, result = self._run_single_search_call(arguments=arguments, required=[])

        self.assertEqual(received, [arguments])
        self.assertEqual(result.tool_calls[0]["status"], "completed")

    def test_required_query_search_with_empty_query_is_rejected(self) -> None:
        received, result = self._run_single_search_call(
            arguments={"query": "", "time_min": "2023-12-11 00:00:00"},
            required=["query"],
        )

        self.assertEqual(received, [])
        self.assertEqual(result.tool_calls[0]["status"], "error")
        self.assertIn("non-empty query", result.tool_calls[0]["error"])

    def test_optional_query_search_without_any_filter_is_rejected(self) -> None:
        received, result = self._run_single_search_call(
            arguments={"query": "  ", "time_min": ""}, required=[]
        )

        self.assertEqual(received, [])
        self.assertEqual(result.tool_calls[0]["status"], "error")
        self.assertIn("non-empty query", result.tool_calls[0]["error"])

    def test_retry_adds_openrouter_provider_and_model_fallbacks(self) -> None:
        client = OpenRouterLLMClient(
            OpenRouterConfig(api_key="test"), {"default": "google/gemma-4-31b-it"}
        )
        fake = _RateLimitOnceClient()
        client.client = fake

        with patch.dict(
            "os.environ",
            {
                "MAS_LLM_RETRY_ATTEMPTS": "2",
                "MAS_LLM_RETRY_BACKOFF_S": "0.1",
                "MAS_LLM_RETRY_MAX_BACKOFF_S": "1",
                "MAS_OPENROUTER_PROVIDER_ORDER": "DeepInfra,Chutes",
                "MAS_OPENROUTER_FALLBACK_MODELS": "google/gemini-2.0-flash-001,qwen/qwen3-32b",
            },
            clear=False,
        ):
            with patch("time.sleep") as sleep_mock:
                result = client.generate(
                    prompt="solve",
                    agent_type="general",
                    task_id="t1",
                    run_index=0,
                    agent_id="agent_0",
                )

        self.assertIn("recovered", result.text)
        calls = fake.chat.completions.calls
        self.assertEqual(len(calls), 2)
        extra_body = calls[0]["extra_body"]
        self.assertEqual(extra_body["provider"]["order"], ["DeepInfra", "Chutes"])
        self.assertTrue(extra_body["provider"]["allow_fallbacks"])
        self.assertEqual(
            extra_body["models"],
            ["google/gemini-2.0-flash-001", "qwen/qwen3-32b"],
        )
        sleep_mock.assert_called_once()

    def test_empty_completion_retry_attempts_can_be_capped(self) -> None:
        client = OpenRouterLLMClient(
            OpenRouterConfig(api_key="test"), {"default": "google/gemma-4-31b-it"}
        )
        fake = _EmptyClient()
        client.client = fake

        with patch.dict(
            "os.environ",
            {
                "MAS_LLM_RETRY_ATTEMPTS": "5",
                "MAS_LLM_EMPTY_COMPLETION_RETRY_ATTEMPTS": "1",
                "MAS_LLM_RETRY_BACKOFF_S": "0",
            },
            clear=False,
        ):
            result = client.generate(
                prompt="solve",
                agent_type="general",
                task_id="t1",
                run_index=0,
                agent_id="agent_0",
            )

        self.assertEqual(len(fake.chat.completions.calls), 1)
        self.assertEqual(result.text, "")
        self.assertTrue(result.metadata.get("empty_completion"))
        self.assertEqual(result.metadata.get("failure_category"), "empty_completion")

    def _empty_reply_run(self, final_text: str):
        calls: list[dict] = []

        class _Completions:
            def create(self, **kwargs):
                calls.append(kwargs)
                if "tools" not in kwargs:
                    return _make_completion(content=final_text, prompt_tokens=5, completion_tokens=4)
                if sum("tools" in call for call in calls) == 1:
                    tool_call = SimpleNamespace(
                        id="call_1",
                        type="function",
                        function=SimpleNamespace(name="search_tool", arguments='{"query":"x"}'),
                    )
                    return _make_completion(tool_calls=[tool_call], prompt_tokens=10)
                return _make_completion(content="", prompt_tokens=10)

        client = OpenRouterLLMClient(
            OpenRouterConfig(api_key="test"), {"default": "google/gemma-4-31b-it"}
        )
        client.client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
        with patch.dict("os.environ", {"MAS_LLM_RETRY_BACKOFF_S": "0"}, clear=False):
            result = client.generate(
                prompt=[{"role": "user", "content": "solve this"}],
                agent_type="general",
                task_id="t1",
                run_index=0,
                agent_id="agent_0",
                tools=[
                    {
                        "name": "search.tool",
                        "description": "Search tool",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                        "handler": lambda args: [],
                    }
                ],
                max_tool_iterations=5,
            )
        return result, calls

    def test_empty_reply_after_tool_results_asks_once_for_a_final_answer(self) -> None:
        from cli.resume import _llm_payload_needs_rerun

        result, calls = self._empty_reply_run("FINAL ANSWER: recovered")

        self.assertEqual(result.text, "FINAL ANSWER: recovered")
        self.assertTrue(result.metadata.get("tool_loop_forced_final_answer"))
        self.assertEqual(result.metadata.get("generation_status"), "answered")
        self.assertNotIn("failure_category", result.metadata)
        self.assertIn("previous reply was empty", calls[-1]["messages"][-1]["content"])
        self.assertFalse(
            _llm_payload_needs_rerun({"mock_used": False, "metadata": result.metadata})
        )

    def test_empty_reply_that_stays_empty_still_marks_the_run_for_rerun(self) -> None:
        from cli.resume import _llm_payload_needs_rerun

        result, _ = self._empty_reply_run("")

        self.assertEqual(result.metadata.get("generation_status"), "failed")
        self.assertTrue(
            _llm_payload_needs_rerun({"mock_used": False, "metadata": result.metadata})
        )


if __name__ == "__main__":
    unittest.main()
