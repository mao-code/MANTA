from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import inspect
import json
import os
import random
import re
import signal
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .config import OpenRouterConfig


@dataclass
class LLMResult:
    text: str
    token_in: int
    token_out: int
    cost_usd: float
    model: str
    mock_used: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _ToolLoopTurn:
    iteration: int
    assistant_msg: dict[str, Any]
    assistant_text: str
    tool_messages: list[dict[str, Any]] = field(default_factory=list)
    tool_records: list[dict[str, Any]] = field(default_factory=list)


class _HardTimeoutExpired(RuntimeError):
    """Raised when an LLM request exceeds the configured wall-clock timeout."""


class _EmptyCompletionError(RuntimeError):
    """Raised when the provider returns a 200 response with no usable choice.

    Distinguishes an *infrastructure* failure (provider returned nothing) from a
    model *choice* (the model deliberately produced an empty/blocked answer). The
    former is retryable across providers; the latter is recorded, not retried.
    """


class OpenRouterLLMClient:
    """OpenRouter chat client with deterministic local fallback."""

    _DEFAULT_MAX_COMPLETION_TOKENS = 4096

    def __init__(self, config: OpenRouterConfig, models: dict[str, str]) -> None:
        self.config = config
        self.models = dict(models)
        self.client = None
        self._request_seq = 0
        self._request_seq_lock = threading.Lock()
        self.disable_live = self._env_flag("MAS_DISABLE_LIVE_LLM")
        self.require_live = (not self.disable_live) and self._env_flag("MAS_REQUIRE_LIVE_LLM")

        if self.disable_live:
            return

        if not config.api_key:
            if self.require_live:
                raise RuntimeError(
                    "MAS_REQUIRE_LIVE_LLM is enabled but OPENROUTER_API_KEY is missing."
                )
            return

        try:
            import openai  # type: ignore
        except Exception as exc:
            if self.require_live:
                raise RuntimeError(
                    "MAS_REQUIRE_LIVE_LLM is enabled but the 'openai' package is unavailable."
                ) from exc
            return

        headers: dict[str, str] = {}
        if config.http_referer:
            headers["HTTP-Referer"] = config.http_referer
        if config.x_title:
            headers["X-Title"] = config.x_title

        kwargs: dict[str, Any] = {
            "base_url": config.base_url,
            "api_key": config.api_key,
            "timeout": config.timeout_s,
        }
        if headers:
            kwargs["default_headers"] = headers

        try:
            self.client = openai.OpenAI(**kwargs)
        except Exception as exc:
            self.client = None
            if self.require_live:
                raise RuntimeError("Failed to initialize the OpenRouter client.") from exc

    def model_for_agent_type(self, agent_type: str) -> str:
        return self.models.get(agent_type, self.models.get("default", "qwen/qwen3-8b"))

    @staticmethod
    def _env_flag(name: str) -> bool:
        return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _env_float(name: str) -> float | None:
        raw = os.getenv(name, "").strip()
        if not raw:
            return None
        return float(raw)

    @staticmethod
    def _env_int(name: str) -> int | None:
        raw = os.getenv(name, "").strip()
        if not raw:
            return None
        return int(raw)

    @staticmethod
    def _env_str(name: str) -> str | None:
        raw = os.getenv(name, "").strip()
        return raw or None

    @staticmethod
    def _env_list(name: str) -> list[str]:
        raw = os.getenv(name, "").strip()
        if not raw:
            return []
        return [item.strip() for item in raw.split(",") if item.strip()]

    def _apply_openrouter_sampling_overrides(
        self,
        kwargs: dict[str, Any],
        *,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        request_kwargs = dict(kwargs)

        temperature_override = self._env_float("OPENROUTER_TEMPERATURE")
        top_p_override = self._env_float("OPENROUTER_TOP_P")
        top_k_override = self._env_int("OPENROUTER_TOP_K")
        max_tokens_override = self._env_int("OPENROUTER_MAX_TOKENS")
        reasoning_effort_override = self._env_str("OPENROUTER_REASONING_EFFORT")

        effective_temperature = (
            temperature_override if temperature_override is not None else temperature
        )
        if effective_temperature is not None:
            request_kwargs["temperature"] = effective_temperature
        if top_p_override is not None:
            request_kwargs["top_p"] = top_p_override
        if top_k_override is not None:
            extra_body = dict(request_kwargs.get("extra_body") or {})
            extra_body["top_k"] = top_k_override
            request_kwargs["extra_body"] = extra_body
        max_tokens = (
            max_tokens_override
            if max_tokens_override is not None
            else self._DEFAULT_MAX_COMPLETION_TOKENS
        )
        if max_tokens < 1:
            raise ValueError("OPENROUTER_MAX_TOKENS must be >= 1")
        request_kwargs["max_tokens"] = max_tokens
        if reasoning_effort_override is not None:
            extra_body = dict(request_kwargs.get("extra_body") or {})
            extra_body["reasoning"] = {
                "effort": reasoning_effort_override,
            }
            request_kwargs["extra_body"] = extra_body

        return request_kwargs

    def _apply_openrouter_routing_overrides(
        self,
        kwargs: dict[str, Any],
        *,
        retry_index: int = 0,
    ) -> dict[str, Any]:
        request_kwargs = dict(kwargs)
        extra_body = dict(request_kwargs.get("extra_body") or {})

        provider = dict(extra_body.get("provider") or {})
        provider.setdefault("allow_fallbacks", True)
        provider_order = self._env_list("MAS_OPENROUTER_PROVIDER_ORDER")
        if provider_order:
            order = list(provider_order)
            if self._env_flag("MAS_OPENROUTER_SHUFFLE_PROVIDER_ORDER"):
                random.shuffle(order)
            provider["order"] = order
        ignored_providers = self._env_list("MAS_OPENROUTER_IGNORE_PROVIDERS")
        if ignored_providers:
            provider["ignore"] = ignored_providers
        extra_body["provider"] = provider

        fallback_models = self._fallback_models_for_request(str(request_kwargs.get("model", "")))
        if fallback_models:
            extra_body["models"] = fallback_models
        request_kwargs["extra_body"] = extra_body
        return request_kwargs

    def _fallback_models_for_request(self, primary_model: str) -> list[str]:
        candidates: list[str] = []
        candidates.extend(self._env_list("MAS_OPENROUTER_FALLBACK_MODELS"))
        for key, value in sorted(self.models.items()):
            if key.startswith("fallback") and str(value).strip():
                candidates.extend(item.strip() for item in str(value).split(",") if item.strip())
        seen = {primary_model}
        fallbacks: list[str] = []
        for model in candidates:
            if model in seen:
                continue
            seen.add(model)
            fallbacks.append(model)
        return fallbacks

    def generate(
        self,
        *,
        prompt: Any,
        agent_type: str,
        task_id: str,
        run_index: int,
        agent_id: str,
        tools: list[dict[str, Any]] | None = None,
        max_tool_iterations: int = 8,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResult:
        model = self.model_for_agent_type(agent_type)
        tool_defs, tool_handlers, original_tool_names = self._normalize_tools(tools or [])
        request_id = self._next_request_id()
        started = time.perf_counter()
        message_count = len(prompt) if isinstance(prompt, list) else 1
        self._log(
            "START "
            f"request_id={request_id} task_id={task_id} run_index={run_index} "
            f"agent_id={agent_id} agent_type={agent_type} model={model} "
            f"messages={message_count} tools={len(tool_defs)} "
            f"max_tool_iterations={max(1, int(max_tool_iterations))}"
        )

        if self.client is not None:
            try:
                if isinstance(prompt, list):
                    messages = prompt
                else:
                    messages = [{"role": "user", "content": str(prompt)}]

                if tool_defs:
                    result = self._generate_with_tools(
                        model=model,
                        messages=messages,
                        tool_defs=tool_defs,
                        tool_handlers=tool_handlers,
                        original_tool_names=original_tool_names,
                        max_tool_iterations=max_tool_iterations,
                        temperature=temperature,
                        request_id=request_id,
                        task_id=task_id,
                        run_index=run_index,
                        agent_id=agent_id,
                    )
                    elapsed_s = max(time.perf_counter() - started, 0.0)
                    self._log(
                        "DONE "
                        f"request_id={request_id} task_id={task_id} run_index={run_index} "
                        f"agent_id={agent_id} model={model} mode=tools "
                        f"elapsed_s={elapsed_s:.2f} token_in={result.token_in} "
                        f"token_out={result.token_out} tool_calls={len(result.tool_calls)}"
                    )
                    return result

                completion = self._create_chat_completion(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text = self._extract_text(completion)
                usage = getattr(completion, "usage", None)
                token_in = int(getattr(usage, "prompt_tokens", self._estimate_tokens(prompt)))
                token_out = int(getattr(usage, "completion_tokens", self._estimate_tokens(text)))
                finish_reason = self._finish_reason(completion)
                empty_completion = not self._completion_has_usable_choice(completion)
                metadata = {
                    "provider": "openrouter",
                    "missing_cost_note": "OpenRouter response did not provide cost_usd; recorded as 0.0",
                    "generation_status": "answered" if text.strip() else "failed",
                    "finish_reason": finish_reason,
                    "hit_output_limit": self._is_output_limit_finish_reason(finish_reason),
                }
                if empty_completion:
                    metadata["empty_completion"] = True
                    metadata["failure_category"] = "empty_completion"

                result = LLMResult(
                    text=text,
                    token_in=token_in,
                    token_out=token_out,
                    cost_usd=0.0,
                    model=model,
                    mock_used=False,
                    metadata=metadata,
                )
                elapsed_s = max(time.perf_counter() - started, 0.0)
                self._log(
                    "DONE "
                    f"request_id={request_id} task_id={task_id} run_index={run_index} "
                    f"agent_id={agent_id} model={model} mode=chat "
                    f"elapsed_s={elapsed_s:.2f} token_in={token_in} token_out={token_out}"
                )
                return result
            except Exception as exc:
                elapsed_s = max(time.perf_counter() - started, 0.0)
                self._log(
                    "ERROR "
                    f"request_id={request_id} task_id={task_id} run_index={run_index} "
                    f"agent_id={agent_id} model={model} elapsed_s={elapsed_s:.2f} "
                    f"error={type(exc).__name__}:{exc}"
                )
                if self.require_live:
                    raise RuntimeError(f"Live OpenRouter generation failed: {exc}") from exc
                result = self._mock_result(
                    prompt=prompt,
                    agent_type=agent_type,
                    task_id=task_id,
                    run_index=run_index,
                    agent_id=agent_id,
                    model=model,
                    fallback_reason=str(exc),
                    tools_available=bool(tool_defs),
                )
                self._log(
                    "FALLBACK "
                    f"request_id={request_id} task_id={task_id} run_index={run_index} "
                    f"agent_id={agent_id} model={model} mock_used=true "
                    f"reason={type(exc).__name__}:{exc}"
                )
                return result

        if self.require_live:
            raise RuntimeError(
                "Live OpenRouter generation is required but the client is unavailable."
            )

        result = self._mock_result(
            prompt=prompt,
            agent_type=agent_type,
            task_id=task_id,
            run_index=run_index,
            agent_id=agent_id,
            model=model,
            fallback_reason="OpenRouter client unavailable or API key missing",
            tools_available=bool(tool_defs),
        )
        elapsed_s = max(time.perf_counter() - started, 0.0)
        self._log(
            "FALLBACK "
            f"request_id={request_id} task_id={task_id} run_index={run_index} "
            f"agent_id={agent_id} model={model} elapsed_s={elapsed_s:.2f} "
            "mock_used=true reason=client_unavailable_or_api_key_missing"
        )
        return result

    def _generate_with_tools(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tool_defs: list[dict[str, Any]],
        tool_handlers: dict[str, Callable[[dict[str, Any]], Any]],
        original_tool_names: dict[str, str],
        max_tool_iterations: int,
        temperature: float,
        request_id: str,
        task_id: str,
        run_index: int,
        agent_id: str,
    ) -> LLMResult:
        base_messages = [dict(item) for item in messages]
        total_token_in = 0
        total_token_out = 0
        tool_call_records: list[dict[str, Any]] = []
        final_text = ""
        stopped_early = False
        stop_reason: str | None = None
        duplicate_call_counts: dict[str, int] = {}
        tool_turns: list[_ToolLoopTurn] = []
        consecutive_stagnant_search_iterations = 0
        previous_search_fingerprint: frozenset[str] | None = None
        prompt_preview_chars = max(80, self._env_int("MAS_TOOL_CONTEXT_PREVIEW_CHARS") or 160)
        latest_context_stats = {
            "raw_turns": 0,
            "summarized_turns": 0,
            "summary_chars": 0,
        }
        timeout_recovery_stage: str | None = None
        timeout_recovery_iteration: int | None = None
        finish_reasons: list[str] = []
        consecutive_tool_failures: dict[str, int] = {}
        tool_failure_circuit_breaker = max(
            1,
            self._env_int("MAS_TOOL_FAILURE_CIRCUIT_BREAKER") or 3,
        )
        search_tool_failure_circuit_breaker = max(
            1,
            self._env_int("MAS_SEARCH_TOOL_FAILURE_CIRCUIT_BREAKER") or 2,
        )
        query_search_tools: set[str] = set()
        # Tools whose schema requires `query` (e.g. BrowseComp `search`) reject an empty
        # query outright; tools with an optional `query` (e.g. WorkBench
        # `calendar.search_events`) accept an empty query when another filter such as a
        # date range is supplied.
        query_required_search_tools: set[str] = set()
        for tool_def in tool_defs:
            function = tool_def.get("function", {})
            if not isinstance(function, dict):
                continue
            parameters = function.get("parameters", {})
            properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
            if not isinstance(properties, dict) or "query" not in properties:
                continue
            api_name = str(function.get("name", "")).strip()
            if api_name:
                names = {api_name, original_tool_names.get(api_name, api_name)}
                query_search_tools.update(names)
                required = parameters.get("required", [])
                if isinstance(required, list) and "query" in required:
                    query_required_search_tools.update(names)
        # Read-gate: in benchmarks that expose a get_document tool, an agent that
        # searches but never opens a document and then gives up with a blocked /
        # "insufficient evidence" answer is the search-without-reading failure mode.
        # When that happens we nudge it to read the top hit instead of bailing out.
        # Inert for benchmarks without get_document (and for mock mode).
        get_document_available = "get_document" in set(original_tool_names.values())
        read_gate_nudged = False

        for iteration_index in range(max(1, int(max_tool_iterations))):
            working_messages, latest_context_stats = self._build_tool_loop_messages(
                base_messages=base_messages,
                tool_turns=tool_turns,
            )
            if latest_context_stats["summarized_turns"] > 0:
                self._log(
                    "TOOL_CONTEXT "
                    f"request_id={request_id} task_id={task_id} run_index={run_index} "
                    f"agent_id={agent_id} raw_turns={latest_context_stats['raw_turns']} "
                    f"summarized_turns={latest_context_stats['summarized_turns']} "
                    f"summary_chars={latest_context_stats['summary_chars']}"
                )
            self._log(
                "TOOL_LOOP "
                f"request_id={request_id} task_id={task_id} run_index={run_index} "
                f"agent_id={agent_id} iteration={iteration_index + 1}/"
                f"{max(1, int(max_tool_iterations))} messages={len(working_messages)}"
            )
            try:
                completion = self._create_chat_completion(
                    model=model,
                    messages=working_messages,
                    temperature=temperature,
                    tools=tool_defs,
                    tool_choice="auto",
                )
            except _HardTimeoutExpired:
                if tool_turns:
                    stop_reason = f"tool_iteration_timeout:iteration={iteration_index + 1}"
                    final_text = self._best_effort_tool_timeout_answer(
                        tool_turns=tool_turns,
                        stop_reason=stop_reason,
                        preview_chars=prompt_preview_chars,
                    )
                    timeout_recovery_stage = "tool_iteration"
                    timeout_recovery_iteration = iteration_index + 1
                    stopped_early = True
                    self._log(
                        "TOOL_LOOP_STOP "
                        f"request_id={request_id} task_id={task_id} run_index={run_index} "
                        f"agent_id={agent_id} reason=tool_iteration_timeout "
                        f"iteration={iteration_index + 1}"
                    )
                    self._log(
                        "RECOVERED "
                        f"request_id={request_id} task_id={task_id} run_index={run_index} "
                        f"agent_id={agent_id} mode=tool_iteration_timeout "
                        f"iteration={iteration_index + 1} "
                        "recovery=best_effort_evidence_fallback"
                    )
                    break
                raise
            usage = getattr(completion, "usage", None)
            total_token_in += int(getattr(usage, "prompt_tokens", 0))
            total_token_out += int(getattr(usage, "completion_tokens", 0))

            if not total_token_in:
                total_token_in = self._estimate_tokens(working_messages)

            choice = completion.choices[0] if getattr(completion, "choices", None) else None
            finish_reason = self._finish_reason(completion)
            if finish_reason:
                finish_reasons.append(finish_reason)
            message = getattr(choice, "message", None)
            if message is None:
                # Empty completion survived all provider retries. Don't silently
                # emit an empty artifact: flag it so the force-final recovery
                # below runs and the result is recorded as a typed failure.
                stopped_early = True
                stop_reason = "empty_completion"
                self._log(
                    "TOOL_LOOP_STOP "
                    f"request_id={request_id} task_id={task_id} run_index={run_index} "
                    f"agent_id={agent_id} reason=empty_completion iteration={iteration_index + 1}"
                )
                break

            message_content = getattr(message, "content", "")
            if isinstance(message_content, list):
                parts = []
                for item in message_content:
                    text = (
                        item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
                    )
                    if isinstance(text, str) and text.strip():
                        parts.append(text)
                assistant_text = "\n".join(parts).strip()
            else:
                assistant_text = str(message_content or "")
            if assistant_text:
                final_text = assistant_text

            raw_tool_calls = list(getattr(message, "tool_calls", None) or [])
            serialized_tool_calls: list[dict[str, Any]] = []
            for tool_call in raw_tool_calls:
                function = getattr(tool_call, "function", None)
                name = str(getattr(function, "name", "") or "")
                arguments_raw = str(getattr(function, "arguments", "") or "{}")
                serialized_tool_calls.append(
                    {
                        "id": str(getattr(tool_call, "id", "")),
                        "type": str(getattr(tool_call, "type", "function") or "function"),
                        "function": {
                            "name": name,
                            "arguments": arguments_raw,
                        },
                    }
                )

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_text,
            }
            if serialized_tool_calls:
                assistant_msg["tool_calls"] = serialized_tool_calls
            current_turn = _ToolLoopTurn(
                iteration=iteration_index + 1,
                assistant_msg=dict(assistant_msg),
                assistant_text=assistant_text,
            )
            self._log(
                "TOOL_LOOP_RESULT "
                f"request_id={request_id} task_id={task_id} run_index={run_index} "
                f"agent_id={agent_id} iteration={iteration_index + 1} "
                f"tool_calls={len(serialized_tool_calls)} assistant_text_chars={len(assistant_text)}"
            )

            if not serialized_tool_calls:
                if (
                    get_document_available
                    and not read_gate_nudged
                    and self._looks_evidence_blocked(assistant_text)
                    and any(
                        self._is_search_tool_name(str(rec.get("tool_name", "")))
                        for rec in tool_call_records
                    )
                    and not self._records_have_successful_read(tool_call_records)
                ):
                    read_gate_nudged = True
                    final_text = ""
                    current_turn.tool_messages.append(
                        {"role": "user", "content": self._READ_GATE_NUDGE}
                    )
                    tool_turns.append(current_turn)
                    self._log(
                        "READ_GATE "
                        f"request_id={request_id} task_id={task_id} run_index={run_index} "
                        f"agent_id={agent_id} iteration={iteration_index + 1} "
                        "reason=blocked_finalize_without_read"
                    )
                    continue
                if not assistant_text.strip():
                    # The model answered with nothing (no text, no tool call). At
                    # temperature 0 a retry repeats it, so ask once for a final answer,
                    # as at the tool-call limit. The empty turn is not kept.
                    stopped_early = True
                    stop_reason = "empty_reply"
                    self._log(
                        "TOOL_LOOP_STOP "
                        f"request_id={request_id} task_id={task_id} run_index={run_index} "
                        f"agent_id={agent_id} reason=empty_reply iteration={iteration_index + 1}"
                    )
                    break
                tool_turns.append(current_turn)
                break

            for tool_call in serialized_tool_calls:
                fn = tool_call["function"]
                api_tool_name = str(fn.get("name") or "")
                handler_key = self._resolve_emitted_tool_name(api_tool_name, tool_handlers)
                tool_name = original_tool_names.get(handler_key, handler_key)
                tool_id = str(tool_call.get("id") or "")
                args_text = str(fn.get("arguments") or "{}")
                try:
                    args = json.loads(args_text) if args_text.strip() else {}
                    if not isinstance(args, dict):
                        args = {"value": args}
                    args = self._normalize_tool_arguments(args)
                except Exception:
                    args = {}

                status = "completed"
                error = None
                output: Any
                handler = tool_handlers.get(handler_key)
                tool_name_lc = tool_name.lower()
                if (
                    "search" in tool_name_lc
                    and tool_name in query_search_tools
                    and not str(args.get("query", "")).strip()
                    and (
                        tool_name in query_required_search_tools
                        or not self._has_non_query_search_argument(args)
                    )
                ):
                    status = "error"
                    error = "Invalid search call: provide a non-empty query string."
                    output = {"error": error}
                elif "get_document" in tool_name_lc and not str(args.get("docid", "")).strip():
                    status = "error"
                    error = "Invalid get_document call: provide a non-empty docid string."
                    output = {"error": error}
                elif handler is None:
                    status = "error"
                    error = f"Unknown tool: {tool_name}"
                    output = {"error": error}
                else:
                    try:
                        self._log(
                            "TOOL_CALL "
                            f"request_id={request_id} task_id={task_id} run_index={run_index} "
                            f"agent_id={agent_id} iteration={iteration_index + 1} "
                            f"tool_name={tool_name} args={self._truncate_for_log(args)}"
                        )
                        output = handler(args)
                        if inspect.isawaitable(output):
                            output = asyncio.run(output)
                    except Exception as exc:
                        status = "error"
                        error = str(exc)
                        output = {"error": error}
                output_failure = self._tool_output_failure_reason(output)
                if status == "completed" and output_failure:
                    status = "error"
                    error = output_failure
                self._log(
                    "TOOL_CALL_RESULT "
                    f"request_id={request_id} task_id={task_id} run_index={run_index} "
                    f"agent_id={agent_id} iteration={iteration_index + 1} "
                    f"tool_name={tool_name} status={status} "
                    f"output={self._truncate_for_log(output)}"
                )

                tool_call_records.append(
                    {
                        "tool_name": tool_name,
                        "arguments": args,
                        "status": status,
                        "error": error,
                        "output": output,
                    }
                )
                current_turn.tool_records.append(tool_call_records[-1])

                sig = f"{api_tool_name}:{args_text}"
                duplicate_call_counts[sig] = duplicate_call_counts.get(sig, 0) + 1
                duplicate_threshold = 8 if str(task_id).startswith("workbench:") else 3
                if duplicate_call_counts[sig] >= duplicate_threshold:
                    self._log(
                        "TOOL_LOOP_STOP "
                        f"request_id={request_id} task_id={task_id} run_index={run_index} "
                        f"agent_id={agent_id} reason=duplicate_tool_call signature={sig}"
                    )
                    stop_reason = f"duplicate_tool_call:{sig}"
                    current_turn.tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "name": api_tool_name,
                            "content": (
                                "Repeated identical tool call detected. This exact call has been made "
                                f"{duplicate_threshold} or more times with the same arguments and is returning the same result. "
                                "Please stop repeating this call and provide your final answer now."
                            ),
                        }
                    )
                    stopped_early = True
                    break

                if isinstance(output, str):
                    output_payload = self._preview_text(output, limit=prompt_preview_chars)
                else:
                    output_payload = self._safe_json_dumps(
                        self._compact_tool_output_for_prompt(
                            output,
                            preview_chars=prompt_preview_chars,
                        )
                    )
                current_turn.tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "name": api_tool_name,
                        "content": output_payload,
                    }
                )
                if status == "error":
                    consecutive_tool_failures[tool_name] = (
                        consecutive_tool_failures.get(tool_name, 0) + 1
                    )
                    failure_limit = (
                        search_tool_failure_circuit_breaker
                        if self._is_search_tool_name(tool_name)
                        else tool_failure_circuit_breaker
                    )
                    if consecutive_tool_failures[tool_name] >= failure_limit:
                        self._log(
                            "TOOL_LOOP_STOP "
                            f"request_id={request_id} task_id={task_id} run_index={run_index} "
                            f"agent_id={agent_id} reason=tool_failure_circuit_breaker "
                            f"tool_name={tool_name} "
                            f"consecutive={consecutive_tool_failures[tool_name]}"
                        )
                        stop_reason = (
                            "tool_failure_circuit_breaker:"
                            f"tool={tool_name},consecutive={consecutive_tool_failures[tool_name]},"
                            f"limit={failure_limit}"
                        )
                        current_turn.tool_messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"The tool {tool_name} has failed "
                                    f"{consecutive_tool_failures[tool_name]} consecutive times. "
                                    "Stop calling this tool now. Based only on the evidence already "
                                    "gathered, provide your best final answer. If the evidence is "
                                    "insufficient, explicitly return a concise blocked-status answer."
                                ),
                            }
                        )
                        final_text = ""
                        stopped_early = True
                        break
                else:
                    consecutive_tool_failures[tool_name] = 0

            if not stopped_early:
                search_fingerprint = self._search_iteration_fingerprint(current_turn.tool_records)
                stagnation_threshold = max(
                    1,
                    self._env_int("MAS_SEARCH_STAGNATION_CONSECUTIVE") or 2,
                )
                if search_fingerprint:
                    if previous_search_fingerprint is not None and self._search_results_stagnated(
                        previous_search_fingerprint,
                        search_fingerprint,
                    ):
                        consecutive_stagnant_search_iterations += 1
                    else:
                        consecutive_stagnant_search_iterations = 0
                    previous_search_fingerprint = search_fingerprint
                else:
                    consecutive_stagnant_search_iterations = 0
                    previous_search_fingerprint = None

                if (
                    search_fingerprint
                    and not str(task_id).startswith("workbench:")
                    and consecutive_stagnant_search_iterations >= stagnation_threshold
                ):
                    if (
                        get_document_available
                        and not read_gate_nudged
                        and not self._records_have_successful_read(tool_call_records)
                    ):
                        # Searches have stagnated but no document was ever opened.
                        # Redirect to reading the top hit instead of bailing out with
                        # a blocked answer — that is the real evidence the agent needs.
                        read_gate_nudged = True
                        consecutive_stagnant_search_iterations = 0
                        previous_search_fingerprint = None
                        current_turn.tool_messages.append(
                            {"role": "user", "content": self._READ_GATE_NUDGE}
                        )
                        self._log(
                            "READ_GATE "
                            f"request_id={request_id} task_id={task_id} run_index={run_index} "
                            f"agent_id={agent_id} iteration={iteration_index + 1} "
                            "reason=stagnation_without_read"
                        )
                    else:
                        self._log(
                            "TOOL_LOOP_STOP "
                            f"request_id={request_id} task_id={task_id} run_index={run_index} "
                            f"agent_id={agent_id} reason=search_results_stagnated "
                            f"consecutive={consecutive_stagnant_search_iterations} "
                            f"fingerprint_size={len(search_fingerprint)}"
                        )
                        stop_reason = (
                            "search_results_stagnated:"
                            f"consecutive={consecutive_stagnant_search_iterations}"
                        )
                        current_turn.tool_messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Recent search iterations are stagnating: consecutive search calls are "
                                    "returning substantially overlapping documents without meaningful new evidence. "
                                    "Stop searching now. Based only on the evidence already gathered, provide "
                                    "your best final answer. If the evidence is insufficient, explicitly return "
                                    "a concise insufficient-evidence or blocked-status answer instead of making "
                                    "more paraphrase searches."
                                ),
                            }
                        )
                        final_text = ""
                        stopped_early = True

            tool_turns.append(current_turn)
            if stopped_early:
                break
        else:
            stopped_early = True
            stop_reason = "max_tool_iterations"

        if stopped_early and not final_text:
            final_messages, latest_context_stats = self._build_tool_loop_messages(
                base_messages=base_messages,
                tool_turns=tool_turns,
            )
            self._log(
                "FORCE_FINAL "
                f"request_id={request_id} task_id={task_id} run_index={run_index} "
                f"agent_id={agent_id}"
            )
            try:
                final_text, extra_in, extra_out, force_finish_reason = self._force_final_response(
                    model=model,
                    messages=final_messages,
                    temperature=temperature,
                    max_tool_iterations=max_tool_iterations,
                    stop_reason=stop_reason,
                )
                total_token_in += extra_in
                total_token_out += extra_out
                if force_finish_reason:
                    finish_reasons.append(force_finish_reason)
            except _HardTimeoutExpired:
                if any(turn.tool_records for turn in tool_turns):
                    final_text = self._best_effort_tool_timeout_answer(
                        tool_turns=tool_turns,
                        stop_reason=stop_reason,
                        preview_chars=prompt_preview_chars,
                    )
                    timeout_recovery_stage = "force_final"
                    self._log(
                        "RECOVERED "
                        f"request_id={request_id} task_id={task_id} run_index={run_index} "
                        f"agent_id={agent_id} mode=force_final_timeout "
                        "recovery=best_effort_evidence_fallback"
                    )
                else:
                    raise

        if not total_token_out:
            total_token_out = self._estimate_tokens(final_text)

        metadata = {
            "provider": "openrouter",
            "missing_cost_note": "OpenRouter response did not provide cost_usd; recorded as 0.0",
            "tool_context_raw_turns": latest_context_stats["raw_turns"],
            "tool_context_summarized_turns": latest_context_stats["summarized_turns"],
            "finish_reasons": finish_reasons,
            "hit_output_limit": any(
                self._is_output_limit_finish_reason(reason) for reason in finish_reasons
            ),
        }
        if latest_context_stats["summarized_turns"] > 0:
            metadata["tool_context_compacted"] = True
        if timeout_recovery_stage is not None:
            metadata["tool_loop_timeout_recovered"] = True
            metadata["tool_loop_timeout_stage"] = timeout_recovery_stage
            metadata["tool_loop_recovery_mode"] = "best_effort_evidence_fallback"
            if timeout_recovery_iteration is not None:
                metadata["tool_loop_timeout_iteration"] = timeout_recovery_iteration
        if stopped_early:
            if stop_reason and stop_reason.startswith("duplicate_tool_call:"):
                repeated_sig = stop_reason.split(":", 1)[1]
                metadata["tool_loop_stopped_reason"] = (
                    f"Duplicate tool call detected (repeated >=3 times): {repeated_sig}"
                )
            elif stop_reason and stop_reason.startswith("search_results_stagnated:"):
                metadata["tool_loop_stopped_reason"] = (
                    "Search results stagnated across consecutive iterations; "
                    f"{stop_reason.split(':', 1)[1]}"
                )
            elif stop_reason and stop_reason.startswith("tool_iteration_timeout:"):
                metadata["tool_loop_stopped_reason"] = (
                    "Tool-loop iteration timed out; returned a best-effort answer from gathered evidence. "
                    f"{stop_reason.split(':', 1)[1]}"
                )
            elif stop_reason and stop_reason.startswith("tool_failure_circuit_breaker:"):
                metadata["tool_loop_stopped_reason"] = (
                    "Repeated tool failures triggered the circuit breaker; "
                    f"{stop_reason.split(':', 1)[1]}"
                )
                metadata["tool_failure_circuit_breaker_triggered"] = True
                details = self._parse_stop_reason_details(stop_reason)
                failed_tool = str(details.get("tool", ""))
                metadata["tool_failure_tool_name"] = failed_tool
                metadata["tool_failure_consecutive_count"] = int(details.get("consecutive", 0) or 0)
                metadata["tool_failure_limit"] = int(details.get("limit", 0) or 0)
                metadata["tool_failure_is_search"] = self._is_search_tool_name(failed_tool)
                if metadata["tool_failure_is_search"]:
                    metadata["fatal_tool_failure"] = True
                    metadata["failure_category"] = "tool_failure"
            elif stop_reason == "empty_completion":
                metadata["tool_loop_stopped_reason"] = (
                    "Provider returned an empty completion (no usable choice) after retries."
                )
                metadata["empty_completion"] = True
                metadata["failure_category"] = "empty_completion"
            elif stop_reason == "empty_reply":
                metadata["tool_loop_stopped_reason"] = (
                    "The model returned an empty reply; it was asked once for a final answer."
                )
            else:
                metadata["tool_loop_stopped_reason"] = (
                    f"Reached max_tool_iterations={max(1, int(max_tool_iterations))}"
                )
            if final_text:
                metadata["tool_loop_forced_final_answer"] = True

        # Single typed post-condition: every result carries an explicit
        # generation_status so an infrastructure failure (provider returned
        # nothing) is never indistinguishable from a deliberate empty answer.
        metadata["generation_status"] = "answered" if final_text.strip() else "failed"

        return LLMResult(
            text=final_text,
            token_in=total_token_in,
            token_out=total_token_out,
            cost_usd=0.0,
            model=model,
            mock_used=False,
            metadata=metadata,
            tool_calls=tool_call_records,
        )

    def _mock_result(
        self,
        *,
        prompt: Any,
        agent_type: str,
        task_id: str,
        run_index: int,
        agent_id: str,
        model: str,
        fallback_reason: str,
        tools_available: bool,
    ) -> LLMResult:
        prompt_str = (
            prompt
            if isinstance(prompt, str)
            else " ".join(str(m.get("content", "")) for m in prompt)
        )
        seed_value = self._stable_seed(task_id, str(run_index), agent_id, agent_type, prompt_str)
        rng = random.Random(seed_value)

        words = [token for token in re.split(r"\s+", prompt_str.strip()) if token]
        if not words:
            words = ["empty", "prompt"]
        sampled = [words[rng.randrange(len(words))] for _ in range(min(8, len(words) + 2))]
        # Avoid square brackets so benchmark citation/docid regexes do not
        # treat mock metadata as retrieved evidence docids.
        answer = (
            f"MOCK({agent_id}|{agent_type}) "
            f"Synthesized response with seed={seed_value % 100000}: " + " ".join(sampled)
        )

        return LLMResult(
            text=answer,
            token_in=self._estimate_tokens(prompt),
            token_out=self._estimate_tokens(answer),
            cost_usd=0.0,
            model=model,
            mock_used=True,
            metadata={
                "provider": "mock",
                "fallback_reason": fallback_reason,
                "seed": seed_value,
                "tools_available": tools_available,
            },
        )

    @staticmethod
    def _normalize_tools(
        tools: list[dict[str, Any]],
    ) -> tuple[
        list[dict[str, Any]],
        dict[str, Callable[[dict[str, Any]], Any]],
        dict[str, str],
    ]:
        defs: list[dict[str, Any]] = []
        handlers: dict[str, Callable[[dict[str, Any]], Any]] = {}
        original_names: dict[str, str] = {}
        used_names: set[str] = set()
        for tool in tools:
            original_name = str(tool.get("name", "")).strip()
            handler = tool.get("handler")
            if not original_name or not callable(handler):
                continue
            api_name = OpenRouterLLMClient._sanitize_tool_name(original_name, used_names)
            description = str(tool.get("description", "")).strip() or f"Call {original_name}"
            params = tool.get("parameters")
            if not isinstance(params, dict):
                params = {"type": "object", "properties": {}, "required": []}
            defs.append(
                {
                    "type": "function",
                    "function": {
                        "name": api_name,
                        "description": description,
                        "parameters": params,
                    },
                }
            )
            handlers[api_name] = handler
            # Some models still emit the original dotted tool name after seeing it in
            # the prompt context, even when the API-exposed function name is sanitized.
            # Resolve both forms to the same callable so tool execution remains stable.
            handlers[original_name] = handler
            original_names[api_name] = original_name
            original_names[original_name] = original_name
        return defs, handlers, original_names

    def _build_tool_loop_messages(
        self,
        *,
        base_messages: list[dict[str, Any]],
        tool_turns: list[_ToolLoopTurn],
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        raw_turn_window = max(1, self._env_int("MAS_TOOL_CONTEXT_RAW_TURNS") or 2)
        preview_chars = max(80, self._env_int("MAS_TOOL_CONTEXT_PREVIEW_CHARS") or 160)
        summary_max_chars = max(
            160,
            self._env_int("MAS_TOOL_CONTEXT_SUMMARY_MAX_CHARS") or 6000,
        )
        summarized_turns = max(0, len(tool_turns) - raw_turn_window)
        older_turns = tool_turns[:summarized_turns]
        recent_turns = tool_turns[summarized_turns:]
        messages = [dict(item) for item in base_messages]
        summary_chars = 0

        if older_turns:
            summary = self._summarize_tool_turns(
                older_turns,
                preview_chars=preview_chars,
                max_chars=summary_max_chars,
            )
            summary_chars = len(summary)
            messages = self._insert_context_message(
                messages,
                {
                    "role": "system",
                    "content": summary,
                },
            )

        for turn in recent_turns:
            messages.append(dict(turn.assistant_msg))
            for tool_message in turn.tool_messages:
                messages.append(dict(tool_message))

        return messages, {
            "raw_turns": len(recent_turns),
            "summarized_turns": len(older_turns),
            "summary_chars": summary_chars,
        }

    def _summarize_tool_turns(
        self,
        tool_turns: list[_ToolLoopTurn],
        *,
        preview_chars: int,
        max_chars: int,
    ) -> str:
        tool_call_count = sum(len(turn.tool_records) for turn in tool_turns)
        lines = [
            "Tool-memory summary from earlier tool iterations.",
            (
                "This is a compressed summary of older tool-use. "
                "Prefer the latest raw tool messages later in the conversation if there is any conflict."
            ),
            f"Earlier iterations summarized: {len(tool_turns)}",
            f"Earlier tool calls summarized: {tool_call_count}",
        ]

        for turn in tool_turns:
            assistant_preview = self._preview_text(turn.assistant_text, limit=preview_chars)
            if assistant_preview:
                lines.append(f"Iteration {turn.iteration} assistant: {assistant_preview}")
            else:
                lines.append(f"Iteration {turn.iteration} assistant: [no direct answer text]")
            for record in turn.tool_records:
                compact_output = self._compact_tool_output_for_prompt(
                    record.get("output"),
                    preview_chars=preview_chars,
                )
                output_limit = preview_chars
                output = record.get("output")
                if isinstance(output, dict) and output.get("text"):
                    output_limit = max(
                        preview_chars,
                        min(
                            max_chars,
                            self._env_int("MAS_TOOL_CONTEXT_DOCUMENT_CHARS") or 12000,
                        ),
                    )
                lines.append(
                    f"Iteration {turn.iteration} tool {record['tool_name']} "
                    f"args={self._truncate_for_log(record.get('arguments'), limit=preview_chars)} "
                    f"status={record.get('status', 'completed')} "
                    "output="
                    f"{self._truncate_for_log(compact_output, limit=output_limit)}"
                )

        clipped: list[str] = []
        used = 0
        for line in lines:
            addition = ("\n" if clipped else "") + line
            if used + len(addition) <= max_chars:
                clipped.append(line)
                used += len(addition)
                continue

            remaining = max_chars - used - (1 if clipped else 0)
            marker = "... [truncated]"
            if remaining >= len(marker):
                prefix = line[: max(0, remaining - len(marker))].rstrip()
                clipped.append((prefix + marker) if prefix else marker[:remaining])
            elif not clipped:
                clipped.append("Tool-memory summary truncated.")
            else:
                clipped.append(marker[: max(0, remaining)])
            break

        return "\n".join(clipped)

    def _compact_tool_output_for_prompt(
        self,
        output: Any,
        *,
        preview_chars: int,
        max_list_items: int = 2,
    ) -> Any:
        if isinstance(output, (dict, list)) and not self._looks_like_retrieval_output(output):
            full_json_chars = max(
                preview_chars,
                self._env_int("MAS_TOOL_CONTEXT_FULL_JSON_CHARS") or 12000,
            )
            jsonable_output = self._to_jsonable(output)
            if len(self._safe_json_dumps(jsonable_output)) <= full_json_chars:
                return jsonable_output

        if isinstance(output, list):
            compact_items = [
                self._compact_tool_output_for_prompt(
                    item,
                    preview_chars=preview_chars,
                    max_list_items=max_list_items,
                )
                for item in output[:max_list_items]
            ]
            if len(output) > max_list_items:
                compact_items.append({"omitted_items": len(output) - max_list_items})
            return compact_items

        if isinstance(output, dict):
            compact: dict[str, Any] = {}
            if output.get("docid") is not None:
                compact["docid"] = str(output.get("docid"))

            if output.get("score") is not None:
                try:
                    compact["score"] = round(float(output.get("score")), 3)
                except (TypeError, ValueError):
                    compact["score"] = self._preview_text(output.get("score"), limit=preview_chars)

            for key in ("title", "name", "query"):
                value = output.get(key)
                if value:
                    compact[key] = self._preview_text(value, limit=preview_chars)

            snippet = output.get("snippet")
            if snippet:
                # A search hit's snippet is the only evidence the calling agent gets
                # from that hit (full document text is fetched separately via
                # get_document). Give it a much larger budget than generic preview
                # fields so the agent that ran the search is not starved of its own
                # retrieval evidence and forced to re-search instead of reading.
                snippet_chars = max(
                    preview_chars,
                    self._env_int("MAS_TOOL_CONTEXT_SNIPPET_CHARS") or 1200,
                )
                compact["snippet_preview"] = self._preview_text(snippet, limit=snippet_chars)

            text = output.get("text")
            if text:
                document_chars = max(
                    preview_chars,
                    self._env_int("MAS_TOOL_CONTEXT_DOCUMENT_CHARS") or 12000,
                )
                text_value = str(text)
                compact["text_preview"] = self._preview_document_text(
                    text_value,
                    limit=document_chars,
                )
                if len(text_value) > document_chars:
                    compact["text_omitted_chars"] = len(text_value) - document_chars
            elif output.get("content"):
                compact["content_preview"] = self._preview_text(
                    output.get("content"),
                    limit=preview_chars,
                )
            elif output.get("payload"):
                compact["payload_preview"] = self._preview_text(
                    output.get("payload"),
                    limit=preview_chars,
                )

            if output.get("url"):
                compact["url"] = self._preview_text(
                    output.get("url"), limit=min(120, preview_chars)
                )
            if output.get("error"):
                compact["error"] = self._preview_text(output.get("error"), limit=preview_chars)

            extra_fields = 0
            special_keys = {
                "docid",
                "score",
                "title",
                "name",
                "query",
                "snippet",
                "text",
                "content",
                "payload",
                "url",
                "error",
            }
            for key, value in output.items():
                if key in special_keys or extra_fields >= 2:
                    continue
                if isinstance(value, str):
                    compact[str(key)] = self._preview_text(value, limit=preview_chars)
                    extra_fields += 1
                elif isinstance(value, (int, float, bool)) or value is None:
                    compact[str(key)] = value
                    extra_fields += 1

            if compact:
                return compact
            return {"preview": self._truncate_for_log(output, limit=preview_chars)}

        if isinstance(output, str):
            return self._preview_text(output, limit=preview_chars)

        return self._to_jsonable(output)

    @staticmethod
    def _looks_like_retrieval_output(output: Any) -> bool:
        retrieval_keys = {
            "docid",
            "score",
            "title",
            "query",
            "snippet",
            "text",
            "content",
            "payload",
            "url",
            "error",
        }
        if isinstance(output, dict):
            if any(key in output for key in retrieval_keys):
                return True
            return any(
                OpenRouterLLMClient._looks_like_retrieval_output(value)
                for value in output.values()
                if isinstance(value, (dict, list))
            )
        if isinstance(output, list):
            return any(
                OpenRouterLLMClient._looks_like_retrieval_output(item)
                for item in output
                if isinstance(item, (dict, list))
            )
        return False

    def _best_effort_tool_timeout_answer(
        self,
        *,
        tool_turns: list[_ToolLoopTurn],
        stop_reason: str | None,
        preview_chars: int,
    ) -> str:
        latest_assistant = ""
        for turn in reversed(tool_turns):
            if turn.assistant_text.strip():
                latest_assistant = self._preview_text(turn.assistant_text, limit=360)
                break

        evidence_lines: list[str] = []
        seen_lines: set[str] = set()
        for turn in reversed(tool_turns):
            for record in turn.tool_records:
                compact_output = self._compact_tool_output_for_prompt(
                    record.get("output"),
                    preview_chars=preview_chars,
                )
                compact_text = self._truncate_for_log(compact_output, limit=preview_chars)
                if not compact_text or compact_text in seen_lines:
                    continue
                evidence_line = f"{record['tool_name']}: {compact_text}"
                evidence_lines.append(evidence_line)
                seen_lines.add(compact_text)
                if len(evidence_lines) >= 4:
                    break
            if len(evidence_lines) >= 4:
                break

        lines = [
            "The tool-assisted run timed out before final synthesis.",
        ]
        if latest_assistant:
            lines.append(f"Latest reasoning: {latest_assistant}")
        if evidence_lines:
            lines.append("Evidence snapshot:")
            lines.extend(f"- {line}" for line in evidence_lines)
        if stop_reason and stop_reason.startswith("search_results_stagnated:"):
            lines.append(
                "Best-effort conclusion: insufficient evidence from the gathered search results."
            )
        else:
            lines.append(
                "Best-effort conclusion: insufficient evidence to provide a reliable final answer."
            )
        return "\n".join(lines)

    def _preview_document_text(self, value: Any, *, limit: int) -> str:
        text = str(value or "")
        if len(text) <= limit:
            return text
        if limit < 900:
            return self._preview_text(text, limit=limit)

        marker_1 = "\n... [middle of document omitted] ...\n"
        marker_2 = "\n... [later middle omitted] ...\n"
        content_budget = max(1, limit - len(marker_1) - len(marker_2))
        head_chars = max(1, content_budget // 2)
        middle_chars = max(1, content_budget // 4)
        tail_chars = max(1, content_budget - head_chars - middle_chars)
        middle_start = max(0, (len(text) - middle_chars) // 2)

        return (
            text[:head_chars].rstrip()
            + marker_1
            + text[middle_start : middle_start + middle_chars].strip()
            + marker_2
            + text[-tail_chars:].lstrip()
        )

    @staticmethod
    def _insert_context_message(
        base_messages: list[dict[str, Any]],
        context_message: dict[str, Any],
    ) -> list[dict[str, Any]]:
        insert_at = 0
        while insert_at < len(base_messages) and base_messages[insert_at].get("role") == "system":
            insert_at += 1
        return [
            *[dict(item) for item in base_messages[:insert_at]],
            dict(context_message),
            *[dict(item) for item in base_messages[insert_at:]],
        ]

    def _search_iteration_fingerprint(
        self,
        tool_records: list[dict[str, Any]],
    ) -> frozenset[str]:
        fingerprints: list[str] = []
        for record in tool_records:
            tool_name = str(record.get("tool_name", "") or "").lower()
            arguments = record.get("arguments")
            if "search" not in tool_name:
                continue
            if not isinstance(arguments, dict):
                continue
            query = self._extract_search_query(arguments)
            output = record.get("output")
            docids = self._search_result_docids(output)
            if docids:
                fingerprints.extend(docids[:3])
                continue
            output_text = self._truncate_for_log(output, limit=160)
            if output_text:
                fingerprints.append(
                    f"text:{hashlib.sha256(output_text.encode('utf-8')).hexdigest()[:16]}"
                )
            elif query:
                fingerprints.append(
                    f"query:{hashlib.sha256(query.encode('utf-8')).hexdigest()[:16]}"
                )
            else:
                fingerprints.append("empty")
        return frozenset(fingerprints)

    @staticmethod
    def _extract_search_query(arguments: dict[str, Any]) -> str:
        for key in ("query", "search_query", "q", "search", "text", "keyword"):
            value = arguments.get(key)
            if value is None:
                continue
            text = " ".join(str(value).lower().split())
            if text:
                return text
        return ""

    @staticmethod
    def _is_search_tool_name(tool_name: str) -> bool:
        lowered = str(tool_name or "").lower()
        return "search" in lowered or "tavily" in lowered or "serp" in lowered

    _READ_GATE_NUDGE = (
        "You have run searches and found candidate documents, but you have not opened any of "
        "them with get_document. Before concluding, call get_document on the single most "
        "relevant docid from your search results to read its full text. Do not answer "
        "'insufficient evidence' or return a blocked status until you have read at least one "
        "full document."
    )

    _READ_GATE_BLOCKED_MARKERS = (
        "insufficient evidence",
        "insufficient information",
        "no evidence",
        "not enough information",
        "cannot determine",
        "unable to determine",
        "blocked",
    )

    @staticmethod
    def _looks_evidence_blocked(text: str) -> bool:
        lowered = str(text or "").lower()
        return any(marker in lowered for marker in OpenRouterLLMClient._READ_GATE_BLOCKED_MARKERS)

    @staticmethod
    def _records_have_successful_read(records: list[dict[str, Any]]) -> bool:
        return any(
            str(rec.get("tool_name")) == "get_document" and rec.get("status") == "completed"
            for rec in records
        )

    @staticmethod
    def _parse_stop_reason_details(stop_reason: str | None) -> dict[str, str]:
        if not stop_reason or ":" not in stop_reason:
            return {}
        details: dict[str, str] = {}
        for part in stop_reason.split(":", 1)[1].split(","):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            details[key.strip()] = value.strip()
        return details

    def _tool_output_failure_reason(self, output: Any) -> str | None:
        if isinstance(output, dict):
            success = output.get("success")
            if success is False:
                return self._preview_text(
                    output.get("error") or output.get("result") or output,
                    limit=240,
                )
            if output.get("error"):
                return self._preview_text(output.get("error"), limit=240)

        if not isinstance(output, str):
            return None

        output_text = self._truncate_for_log(output, limit=320).lower()
        if "429" in output_text and (
            "rate" in output_text or "limit" in output_text or "error" in output_text
        ):
            return "tool returned 429/rate-limit failure"
        if "432" in output_text and "error" in output_text:
            return "tool returned 432 failure"
        if re.search(r"\b(?:not provided|missing required|required argument)\b", output_text):
            return "tool reported a missing required argument"
        return None

    @staticmethod
    def _search_result_docids(output: Any) -> list[str]:
        if not isinstance(output, list):
            return []
        docids: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            docid = item.get("docid")
            if docid is None:
                continue
            docids.append(str(docid))
        return docids

    @staticmethod
    def _search_results_stagnated(
        previous_fingerprint: frozenset[str],
        current_fingerprint: frozenset[str],
    ) -> bool:
        if not previous_fingerprint or not current_fingerprint:
            return False
        intersection = previous_fingerprint & current_fingerprint
        union = previous_fingerprint | current_fingerprint
        if not union:
            return False
        overlap_ratio = len(intersection) / len(union)
        return overlap_ratio >= 0.8 and current_fingerprint.issubset(previous_fingerprint)

    def _force_final_response(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tool_iterations: int,
        stop_reason: str | None = None,
    ) -> tuple[str, int, int, str | None]:
        follow_up_messages = [dict(item) for item in messages]
        if stop_reason and stop_reason.startswith("search_results_stagnated:"):
            follow_up_messages.append(
                {
                    "role": "user",
                    "content": (
                        "The search results have stagnated and further search is unlikely to help. "
                        "Do not call more tools. If the currently gathered evidence is insufficient, "
                        "say so explicitly and return a concise insufficient-evidence or blocked-status answer."
                    ),
                }
            )
        elif stop_reason and stop_reason.startswith("tool_failure_circuit_breaker:"):
            follow_up_messages.append(
                {
                    "role": "user",
                    "content": (
                        "A tool failure circuit breaker has stopped further tool use. "
                        "Do not call more tools. If the currently gathered evidence is insufficient, "
                        "say so explicitly and return a concise blocked-status answer."
                    ),
                }
            )
        if stop_reason == "empty_reply":
            closing = "Your previous reply was empty. "
        else:
            closing = (
                "You have reached the maximum number of tool calls "
                f"({max(1, int(max_tool_iterations))}). "
            )
        follow_up_messages.append(
            {
                "role": "user",
                "content": (
                    closing
                    + "Based only on the information already gathered, provide your best final answer now. "
                    "Do not call any more tools."
                ),
            }
        )
        completion = self._create_chat_completion(
            model=model,
            messages=follow_up_messages,
            temperature=temperature,
        )
        text = self._extract_text(completion).strip()
        usage = getattr(completion, "usage", None)
        token_in = int(getattr(usage, "prompt_tokens", self._estimate_tokens(follow_up_messages)))
        token_out = int(getattr(usage, "completion_tokens", self._estimate_tokens(text)))
        return text, token_in, token_out, self._finish_reason(completion)

    @staticmethod
    def _finish_reason(completion: Any) -> str | None:
        choices = getattr(completion, "choices", None) or []
        if not choices:
            return None
        return getattr(choices[0], "finish_reason", None)

    @staticmethod
    def _is_output_limit_finish_reason(reason: str | None) -> bool:
        return str(reason or "").strip().lower() in {
            "length",
            "max_tokens",
            "max_completion_tokens",
            "content_length",
        }

    def _create_chat_completion(self, **kwargs: Any) -> Any:
        if self.client is None:
            raise RuntimeError("OpenRouter client unavailable")
        kwargs = self._apply_openrouter_sampling_overrides(
            kwargs,
            temperature=kwargs.get("temperature"),
        )
        kwargs.setdefault("timeout", self.config.timeout_s)
        attempts = max(1, self._env_int("MAS_LLM_RETRY_ATTEMPTS") or 3)
        empty_completion_attempts = max(
            1,
            self._env_int("MAS_LLM_EMPTY_COMPLETION_RETRY_ATTEMPTS") or attempts,
        )
        timeout_attempts = max(1, self._env_int("MAS_LLM_TIMEOUT_RETRY_ATTEMPTS") or 2)
        backoff_s = max(0.0, self._env_float("MAS_LLM_RETRY_BACKOFF_S") or 2.0)
        max_backoff_s = max(backoff_s, self._env_float("MAS_LLM_RETRY_MAX_BACKOFF_S") or 60.0)
        for attempt_index in range(attempts):
            request_kwargs = self._apply_openrouter_routing_overrides(
                kwargs,
                retry_index=attempt_index,
            )
            try:
                completion = self._create_chat_completion_once(**request_kwargs)
                if self._completion_has_usable_choice(completion):
                    return completion
                # Provider returned a 200 with no usable choice. On the final
                # attempt, hand the empty completion back so the caller records a
                # typed failure; otherwise retry — routing overrides rotate the
                # provider per attempt, so a different upstream often succeeds.
                if attempt_index >= min(attempts, empty_completion_attempts) - 1:
                    self._log(
                        "EMPTY_COMPLETION "
                        f"model={request_kwargs.get('model', '')} "
                        f"attempt={attempt_index + 1}/{min(attempts, empty_completion_attempts)} "
                        "returning_empty_after_retries"
                    )
                    return completion
                raise _EmptyCompletionError("provider returned an empty completion")
            except _EmptyCompletionError:
                delay_s = min(max_backoff_s, backoff_s * (2**attempt_index))
                jitter_factor = random.uniform(0.75, 1.35) if delay_s else 0.0
                sleep_s = delay_s * jitter_factor if delay_s else 0.0
                self._log(
                    "RETRY "
                    f"model={request_kwargs.get('model', '')} "
                    f"attempt={attempt_index + 1}/{attempts} delay_s={sleep_s:.2f} "
                    f"fallback_models={len(self._fallback_models_for_request(str(request_kwargs.get('model', ''))))} "
                    "reason=empty_completion"
                )
                if sleep_s:
                    time.sleep(sleep_s)
            except _HardTimeoutExpired as exc:
                if attempt_index >= min(attempts, timeout_attempts) - 1:
                    raise
                retry_after_s = self._retry_after_seconds(exc)
                delay_s = min(max_backoff_s, backoff_s * (2**attempt_index))
                if retry_after_s is not None:
                    delay_s = min(max_backoff_s, max(delay_s, retry_after_s))
                jitter_factor = random.uniform(0.75, 1.35) if delay_s else 0.0
                sleep_s = delay_s * jitter_factor if delay_s else 0.0
                self._log(
                    "RETRY "
                    f"model={request_kwargs.get('model', '')} "
                    f"attempt={attempt_index + 1}/{min(attempts, timeout_attempts)} "
                    f"delay_s={sleep_s:.2f} "
                    f"fallback_models={len(self._fallback_models_for_request(str(request_kwargs.get('model', ''))))} "
                    f"error={type(exc).__name__}:{exc}"
                )
                if sleep_s:
                    time.sleep(sleep_s)
            except Exception as exc:
                if attempt_index >= attempts - 1 or not self._is_retryable_llm_error(exc):
                    raise
                retry_after_s = self._retry_after_seconds(exc)
                delay_s = min(max_backoff_s, backoff_s * (2**attempt_index))
                if retry_after_s is not None:
                    delay_s = min(max_backoff_s, max(delay_s, retry_after_s))
                jitter_factor = random.uniform(0.75, 1.35) if delay_s else 0.0
                sleep_s = delay_s * jitter_factor if delay_s else 0.0
                self._log(
                    "RETRY "
                    f"model={request_kwargs.get('model', '')} "
                    f"attempt={attempt_index + 1}/{attempts} "
                    f"delay_s={sleep_s:.2f} "
                    f"fallback_models={len(self._fallback_models_for_request(str(request_kwargs.get('model', ''))))} "
                    f"error={type(exc).__name__}:{exc}"
                )
                if sleep_s:
                    time.sleep(sleep_s)
        raise RuntimeError("unreachable LLM retry loop state")

    def _create_chat_completion_once(self, **kwargs: Any) -> Any:
        timeout_value = float(self.config.timeout_s or 0.0)
        # On the main thread we can use SIGALRM for a true wall-clock deadline.
        # On worker threads (LangGraph Send branches) SIGALRM is unavailable, so
        # we run the HTTP call in a disposable thread and join with a timeout.
        # This ensures every API call is bounded regardless of which thread calls it.
        if timeout_value > 0.0 and threading.current_thread() is not threading.main_thread():
            # Do NOT use the context-manager form of ThreadPoolExecutor here.
            # Its __exit__ calls shutdown(wait=True), which would block forever
            # waiting for the background API thread even after a timeout fires.
            _ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            _future = _ex.submit(self.client.chat.completions.create, **kwargs)
            try:
                result = _future.result(timeout=timeout_value)
                _ex.shutdown(wait=False)
                return result
            except concurrent.futures.TimeoutError as exc:
                _ex.shutdown(wait=False)
                self._log(
                    "TIMEOUT "
                    f"mode=worker_thread timeout_s={timeout_value:.3f} "
                    f"model={kwargs.get('model', '')}"
                )
                raise _HardTimeoutExpired(
                    f"LLM request exceeded configured timeout of {timeout_value:.3f}s"
                ) from exc
            except Exception:
                _ex.shutdown(wait=False)
                raise
        with self._hard_timeout(self.config.timeout_s):
            return self.client.chat.completions.create(**kwargs)

    @staticmethod
    def _is_retryable_llm_error(exc: Exception) -> bool:
        text = f"{type(exc).__name__}:{exc}".lower()
        return any(
            marker in text
            for marker in (
                "429",
                "rate limit",
                "rate-limit",
                "temporarily rate-limited",
                "connection error",
                "connectionerror",
                "jsondecodeerror",
                "expecting value",
                "server disconnected",
                "temporarily unavailable",
            )
        )

    @staticmethod
    def _retry_after_seconds(exc: Exception) -> float | None:
        text = f"{type(exc).__name__}:{exc}"
        patterns = (
            r"retry_after_seconds['\"]?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
            r"retry-after['\"]?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
            r"retry after\s+([0-9]+(?:\.[0-9]+)?)",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                try:
                    return max(0.0, float(match.group(1)))
                except Exception:
                    return None
        return None

    @contextmanager
    def _hard_timeout(self, timeout_s: float | None):
        timeout_value = float(timeout_s or 0.0)
        signal_supported = (
            timeout_value > 0.0
            and os.name != "nt"
            and hasattr(signal, "SIGALRM")
            and hasattr(signal, "setitimer")
            and hasattr(signal, "ITIMER_REAL")
            and threading.current_thread() is threading.main_thread()
        )
        if not signal_supported:
            yield
            return

        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_delay, previous_interval = signal.setitimer(signal.ITIMER_REAL, 0.0)

        def _handle_timeout(signum: int, frame: Any) -> None:
            self._log(f"TIMEOUT mode=main_thread timeout_s={timeout_value:.3f}")
            raise _HardTimeoutExpired(
                f"LLM request exceeded configured timeout of {timeout_value:.3f}s"
            )

        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, timeout_value)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)
            if previous_delay > 0.0 or previous_interval > 0.0:
                signal.setitimer(signal.ITIMER_REAL, previous_delay, previous_interval)

    @staticmethod
    def _sanitize_tool_name(name: str, used_names: set[str]) -> str:
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name).strip("_")
        if not sanitized:
            sanitized = "tool"
        candidate = sanitized
        suffix = 2
        while candidate in used_names:
            candidate = f"{sanitized}_{suffix}"
            suffix += 1
        used_names.add(candidate)
        return candidate

    @staticmethod
    def _resolve_emitted_tool_name(
        name: str, handlers: dict[str, Callable[[dict[str, Any]], Any]]
    ) -> str:
        """Recover a known tool when a provider appends harmless punctuation."""

        if name in handlers:
            return name
        stripped = re.sub(r"[^a-zA-Z0-9_-]+$", "", str(name or "").strip())
        if stripped in handlers:
            return stripped
        return name

    @staticmethod
    def _has_non_query_search_argument(args: dict[str, Any]) -> bool:
        """Return whether a search call sets a filter other than its free-text query."""

        return any(
            key != "query" and value is not None and str(value).strip()
            for key, value in args.items()
        )

    @staticmethod
    def _normalize_tool_arguments(args: dict[str, Any]) -> dict[str, Any]:
        """Repair harmless provider quoting around argument keys and identifiers."""

        normalized: dict[str, Any] = {}
        for key, value in args.items():
            original = str(key).strip()
            repaired = original
            while (
                len(repaired) >= 2
                and repaired[0] == repaired[-1]
                and repaired[0] in {'"', "'", "`"}
            ):
                repaired = repaired[1:-1].strip()
            normalized_value = value
            identifier_key = repaired.lower() in {"id", "docid"} or repaired.lower().endswith(
                "_id"
            )
            if identifier_key and isinstance(value, str):
                normalized_value = value.strip()
                while (
                    len(normalized_value) >= 2
                    and normalized_value[0] == normalized_value[-1]
                    and normalized_value[0] in {'"', "'", "`"}
                ):
                    normalized_value = normalized_value[1:-1].strip()
            if repaired not in normalized or repaired == original:
                normalized[repaired] = normalized_value
        return normalized

    @staticmethod
    def _completion_has_usable_choice(completion: Any) -> bool:
        """True if the completion carries a usable assistant turn.

        A tool-call turn legitimately has empty ``content``, so it counts as
        usable. "Unusable" means no choices, no message, or a message with
        neither text content nor tool calls — i.e. the provider gave us nothing
        to act on (an infrastructure failure, not a deliberate empty answer).
        """

        choices = getattr(completion, "choices", None)
        if not choices:
            return False
        message = getattr(choices[0], "message", None)
        if message is None:
            return False
        if getattr(message, "tool_calls", None):
            return True
        content = getattr(message, "content", "")
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
            content = " ".join(parts)
        return bool(str(content or "").strip())

    @staticmethod
    def _extract_text(completion: Any) -> str:
        choices = getattr(completion, "choices", [])
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        if message is None:
            return ""
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                else:
                    text = getattr(item, "text", None)
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts)
        return str(content)

    @staticmethod
    def _estimate_tokens(text: Any) -> int:
        if isinstance(text, list):
            content = " ".join(str(m.get("content", "")) for m in text)
            if not content.strip():
                return 0
            return max(1, int(len(re.findall(r"\S+", content)) * 1.3))

        if not text or not str(text).strip():
            return 0
        return max(1, int(len(re.findall(r"\S+", str(text))) * 1.3))

    @staticmethod
    def _safe_json_dumps(value: Any) -> str:
        return json.dumps(OpenRouterLLMClient._to_jsonable(value), ensure_ascii=False)

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): OpenRouterLLMClient._to_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [OpenRouterLLMClient._to_jsonable(item) for item in value]

        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            try:
                return OpenRouterLLMClient._to_jsonable(to_dict())
            except Exception:
                pass

        to_list = getattr(value, "tolist", None)
        if callable(to_list):
            try:
                return OpenRouterLLMClient._to_jsonable(to_list())
            except Exception:
                pass

        item = getattr(value, "item", None)
        if callable(item):
            try:
                return OpenRouterLLMClient._to_jsonable(item())
            except Exception:
                pass

        return str(value)

    @staticmethod
    def _stable_seed(*parts: str) -> int:
        data = "||".join(parts).encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        return int(digest[:16], 16)

    def _next_request_id(self) -> str:
        with self._request_seq_lock:
            self._request_seq += 1
            return f"llm-{self._request_seq}"

    @staticmethod
    def _truncate_for_log(value: Any, *, limit: int = 240) -> str:
        text = OpenRouterLLMClient._safe_json_dumps(value)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    @staticmethod
    def _preview_text(text: Any, *, limit: int = 320) -> str:
        if not text:
            return ""
        cleaned = re.sub(r"\s+", " ", str(text)).strip()
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: limit - 3] + "..."

    @staticmethod
    def _log(message: str) -> None:
        print(f"[llm] {message}", flush=True)
