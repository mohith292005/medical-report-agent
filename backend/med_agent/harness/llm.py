"""Thin OpenRouter LLM client using the official OpenAI Python SDK.

Features:
- Connects to OpenRouter (base_url="https://openrouter.ai/api/v1").
- Passes required and optional headers (HTTP-Referer, X-Title).
- Implements exponential backoff retry for HTTP 429 (Rate Limit) and 5xx (Server Errors).
- Supports standard tool calling (`tools=[...]`) and structured output (`response_format={"type": "json_object"}`).
- Returns a standardized LLMResponse dataclass with the assistant message and token usage.
- Supports pluggable mock handlers for unit testing and offline development without an API key.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from med_agent.harness.config import AgentConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Token consumption tally for a single LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResponse:
    """Standardized response container returned by the LLM wrapper."""
    message: Dict[str, Any]  # Dictionary with role, content, and optional tool_calls
    usage: TokenUsage
    finish_reason: str = "stop"


# Type for custom mock/simulator function used during testing:
# Callable[[messages, tools, json_mode], LLMResponse]
MockHandler = Callable[[List[Dict[str, Any]], Optional[List[Dict[str, Any]]], bool], LLMResponse]


class OpenRouterLLM:
    """Wrapper around OpenAI SDK targeting the OpenRouter API."""

    def __init__(self, config: Optional[AgentConfig] = None, mock_handler: Optional[MockHandler] = None):
        self.config = config or DEFAULT_CONFIG
        self.mock_handler = mock_handler
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        """Lazily initialize the OpenAI client with OpenRouter configuration."""
        if self._client is None:
            extra_headers = {}
            if self.config.http_referer:
                extra_headers["HTTP-Referer"] = self.config.http_referer
            if self.config.title:
                extra_headers["X-Title"] = self.config.title

            self._client = OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key or "no_key_provided",
                default_headers=extra_headers,
            )
        return self._client

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        json_mode: bool = False,
        temperature: float = 0.1,
        max_retries: int = 3,
        initial_backoff: float = 1.5,
    ) -> LLMResponse:
        """Sends a chat completion request with exponential backoff on 429/5xx.
        
        Args:
            messages: List of OpenAI-format message dicts (role, content, etc.).
            tools: Optional list of function tool schemas.
            json_mode: If True, enforces valid JSON output from the model.
            temperature: Sampling temperature (low for deterministic analysis).
            max_retries: Number of retry attempts on transient errors.
            initial_backoff: Initial seconds for exponential backoff.
            
        Returns:
            LLMResponse containing message dict and token usage.
        """
        # If a mock handler is provided (e.g. during unit tests or simulated evals), use it
        if self.mock_handler is not None:
            return self.mock_handler(messages, tools, json_mode)

        client = self._get_client()
        params: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_output_tokens,
            "temperature": temperature,
        }

        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        if json_mode:
            params["response_format"] = {"type": "json_object"}

        attempt = 0
        backoff = initial_backoff

        while attempt < max_retries:
            try:
                response = client.chat.completions.create(**params)
                
                # Extract first choice
                choice = response.choices[0]
                assistant_msg = choice.message

                # Convert message object to standard dict
                msg_dict: Dict[str, Any] = {
                    "role": "assistant",
                    "content": assistant_msg.content,
                }
                
                if assistant_msg.tool_calls:
                    # Convert tool calls to standard dicts
                    msg_dict["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in assistant_msg.tool_calls
                    ]

                # Extract token usage
                usage = TokenUsage()
                if response.usage:
                    usage.prompt_tokens = response.usage.prompt_tokens or 0
                    usage.completion_tokens = response.usage.completion_tokens or 0
                    usage.total_tokens = response.usage.total_tokens or (usage.prompt_tokens + usage.completion_tokens)

                return LLMResponse(
                    message=msg_dict,
                    usage=usage,
                    finish_reason=choice.finish_reason or "stop",
                )

            except (RateLimitError, APIStatusError, APIConnectionError) as err:
                status_code = getattr(err, "status_code", None)
                # Check if retryable (429 or 5xx server errors)
                is_retryable = (
                    isinstance(err, (RateLimitError, APIConnectionError))
                    or (status_code is not None and (status_code == 429 or status_code >= 500))
                )

                if is_retryable and attempt < max_retries - 1:
                    attempt += 1
                    logger.warning(
                        "LLM call transient error (%s). Retrying in %.1fs (attempt %d/%d)...",
                        err,
                        backoff,
                        attempt,
                        max_retries,
                    )
                    time.sleep(backoff)
                    backoff *= 2.0
                else:
                    logger.error("LLM call failed permanently: %s", err)
                    raise
            except Exception as e:
                logger.error("Non-retryable LLM call error: %s", e)
                raise
