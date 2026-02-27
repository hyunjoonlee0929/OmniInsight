"""Base class for OmniInsight LLM agents."""

from __future__ import annotations

import json
import os
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Any, Type

from pydantic import BaseModel


class BaseAgent(ABC):
    """Abstract JSON-in, JSON-out contract for all agents."""

    last_trace: dict[str, Any] | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name used for tracing and metadata."""

    @abstractmethod
    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run agent logic with structured JSON payload."""

    def has_openai_key(self) -> bool:
        """Return True when OPENAI_API_KEY is configured."""
        return bool(os.getenv("OPENAI_API_KEY"))

    def _call_openai_json(self, system_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Call OpenAI Responses API and parse JSON output."""
        from openai import OpenAI

        client = OpenAI()
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=f"{system_prompt}\nInput JSON: {json.dumps(payload)}",
            temperature=0,
        )
        text = response.output_text.strip()
        return json.loads(text)

    def _model_validate(self, schema: Type[BaseModel], payload: dict[str, Any]) -> BaseModel:
        if hasattr(schema, "model_validate"):
            return schema.model_validate(payload)
        return schema.parse_obj(payload)

    def _model_dump(self, model: BaseModel) -> dict[str, Any]:
        if hasattr(model, "model_dump"):
            return model.model_dump()
        return model.dict()

    def _validate_input(self, schema: Type[BaseModel], payload: dict[str, Any]) -> dict[str, Any]:
        validated = self._model_validate(schema, payload)
        return self._model_dump(validated)

    def _validate_output(self, schema: Type[BaseModel], payload: dict[str, Any]) -> dict[str, Any]:
        validated = self._model_validate(schema, payload)
        return self._model_dump(validated)

    def _record_trace(
        self,
        validated_input: dict[str, Any],
        validated_output: dict[str, Any],
        used_openai: bool,
    ) -> None:
        self.last_trace = {
            "agent": self.name,
            "timestamp": datetime.now().isoformat(),
            "used_openai": used_openai,
            "validated_input": validated_input,
            "validated_output": validated_output,
        }
