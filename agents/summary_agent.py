"""Data Summary Agent implementation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .base_agent import BaseAgent


class SummaryInputSchema(BaseModel):
    """Validated input schema for DataSummaryAgent."""

    n_rows: int
    n_columns: int
    numeric_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)
    missing_by_column: dict[str, int] = Field(default_factory=dict)
    target_column: str | None = None


class SummaryOutputSchema(BaseModel):
    """Validated output schema for DataSummaryAgent."""

    agent: str
    mode: str
    summary_text: str
    highlights: dict[str, Any]


class DataSummaryAgent(BaseAgent):
    """Generate concise dataset summary from validation metadata."""

    @property
    def name(self) -> str:
        return "data_summary_agent"

    def _mock(self, payload: dict[str, Any]) -> dict[str, Any]:
        missing_total = int(sum(payload.get("missing_by_column", {}).values()))
        return {
            "agent": self.name,
            "mode": "mock",
            "summary_text": (
                f"Dataset has {payload.get('n_rows', 0)} rows and {payload.get('n_columns', 0)} columns. "
                f"Detected {len(payload.get('numeric_columns', []))} numeric and "
                f"{len(payload.get('categorical_columns', []))} categorical features."
            ),
            "highlights": {
                "target_column": payload.get("target_column"),
                "missing_total": missing_total,
                "missing_by_column": payload.get("missing_by_column", {}),
            },
        }

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Produce structured summary JSON."""
        validated_input = self._validate_input(SummaryInputSchema, payload)

        used_openai = False
        if self.has_openai_key():
            try:
                used_openai = True
                output = self._call_openai_json(
                    (
                        "You are a data summary agent. Return valid JSON only with keys: "
                        "agent, mode, summary_text, highlights."
                    ),
                    validated_input,
                )
            except Exception:
                used_openai = False
                output = self._mock(validated_input)
        else:
            output = self._mock(validated_input)

        validated_output = self._validate_output(SummaryOutputSchema, output)
        self._record_trace(validated_input, validated_output, used_openai)
        return validated_output
