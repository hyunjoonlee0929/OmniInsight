"""Feature Interpretation Agent implementation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .base_agent import BaseAgent


class InterpretationInputSchema(BaseModel):
    """Validated input schema for FeatureInterpretationAgent."""

    status: str
    top_features: list[str] = Field(default_factory=list)
    feature_importance: dict[str, float] = Field(default_factory=dict)
    explainer: str | None = None
    error: str | None = None


class InterpretationOutputSchema(BaseModel):
    """Validated output schema for FeatureInterpretationAgent."""

    agent: str
    mode: str
    top_features: list[str]
    interpretation: dict[str, Any]


class FeatureInterpretationAgent(BaseAgent):
    """Translate raw importance scores into structured interpretation."""

    @property
    def name(self) -> str:
        return "feature_interpretation_agent"

    def _mock(self, payload: dict[str, Any]) -> dict[str, Any]:
        top_features = payload.get("top_features", [])
        importance = payload.get("feature_importance", {})
        narratives = [f"{feat} has influence score {importance.get(feat, 0.0):.4f}" for feat in top_features[:5]]
        return {
            "agent": self.name,
            "mode": "mock",
            "top_features": top_features,
            "interpretation": {
                "shap_status": payload.get("status", "unknown"),
                "explainer": payload.get("explainer", "unknown"),
                "feature_importance": importance,
                "narrative": narratives,
            },
        }

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Produce interpretation JSON from SHAP payload."""
        validated_input = self._validate_input(InterpretationInputSchema, payload)

        used_openai = False
        if self.has_openai_key():
            try:
                used_openai = True
                output = self._call_openai_json(
                    (
                        "You are a feature interpretation agent. Return valid JSON only with keys: "
                        "agent, mode, top_features, interpretation."
                    ),
                    validated_input,
                )
            except Exception:
                used_openai = False
                output = self._mock(validated_input)
        else:
            output = self._mock(validated_input)

        validated_output = self._validate_output(InterpretationOutputSchema, output)
        self._record_trace(validated_input, validated_output, used_openai)
        return validated_output
