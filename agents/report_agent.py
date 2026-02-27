"""Executive Report Agent implementation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .base_agent import BaseAgent
from .domain_agent import DomainMappingAgent


class ReportInputSchema(BaseModel):
    """Validated input schema for ExecutiveReportAgent."""

    summary: dict[str, Any] = Field(default_factory=dict)
    model: dict[str, Any] = Field(default_factory=dict)
    interpretation: dict[str, Any] = Field(default_factory=dict)
    bio_insights: dict[str, Any] = Field(default_factory=dict)


class ReportOutputSchema(BaseModel):
    """Validated output schema for ExecutiveReportAgent."""

    agent: str
    mode: str
    report_title: str
    executive_summary: str
    report: dict[str, Any]


class ExecutiveReportAgent(BaseAgent):
    """Create final executive report from intermediate agent outputs."""

    def __init__(self) -> None:
        self.domain_agent = DomainMappingAgent()

    @property
    def name(self) -> str:
        return "executive_report_agent"

    def _mock(self, payload: dict[str, Any], domain_mapping: dict[str, Any]) -> dict[str, Any]:
        model_payload = payload.get("model", {})
        metrics = model_payload.get("metrics", {})
        metric_text = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))])

        bio_insights = payload.get("bio_insights", {})

        report = {
            "summary": payload.get("summary", {}),
            "model": model_payload,
            "interpretation": payload.get("interpretation", {}),
            "domain_mapping": domain_mapping,
            "top_features": payload.get("interpretation", {}).get("top_features", []),
            "pathway_scores": bio_insights.get("pathway_scores", {}),
            "pathway_interpretation": domain_mapping.get("pathway_interpretation", []),
            "bioengineering_targets": domain_mapping.get("candidate_engineering_targets", []),
            "hypotheses": domain_mapping.get("hypotheses", []),
            "bio_insights": bio_insights,
        }

        return {
            "agent": self.name,
            "mode": "mock",
            "report_title": "OmniInsight Executive Report",
            "executive_summary": f"Pipeline completed successfully. Key metrics: {metric_text}",
            "report": report,
        }

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Assemble final JSON report."""
        validated_input = self._validate_input(ReportInputSchema, payload)

        domain_payload = {
            "summary": validated_input.get("summary", {}),
            "interpretation": validated_input.get("interpretation", {}),
            "bio_insights": validated_input.get("bio_insights", {}),
        }
        domain_mapping = self.domain_agent.run(domain_payload)

        final_payload = {
            "summary": validated_input.get("summary", {}),
            "model": validated_input.get("model", {}),
            "interpretation": validated_input.get("interpretation", {}),
            "domain_mapping": domain_mapping,
            "bio_insights": validated_input.get("bio_insights", {}),
        }

        used_openai = False
        if self.has_openai_key():
            try:
                used_openai = True
                output = self._call_openai_json(
                    (
                        "You are an executive bio-AI report agent. Return valid JSON only with keys: "
                        "agent, mode, report_title, executive_summary, report."
                    ),
                    final_payload,
                )
            except Exception:
                used_openai = False
                output = self._mock(final_payload, domain_mapping)
        else:
            output = self._mock(final_payload, domain_mapping)

        validated_output = self._validate_output(ReportOutputSchema, output)
        self._record_trace(validated_input, validated_output, used_openai)
        return validated_output
