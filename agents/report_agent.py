"""Executive Report Agent implementation."""

from __future__ import annotations

from typing import Any

from .base_agent import BaseAgent
from .domain_agent import DomainMappingAgent


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
        domain_payload = {
            "summary": payload.get("summary", {}),
            "interpretation": payload.get("interpretation", {}),
            "bio_insights": payload.get("bio_insights", {}),
        }
        domain_mapping = self.domain_agent.run(domain_payload)

        final_payload = {
            "summary": payload.get("summary", {}),
            "model": payload.get("model", {}),
            "interpretation": payload.get("interpretation", {}),
            "domain_mapping": domain_mapping,
            "bio_insights": payload.get("bio_insights", {}),
        }

        if not self.has_openai_key():
            return self._mock(final_payload, domain_mapping)

        try:
            return self._call_openai_json(
                (
                    "You are an executive bio-AI report agent. Return valid JSON only with keys: "
                    "agent, mode, report_title, executive_summary, report."
                ),
                final_payload,
            )
        except Exception:
            return self._mock(final_payload, domain_mapping)
