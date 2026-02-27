"""Domain Mapping Agent implementation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .base_agent import BaseAgent


class DomainInputSchema(BaseModel):
    """Validated input schema for DomainMappingAgent."""

    summary: dict[str, Any] = Field(default_factory=dict)
    interpretation: dict[str, Any] = Field(default_factory=dict)
    bio_insights: dict[str, Any] = Field(default_factory=dict)


class DomainOutputSchema(BaseModel):
    """Validated output schema for DomainMappingAgent."""

    agent: str
    mode: str
    mapped_concepts: list[dict[str, Any]]
    pathway_interpretation: list[dict[str, Any]]
    candidate_engineering_targets: list[dict[str, Any]]
    hypotheses: list[str]
    caveats: list[str]


class DomainMappingAgent(BaseAgent):
    """Map model insights into domain-specific meaning."""

    @property
    def name(self) -> str:
        return "domain_mapping_agent"

    def _mock(self, payload: dict[str, Any]) -> dict[str, Any]:
        interpretation = payload.get("interpretation", {})
        top_features = interpretation.get("top_features", [])
        bio_insights = payload.get("bio_insights", {})

        mapped = [
            {
                "feature": f,
                "domain_concept": f"Potential domain driver: {f}",
                "confidence": 0.5,
            }
            for f in top_features[:5]
        ]

        dominant_pathways = bio_insights.get("dominant_pathways", [])
        pathway_interpretation = [
            {
                "pathway": item.get("pathway"),
                "interpretation": f"Pathway {item.get('pathway')} shows elevated aggregate importance.",
            }
            for item in dominant_pathways[:5]
        ]

        candidate_targets = bio_insights.get("candidate_engineering_targets", [])
        hypotheses = bio_insights.get("hypotheses", [])

        return {
            "agent": self.name,
            "mode": "mock",
            "mapped_concepts": mapped,
            "pathway_interpretation": pathway_interpretation,
            "candidate_engineering_targets": candidate_targets,
            "hypotheses": hypotheses,
            "caveats": [
                "Domain mappings are deterministic abstractions unless ontology knowledge is connected.",
            ],
        }

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Produce domain mapping JSON."""
        validated_input = self._validate_input(DomainInputSchema, payload)

        used_openai = False
        if self.has_openai_key():
            try:
                used_openai = True
                output = self._call_openai_json(
                    (
                        "You are a domain mapping agent for multi-omics AI. Return valid JSON only with keys: "
                        "agent, mode, mapped_concepts, pathway_interpretation, candidate_engineering_targets, "
                        "hypotheses, caveats."
                    ),
                    validated_input,
                )
            except Exception:
                used_openai = False
                output = self._mock(validated_input)
        else:
            output = self._mock(validated_input)

        validated_output = self._validate_output(DomainOutputSchema, output)
        self._record_trace(validated_input, validated_output, used_openai)
        return validated_output
