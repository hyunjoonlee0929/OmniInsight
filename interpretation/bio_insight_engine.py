"""Biological insight abstraction from feature and pathway importance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json


class BioInsightEngine:
    """Generate pathway and target-level biological insights."""

    def load_pathway_mapping(self, mapping_path: Path) -> dict[str, str]:
        """Load deterministic feature-to-pathway mapping JSON."""
        with mapping_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, dict):
            raise ValueError("pathway_mapping.json must be a JSON object.")

        return {str(k): str(v) for k, v in raw.items()}

    def _normalize_feature_name(self, feature_name: str) -> str:
        """Normalize transformed feature names to original feature IDs."""
        name = feature_name
        if name.startswith("num__"):
            name = name[5:]
        elif name.startswith("cat__"):
            name = name[5:]
        return name

    def aggregate_pathway_scores(
        self,
        feature_importance: dict[str, float],
        pathway_mapping: dict[str, str],
    ) -> dict[str, float]:
        """Aggregate feature importances into pathway-level scores."""
        pathway_scores: dict[str, float] = {}

        for feature, score in feature_importance.items():
            normalized = self._normalize_feature_name(feature)
            pathway = pathway_mapping.get(normalized, pathway_mapping.get(feature, "Unmapped"))
            pathway_scores[pathway] = pathway_scores.get(pathway, 0.0) + float(score)

        return dict(sorted(pathway_scores.items(), key=lambda x: x[1], reverse=True))

    def generate_bio_insights(
        self,
        feature_importance: dict[str, float],
        pathway_scores: dict[str, float],
        feature_blocks: dict[str, list[str]],
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Generate biological interpretation outputs for reporting and agents."""
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        sorted_pathways = sorted(pathway_scores.items(), key=lambda x: x[1], reverse=True)

        top_regulatory_genes = [
            {"gene": f, "importance": float(s)}
            for f, s in sorted_features
            if f.startswith("num__tx__") or f.startswith("tx__") or "tx__" in f
        ][:top_k]

        dominant_pathways = [{"pathway": p, "score": float(s)} for p, s in sorted_pathways[:top_k]]

        candidate_targets = []
        for item in top_regulatory_genes[:top_k]:
            gene = item["gene"]
            score = item["importance"]
            candidate_targets.append(
                {
                    "target": gene,
                    "rationale": f"High model influence score ({score:.4f}) suggests regulatory leverage.",
                    "priority": "high" if score > 0.1 else "medium",
                }
            )

        if not candidate_targets:
            for f, s in sorted_features[:top_k]:
                candidate_targets.append(
                    {
                        "target": f,
                        "rationale": f"Top predictive feature with score {s:.4f}.",
                        "priority": "high" if s > 0.1 else "medium",
                    }
                )

        hypotheses = []
        if dominant_pathways:
            top_path = dominant_pathways[0]["pathway"]
            hypotheses.append(
                f"Modulating genes linked to {top_path} may produce measurable phenotype shifts."
            )
        if candidate_targets:
            hypotheses.append(
                f"Intervening on {candidate_targets[0]['target']} is predicted to impact model outcome strongly."
            )
        if feature_blocks:
            hypotheses.append(
                "Cross-omics signals indicate multi-layer regulation rather than a single-modality effect."
            )

        return {
            "top_regulatory_genes": top_regulatory_genes,
            "dominant_pathways": dominant_pathways,
            "candidate_engineering_targets": candidate_targets,
            "hypotheses": hypotheses,
            "pathway_scores": pathway_scores,
            "feature_blocks": feature_blocks,
        }
