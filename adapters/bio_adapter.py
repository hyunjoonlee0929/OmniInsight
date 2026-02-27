"""Biology-focused adapter with multi-omics integration and insight abstraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from OmniInsight.interpretation.bio_insight_engine import BioInsightEngine

from .general_adapter import GeneralAdapter, GeneralAdapterConfig

logger = logging.getLogger(__name__)


@dataclass
class BioAdapterConfig(GeneralAdapterConfig):
    """Extended config for omics workflows."""

    sample_id_column: str = "sample_id"
    transcriptomics_df: pd.DataFrame | None = None
    proteomics_df: pd.DataFrame | None = None
    metabolomics_df: pd.DataFrame | None = None
    pathway_mapping_path: str = "OmniInsight/config/pathway_mapping.json"


class BioAdapter(GeneralAdapter):
    """Adapter extending the general flow for multi-omics datasets."""

    def __init__(self) -> None:
        super().__init__()
        self.bio_engine = BioInsightEngine()

    def _prefix_block(self, df: pd.DataFrame, block_prefix: str, sample_id_column: str) -> pd.DataFrame:
        """Prefix omics columns while preserving sample ID."""
        renamed = {}
        for c in df.columns:
            if c == sample_id_column:
                continue
            renamed[c] = f"{block_prefix}{c}"
        return df.rename(columns=renamed)

    def _combine_blocks(self, base_df: pd.DataFrame, config: BioAdapterConfig) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        """Merge available omics blocks by sample ID and return feature block index."""
        sid = config.sample_id_column
        if sid not in base_df.columns:
            raise ValueError(f"Base dataframe must include sample ID column '{sid}'.")

        merged = base_df.copy()
        feature_blocks: dict[str, list[str]] = {
            "clinical": [c for c in merged.columns if c not in {sid, config.target_column}],
            "transcriptomics": [],
            "proteomics": [],
            "metabolomics": [],
        }

        block_specs = [
            ("transcriptomics", "tx__", config.transcriptomics_df),
            ("proteomics", "pr__", config.proteomics_df),
            ("metabolomics", "mt__", config.metabolomics_df),
        ]

        for block_name, prefix, block_df in block_specs:
            if block_df is None:
                continue
            if sid not in block_df.columns:
                raise ValueError(f"{block_name} dataframe must include sample ID column '{sid}'.")

            block_prefixed = self._prefix_block(block_df.copy(), prefix, sid)
            block_cols = [c for c in block_prefixed.columns if c != sid]
            merged = merged.merge(block_prefixed, on=sid, how="inner")
            feature_blocks[block_name] = block_cols

        return merged, feature_blocks

    def run_with_details(self, df: pd.DataFrame, config: BioAdapterConfig) -> dict[str, Any]:
        """Run full workflow with optional multi-modal input integration."""
        merged_df, feature_blocks = self._combine_blocks(df, config)

        sid = config.sample_id_column
        if sid in merged_df.columns and sid != config.target_column:
            training_df = merged_df.drop(columns=[sid])
        else:
            training_df = merged_df

        details = super().run_with_details(df=training_df, config=config)
        report = details["report"]

        shap_payload = details.get("shap_payload", {})
        feature_importance = shap_payload.get("feature_importance", {})

        pathway_mapping = self.bio_engine.load_pathway_mapping(Path(config.pathway_mapping_path))
        pathway_scores = self.bio_engine.aggregate_pathway_scores(
            feature_importance=feature_importance,
            pathway_mapping=pathway_mapping,
        )

        bio_insights = self.bio_engine.generate_bio_insights(
            feature_importance=feature_importance,
            pathway_scores=pathway_scores,
            feature_blocks=feature_blocks,
            top_k=config.top_k_features,
        )

        payload = details.get("payload", {}).copy()
        payload["bio_insights"] = bio_insights
        report = self.generate_report(payload)

        details["report"] = report
        details["payload"] = payload
        details["bio_insights"] = bio_insights
        details["pathway_scores"] = pathway_scores
        details["feature_blocks"] = feature_blocks
        details["merged_rows"] = len(merged_df)
        details["agent_traces"] = {
            "summary_agent": self.summary_agent.last_trace,
            "interpretation_agent": self.interpretation_agent.last_trace,
            "domain_mapping_agent": self.report_agent.domain_agent.last_trace,
            "executive_report_agent": self.report_agent.last_trace,
        }

        return details

    def run(self, df: pd.DataFrame, config: BioAdapterConfig) -> dict[str, Any]:
        """Run full workflow with optional multi-modal input integration."""
        return self.run_with_details(df=df, config=config)["report"]
