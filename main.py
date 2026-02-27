"""Entrypoint for running the full OmniInsight pipeline with experiment tracking."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from OmniInsight.adapters.bio_adapter import BioAdapter, BioAdapterConfig
from OmniInsight.adapters.general_adapter import GeneralAdapter, GeneralAdapterConfig
from OmniInsight.core.experiment_tracker import ExperimentTracker
from OmniInsight.core.logging_utils import configure_logging

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """Set global random seed for reproducibility across libraries."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run OmniInsight end-to-end pipeline")
    parser.add_argument("--data", type=Path, default=Path("OmniInsight/data/example_dataset.csv"), help="Path to input CSV")
    parser.add_argument("--protein-data", type=Path, default=None, help="Optional protein CSV for bio adapter mode")
    parser.add_argument("--config", type=Path, default=Path("OmniInsight/config/model_config.yaml"), help="Path to model config")
    parser.add_argument("--task-type", choices=["classification", "regression"], default=None, help="Override task type")
    parser.add_argument("--model-type", choices=["xgboost", "dnn"], default=None, help="Override model type")
    parser.add_argument("--target-column", type=str, default=None, help="Override target column")
    parser.add_argument("--top-k", type=int, default=None, help="Number of top SHAP features")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--from-run", type=str, default=None, help="Re-run using saved config from runs/{run_id}")
    parser.add_argument("--use-bio-adapter", action="store_true", help="Use BioAdapter for multi-omics mode")
    parser.add_argument("--output", type=Path, default=None, help="Optional output path for JSON report")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def _build_general_config(config_raw: dict[str, Any], args: argparse.Namespace) -> GeneralAdapterConfig:
    default_cfg = config_raw["default"]
    dnn_cfg = config_raw.get("dnn", {})

    return GeneralAdapterConfig(
        target_column=args.target_column or default_cfg["target_column"],
        task_type=args.task_type or default_cfg["task_type"],
        model_type=args.model_type or default_cfg["model_type"],
        top_k_features=args.top_k or default_cfg.get("top_k_features", 10),
        dnn_hidden_layers=dnn_cfg.get("hidden_layers", [128, 64, 32]),
        dnn_dropout=float(dnn_cfg.get("dropout", 0.2)),
        dnn_learning_rate=float(dnn_cfg.get("learning_rate", 1e-3)),
        dnn_max_epochs=int(dnn_cfg.get("max_epochs", 200)),
        dnn_patience=int(dnn_cfg.get("patience", 20)),
        dnn_batch_size=int(dnn_cfg.get("batch_size", 32)),
        random_seed=int(args.seed),
    )


def _apply_from_run(args: argparse.Namespace, tracker: ExperimentTracker) -> argparse.Namespace:
    """Load run snapshot and mutate args for reproducible reruns."""
    if not args.from_run:
        return args

    snapshot = tracker.load_run_snapshot(args.from_run)
    run_input = snapshot.get("input", {})
    run_cfg = snapshot.get("adapter_config", {})
    args.loaded_adapter_config = run_cfg

    args.data = Path(run_input["data_path"])
    protein_path = run_input.get("protein_data_path")
    args.protein_data = Path(protein_path) if protein_path else None
    args.use_bio_adapter = bool(run_input.get("use_bio_adapter", False))

    args.config = Path(snapshot["config_path"])
    args.seed = int(snapshot.get("seed", run_cfg.get("random_seed", 42)))

    args.task_type = run_cfg.get("task_type")
    args.model_type = run_cfg.get("model_type")
    args.target_column = run_cfg.get("target_column")
    args.top_k = int(run_cfg.get("top_k_features", 10))

    logger.info("Loaded run snapshot from %s", args.from_run)
    return args


def main() -> None:
    """Run full pipeline, persist artifacts, and print structured JSON report."""
    args = parse_args()
    configure_logging(args.log_level)

    tracker = ExperimentTracker(base_dir=Path("runs"))
    args = _apply_from_run(args, tracker)
    set_global_seed(args.seed)

    logger.info("Loading config from %s", args.config)
    cfg = load_config(args.config)
    cfg_hash = tracker.config_hash(args.config)

    logger.info("Config hash: %s", cfg_hash)
    logger.info("Reading data from %s", args.data)
    df = pd.read_csv(args.data)

    loaded_adapter_config = getattr(args, "loaded_adapter_config", None)
    if loaded_adapter_config:
        adapter_cfg = GeneralAdapterConfig(**loaded_adapter_config)
    else:
        adapter_cfg = _build_general_config(cfg, args)
    run_id, run_path = tracker.create_run(adapter_cfg.model_type)

    if args.use_bio_adapter:
        protein_df = pd.read_csv(args.protein_data) if args.protein_data else None
        adapter = BioAdapter()
        exec_cfg = BioAdapterConfig(**asdict(adapter_cfg), protein_df=protein_df)
        logger.info("Running BioAdapter")
        details = adapter.run_with_details(df=df, config=exec_cfg)
    else:
        adapter = GeneralAdapter()
        logger.info("Running GeneralAdapter")
        details = adapter.run_with_details(df=df, config=adapter_cfg)

    report = details["report"]
    model_result = details["model_result"]

    report_body = report.get("report", {})
    top_features = report_body.get("top_features", [])
    metrics = report_body.get("model", {}).get("metrics", model_result.metrics)

    snapshot = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "config_path": str(args.config.resolve()),
        "config_hash": cfg_hash,
        "seed": args.seed,
        "input": {
            "data_path": str(args.data.resolve()),
            "protein_data_path": None if args.protein_data is None else str(args.protein_data.resolve()),
            "use_bio_adapter": args.use_bio_adapter,
        },
        "adapter_config": asdict(adapter_cfg),
    }

    tracker.save_yaml(run_path / "config_snapshot.yaml", snapshot)
    tracker.save_yaml(run_path / "config_model_snapshot.yaml", cfg)
    tracker.save_yaml(run_path / "model_hyperparameters.yaml", asdict(adapter_cfg))
    tracker.save_json(run_path / "metrics.json", metrics)
    tracker.save_json(run_path / "top_features.json", top_features)
    tracker.save_text(run_path / "config_hash.txt", cfg_hash)

    tracker.save_json(
        run_path / "agent_outputs.json",
        {
            "summary_agent": details["summary_output"],
            "interpretation_agent": details["interpretation_output"],
            "domain_mapping_agent": report_body.get("domain_mapping", {}),
            "executive_report_agent": report,
        },
    )

    tracker.save_json(run_path / "final_report.json", report)
    model_path = tracker.save_model_artifact(model=model_result.model, model_type=adapter_cfg.model_type, run_path=run_path)

    logger.info("Run saved to %s", run_path)
    logger.info("Model artifact saved to %s", model_path)

    output_payload = {
        "run_id": run_id,
        "config_hash": cfg_hash,
        "run_path": str(run_path.resolve()),
        "report": report,
    }

    output = json.dumps(output_payload, indent=2)
    print(output)

    if args.output is not None:
        args.output.write_text(output, encoding="utf-8")
        logger.info("Saved CLI output to %s", args.output)


if __name__ == "__main__":
    main()
