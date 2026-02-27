# OmniInsight

## 1. Project Motivation

OmniInsight is a modular AI system designed to simulate end-to-end biological and general data intelligence pipelines.

This project was built to explore how AI can:
- Integrate heterogeneous datasets (including multi-omics)
- Train reproducible machine learning models
- Abstract feature-level signals into domain-level insights
- Generate hypothesis-oriented executive reports
- Maintain experiment traceability and reproducibility

The architecture reflects real-world AI-driven bio research systems where modeling, interpretability, and domain reasoning must operate as loosely coupled but coordinated modules.

---

## 2. System Architecture

### 2.1 High-Level Flow

Data Ingestion  
-> Preprocessing  
-> Model Training  
-> Feature Interpretation  
-> Domain Mapping  
-> Executive Report Generation  
-> Run Artifact Tracking

### 2.2 Design Principles

**Modular Core**
- `core/`: preprocessing, training, model engine
- `interpretation/`: SHAP + biological abstraction
- `adapters/`: domain-specific orchestration
- `agents/`: contract-based reasoning modules
- `dashboard/`: user-facing interface

**Adapter Pattern**
- `GeneralAdapter` for generic ML workflows
- `BioAdapter` for multi-omics and biological abstraction

**Contract-Based Multi-Agent Design**
- Each agent defines:
  - `InputSchema` (Pydantic)
  - `OutputSchema` (Pydantic)
- All inputs/outputs validated before execution
- Execution traces persisted per run

---

## 3. Bio AI Extension

### 3.1 Multi-Omics Integration

Supports:
- Transcriptomics
- Proteomics
- Metabolomics

Features are block-tracked and prefixed:
- `tx__`
- `pr__`
- `mt__`

Merged by sample ID and preserved through modeling and interpretation.

### 3.2 Pathway-Level Aggregation

Feature importance is abstracted:  
Gene-level importance -> Pathway-level scoring

- Deterministic pathway mapping
- Aggregated pathway importance scores
- Persisted per run

### 3.3 Biological Insight Engine

Generates:
- Top regulatory genes
- Dominant pathways
- Candidate bioengineering targets
- Hypothesis statements

---

## 4. Reproducibility & Experiment Tracking

Each execution creates:

`runs/{run_id}/`
- `config_snapshot.yaml`
- `metrics.json`
- model artifact (`.pt` / `.json`)
- `top_features.json`
- `pathway_scores.json`
- `agent_execution_trace.json`
- `final_report.json`

Features:
- Global seed control
- Config hashing (SHA256)
- Re-run from saved run (`--from-run`)

This ensures deterministic and traceable experimentation.

---

## 5. Example Use Case

A researcher uploads multi-omics data to predict production yield.

OmniInsight:
1. Integrates heterogeneous omics blocks
2. Trains predictive model
3. Identifies high-impact genes
4. Aggregates pathway-level importance
5. Suggests potential engineering targets
6. Generates structured executive report

---

## 6. Technical Stack

- Python
- XGBoost
- PyTorch
- SHAP (with fallback)
- Pydantic (agent contract validation)
- Streamlit (dashboard)

---

## 7. Future Extensions

- Ontology integration
- Real enrichment analysis
- LLM-based biological reasoning
- Production deployment mode
