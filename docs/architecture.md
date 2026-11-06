
# Architecture - NHL Fantasy Data Platform

## 1. Project Objective

Build an automated NHL data platform to ingest, process and analyze hockey statistics in order to generate fantasy hockey player projections.

## 2. Data Sources

- NHL API
- Player statistics
- Team statistics
- Game schedules and results

## 3. Technology Stack

| Component | Technology |
|---|---|
| Data Lake | Azure Data Lake Storage Gen2 |
| Orchestration | Azure Data Factory |
| Processing | Azure Databricks |
| Languages | Python, PySpark, SQL |
| Storage Format | JSON, Delta |
| Machine Learning | Azure Machine Learning, MLflow |
| Version Control | GitHub |

## 4. Data Architecture

### Bronze
Raw data extracted from the NHL API and stored in JSON format.

### Silver
Cleaned, validated and standardized datasets.

### Gold
Business-ready datasets designed for fantasy hockey analysis and machine learning.

## 5. Machine Learning Objective

Develop a model that predicts player performance and helps identify valuable fantasy hockey players.

## 6. Automation

Schedule recurring data ingestion and transformation pipelines using Azure Data Factory and Databricks Workflows.
