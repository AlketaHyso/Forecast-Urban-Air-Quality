# Project Charter

## Project Title

Albania Urban Air Quality Early Warning and Decision-Support System

## Problem Statement

Urban air quality affects public health, transport exposure, and day-to-day environmental risk in Albanian cities. Open environmental data exist, but they are rarely turned into usable operational tools for monitoring, warning, and data-driven urban interpretation. The project addresses this gap by building a JupyterLab-based analytical application that integrates air-quality and weather data, detects pollution episodes, and prepares the foundation for short-horizon forecasting.

## Project Goal

Design a serious engineering prototype that can:

- ingest city-level air-quality and weather data for Albania
- produce validated hourly and daily analytical datasets
- label pollution-risk episodes using a transparent rule-based scheme
- support visual monitoring across cities
- serve as a base for later forecasting and early-warning modeling

## Primary Users

- environmental researchers
- municipal analysts
- public policy and planning staff
- engineering students and applied data teams

## Scope of the First Serious Prototype

### Included

- automated data ingestion from free APIs
- processed hourly and daily datasets
- AQI-based risk labeling
- city comparison views
- pollution-episode summaries
- JupyterLab dashboard for exploratory decision support

### Deferred to the Next Phase

- 24-72 hour forecasting models
- automated refresh scheduling
- web deployment outside JupyterLab
- alert notifications

## Engineering Research Questions

1. Can open environmental data support a city-level early-warning workflow for Albanian urban air quality?
2. Which Albanian cities exhibit the highest concentration of elevated AQI days in the current observation window?
3. Can a structured daily feature layer support later short-term forecasting of PM2.5 and AQI?

## Core Deliverables

1. Data ingestion script
2. Daily feature engineering and risk-label pipeline
3. Processed datasets and summary tables
4. JupyterLab dashboard notebook
5. Architecture and documentation files

## Success Criteria for Phase 1

- the pipeline runs end to end without manual spreadsheet editing
- the daily dataset is complete and traceable
- high-risk days are labeled consistently
- the dashboard helps compare cities and inspect pollution events
- the project is ready for a forecasting phase without redesigning the data layer

## Phase Roadmap

### Phase 1

Data ingestion, daily aggregation, risk labeling, and dashboard exploration

### Phase 2

Short-term forecasting benchmark for PM2.5 and AQI

### Phase 3

Early-warning logic, model comparison, and publication-grade outputs
