# Twitter/X Scraper + Processor + Features + Analytics

This repository implements an ingestion pipeline for market-related tweets and converts textual data into quantitative signals for algorithmic trading.

## Components
- `ingestion/` (Playwright scraper): collects tweets into `india_stock_tweets_pw_auth.csv`.
- `processor/` (worker): normalizes, deduplicates and writes Parquet partitions + batch JSONL files.
- `features/` (feature extraction): TF-IDF + sentence-transformer embeddings per batch.
- `analytics/` (signal aggregation): aggregates batch-level features into signals with confidence intervals.

## Quickstart
1. Create virtualenv and install requirements:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   playwright install
