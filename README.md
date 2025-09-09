# India Stock Tweets Pipeline

Pipeline to collect, process and analyze Twitter / X posts related to Indian stock markets,
produce features (embeddings + TF-IDF) and aggregate signals for simple algorithmic trading insights.

This repository contains:
- Scraper (Playwright) that re-uses an authenticated storage state (ingestion/)
- CSV → Parquet / JSONL processor for deduplication and cleaning (processor/)
- Streaming / batch feature extractor (features/)
- Analytics aggregator producing time-windowed signals and visualizations (analytics/)
- Tools for merging Parquet → JSONL and other helpers (tools/)

## Quick start (developer machine, Windows)
1. clone repo:
```bash
git clone <your-repo-url>
cd india-stock-twitter-pipeline

2. create & activate venv (Windows cmd):
C:\> python -m venv .venv
C:\> .\.venv\Scripts\activate

3. install dependencies:
pip install -r requirements.txt

   

india-stock-twitter-pipeline/
├─ .github/
│  └─ workflows/
│     └─ ci.yml
├─ analytics/
│  └─ produce_signals.py     # updated script you ran
│  └─ (signals_*.csv, signals_*.json, signals_*.png)  # put sample outputs here
├─ features/
│  └─ jsonl_to_features.py
│  └─ stream_features.py
│  └─ (features_*.npz)      # sample features
├─ ingestion/
│  └─ pw_collect_india_stocks_with_state_final.py
│  └─ (other ingestion scripts)
├─ processor/
│  └─ processor.py
├─ tools/
│  └─ merge_parquet_to_json.py
│  └─ parquet_to_json.py
├─ data/                     # small example data (NOT large). Add .gitkeep if empty.
│  └─ parquet/
│  └─ json/
├─ ingest/                   # ingest JSONL files created by processor
├─ all_tweets.jsonl          # example merged JSONL (small)
├─ README.md
├─ TECHNICAL.md              # architecture & decisions
├─ requirements.txt
├─ .gitignore
└─ LICENSE

