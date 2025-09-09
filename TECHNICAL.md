
## `TECHNICAL.md` (brief technical documentation & approach)
```markdown
# Technical documentation — India Stock Tweets Pipeline

## Goals matched to assignment
1. **Real-time processing + deduplication**
   - Playwright collects tweets into `india_stock_tweets_pw_auth.csv`.
   - Processor `processor/process_all_now.py` deduplicates with SQLite (`dedupe.db`), normalizes Unicode with `ftfy` and writes partitioned Parquet (by date).
   - JSONL batches are written to `ingest/` for feature extraction.

2. **Data cleaning & storage**
   - `ftfy` and Unicode NFC normalization used to clean text.
   - Deduplication by SHA1(content) persisted to SQLite.
   - Parquet storage (pyarrow) used for compact, columnar storage by date partition.

3. **Feature engineering**
   - Stateless TF via `HashingVectorizer` + local `TfidfTransformer` per chunk.
   - Sentence embeddings via `sentence-transformers` (default `all-MiniLM-L6-v2`).
   - `.npz` per batch contains `embeddings`, `meta` and sparse TF-IDF arrays.

4. **Analysis & signals**
   - Embedding-based sentiment: cosine similarity to positive/negative centroids.
   - Lexical polarity: keyword-based normalized score.
   - Composite signal = weighted sum of normalized embedding + lexical scores.
   - Windows aggregated (1m/5m/15m), compute mean + SEM + 95% CI (student-t).
   - Decision rule implemented: **if most tweets in window are bullish → SHORT; if most bearish → LONG**.

5. **Visualization**
   - Memory-aware plotting: downsampling windows when plotting > 800 points; shaded 95% CI and trade markers.

6. **Scalability**
   - Streaming design: JSONL batches processed in chunks to cap memory usage.
   - HashingVectorizer avoids building large vocabularies.
   - For scale, swap SQLite for Redis/Kafka and Parquet with a data lake writer, and add distributed workers.

## Files & Usage
- `ingestion/` — Playwright scripts; requires `state.json` created by manual login script (save_x_session.py)
- `processor/` — CSV → Parquet and JSONL generator
- `features/` — JSONL → features `.npz`
- `analytics/` — aggregation and visualization

## Repro steps (example)
1. Create venv & install `pip install -r requirements.txt`.
2. Put sample `all_tweets.jsonl` in root.
3. Run `python features/jsonl_to_features.py --input all_tweets.jsonl`.
4. Run `python analytics/produce_signals.py --npz-pattern "features/features_*.npz" --jsonl all_tweets.jsonl`.

