# tools/merge_parquet_to_json.py
"""
Merge all Parquet files under data/parquet/ (any date=YYYY-MM-DD folder)
into one big JSONL file: all_tweets.jsonl
"""

import glob
import os
import pandas as pd

PARQUET_DIR = r"C:\Users\admin\scraper\data\parquet"
OUT_FILE = r"C:\Users\admin\scraper\all_tweets.jsonl"

def main():
    files = glob.glob(os.path.join(PARQUET_DIR, "date=*/*.parquet"))
    print(f"Found {len(files)} parquet files")
    if not files:
        return

    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
            print(f"Loaded {f}, rows={len(df)}")
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not dfs:
        print("No data loaded, exiting.")
        return

    merged = pd.concat(dfs, ignore_index=True)
    print("Total merged rows:", len(merged))

    # Write JSONL
    merged.to_json(OUT_FILE, orient="records", lines=True, force_ascii=False)
    print(f"Wrote {len(merged)} rows to {OUT_FILE}")

if __name__ == "__main__":
    main()
