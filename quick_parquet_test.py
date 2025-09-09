import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os, time

csv_path = "india_stock_tweets_pw_auth.csv"
out_dir = "data/parquet_test"
os.makedirs(out_dir, exist_ok=True)

print("Reading CSV:", csv_path)
df = pd.read_csv(csv_path)

print("Rows read:", len(df))
if df.empty:
    print("No rows in CSV, nothing to write.")
else:
    table = pa.Table.from_pandas(df)
    out_file = os.path.join(out_dir, f"test_{int(time.time())}.parquet")
    pq.write_table(table, out_file)
    print("Parquet file created:", out_file)
