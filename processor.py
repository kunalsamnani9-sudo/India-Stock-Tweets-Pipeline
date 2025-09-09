# processor/process_all_now.py
"""
One-off synchronous CSV -> Parquet processor for debugging.
Reads the entire CSV, dedupes against dedupe.db, writes parquet synchronously,
and prints detailed timing & counts. Also writes per-batch JSONL files into ingest/
so the features worker can pick them up.

Usage: python processor/process_all_now.py
"""

import os, time, csv, sqlite3, json, traceback
from datetime import datetime, timezone
import hashlib
import ftfy, unicodedata
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RAW_CSV = os.environ.get('RAW_CSV', 'india_stock_tweets_pw_auth.csv')
DUP_DB = os.environ.get('DUP_DB', 'dedupe.db')
OUT_PARQUET_DIR = os.environ.get('OUT_PARQUET_DIR', 'data/parquet')
INGEST_DIR = os.environ.get('INGEST_DIR', 'ingest')
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '200'))
PARQUET_COMPRESSION = os.environ.get('PARQUET_COMPRESSION', 'snappy')

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = ftfy.fix_text(s)
    s = unicodedata.normalize('NFC', s)
    # collapse whitespace
    s = " ".join(s.split())
    return s

def sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode('utf-8')).hexdigest()

def ensure_dirs():
    os.makedirs(OUT_PARQUET_DIR, exist_ok=True)
    os.makedirs(INGEST_DIR, exist_ok=True)

class DedupeDB:
    def __init__(self, path=DUP_DB):
        self.path = path
        existed = os.path.exists(path)
        self.conn = sqlite3.connect(path, timeout=30)
        self.conn.execute('PRAGMA journal_mode=WAL;')  # speed up concurrency
        self.conn.execute('PRAGMA synchronous=NORMAL;')
        self.conn.commit()
        self.conn.execute('CREATE TABLE IF NOT EXISTS dedupe (content_hash TEXT PRIMARY KEY, seen_at TEXT);')
        self.conn.commit()
    def seen_batch(self, hashes):
        if not hashes:
            return set()
        found = set()
        cur = self.conn.cursor()
        chunk = 800
        for i in range(0, len(hashes), chunk):
            part = hashes[i:i+chunk]
            placeholders = ",".join("?" for _ in part)
            sql = f"SELECT content_hash FROM dedupe WHERE content_hash IN ({placeholders})"
            cur.execute(sql, part)
            for r in cur.fetchall():
                found.add(r[0])
        return found
    def mark_batch(self, hashes_with_ts):
        if not hashes_with_ts:
            return
        cur = self.conn.cursor()
        try:
            cur.executemany("INSERT OR IGNORE INTO dedupe(content_hash, seen_at) VALUES(?,?)", hashes_with_ts)
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

def read_all_rows(csv_path):
    rows = []
    if not os.path.exists(csv_path):
        print("CSV not found:", csv_path)
        return rows
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def write_parquet_sync(df):
    if df.empty:
        return []
    df = df.copy()
    df['ts'] = pd.to_datetime(df['date'], errors='coerce')
    df['date_only'] = df['ts'].dt.strftime('%Y-%m-%d').fillna('unknown')

    written = []
    for date_val, group in df.groupby('date_only'):
        table = pa.Table.from_pandas(group.drop(columns=['date_only']))
        out_path = os.path.join(OUT_PARQUET_DIR, f'date={date_val}')
        os.makedirs(out_path, exist_ok=True)
        fn = os.path.join(out_path, f'tweets_{int(time.time())}.parquet')
        pq.write_table(table, fn, compression=PARQUET_COMPRESSION)
        print(f"WROTE {len(group)} rows -> {fn}")
        written.append(fn)
    return written

def write_jsonl_batch(cleaned_records: list, batch_ts: int, batch_idx: int) -> str:
    """
    Write cleaned records (list of dicts) as a JSONL file into INGEST_DIR.
    Returns path to created file.
    """
    if not cleaned_records:
        return ""
    os.makedirs(INGEST_DIR, exist_ok=True)
    fn = os.path.join(INGEST_DIR, f"batch_{batch_ts}_{batch_idx}.jsonl")
    tmp = fn + ".tmp"
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            for rec in cleaned_records:
                # Ensure JSON safe types
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp, fn)
        print(f"WROTE JSONL -> {fn}")
        return fn
    except Exception as e:
        print("Failed to write JSONL batch:", e)
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass
        return ""

def process_all():
    ensure_dirs()
    rows = read_all_rows(RAW_CSV)
    print("Total rows read from CSV:", len(rows))
    if not rows:
        return
    ded = DedupeDB()
    total_cleaned = 0
    all_written = []
    jsonl_files = []
    idx = 0
    start = time.time()
    batch_counter = 0
    while idx < len(rows):
        batch = rows[idx: idx + BATCH_SIZE]
        idx += BATCH_SIZE
        batch_counter += 1
        print(f"\nProcessing batch rows {idx - len(batch)}..{idx-1} (size={len(batch)})")
        # prepare candidates
        candidates = []
        for r in batch:
            try:
                content = normalize_text(r.get('content') or r.get('text') or '')
                if not content:
                    continue
                c_hash = sha1_hex(content)
                candidates.append((c_hash, content, r))
            except Exception as e:
                print("candidate prep error:", e)
        hashes = [c[0] for c in candidates]
        seen = ded.seen_batch(hashes)
        print("  batch hashes:", len(hashes), "already seen:", len(seen))
        cleaned = []
        to_mark = []
        for c_hash, content, orig in candidates:
            if c_hash in seen:
                continue
            ts = orig.get('date') or orig.get('ts') or ''
            try:
                parsed = pd.to_datetime(ts, utc=True, errors='coerce')
                if pd.isna(parsed):
                    parsed = datetime.utcnow().replace(tzinfo=timezone.utc)
            except Exception:
                parsed = datetime.utcnow().replace(tzinfo=timezone.utc)
            rec = {
                'id': orig.get('id') or c_hash,
                'username': (orig.get('username') or '').strip(),
                'date': parsed.isoformat(),
                'content': content,
                'replyCount': int(orig.get('replyCount') or 0) if orig.get('replyCount') else None,
                'retweetCount': int(orig.get('retweetCount') or 0) if orig.get('retweetCount') else None,
                'likeCount': int(orig.get('likeCount') or 0) if orig.get('likeCount') else None,
                'quoteCount': int(orig.get('quoteCount') or 0) if orig.get('quoteCount') else None,
                'mentions': orig.get('mentions') or '',
                'hashtags': orig.get('hashtags') or '',
                'url': orig.get('url') or '',
                'source_hashtag': orig.get('source_hashtag') or ''
            }
            cleaned.append(rec)
            to_mark.append((c_hash, datetime.utcnow().isoformat()))
        print("  cleaned records to write:", len(cleaned))

        if cleaned:
            # write Parquet
            df = pd.DataFrame(cleaned)
            t0 = time.time()
            try:
                written_files = write_parquet_sync(df)
                all_written.extend(written_files)
            except Exception as e:
                print("Parquet write error:", e)
                traceback.print_exc()
                return
            t1 = time.time()
            print(f"  parquet write time: {t1-t0:.3f}s")

            # write JSONL for downstream feature extraction
            jsonl_path = write_jsonl_batch(cleaned, int(time.time()), batch_counter)
            if jsonl_path:
                jsonl_files.append(jsonl_path)

            # mark dedupe db
            try:
                ded.mark_batch(to_mark)
            except Exception as e:
                print("Dedupe mark error:", e)
                traceback.print_exc()
                return

            total_cleaned += len(cleaned)

    end = time.time()
    print(f"\nDone. total_cleaned={total_cleaned}. total_files_written={len(all_written)}. elapsed={end-start:.2f}s")
    if jsonl_files:
        print("JSONL batches written:")
        for j in jsonl_files:
            print(" ", j)

if __name__ == '__main__':
    process_all()
