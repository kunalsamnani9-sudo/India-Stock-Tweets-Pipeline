# features/stream_features.py
"""
Streaming feature extractor (improved).

- Reads JSONL batch files (ingest/*.jsonl)
- Produces per-batch .npz files in features/
- Moves processed JSONL -> ingest/processed/ to avoid re-processing
- Safe: per-file exception handling, minimal memory overhead
- Config via environment variables:
    INGEST_DIR, OUT_DIR, EMBED_MODEL, EMBED_BATCH, HASHING_FEATURES, PAUSE_BETWEEN_FILES
"""
import os
import json
import time
import glob
import shutil
from typing import List, Dict, Optional

import numpy as np

# sklearn import can be heavy; keep top-level but safe to fail early
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sentence_transformers import SentenceTransformer

# ------- Config -------
INGEST_DIR = os.environ.get('INGEST_DIR', 'ingest')
PROCESSED_DIR = os.path.join(INGEST_DIR, 'processed')
OUT_DIR = os.environ.get('FEATURE_DIR', 'features')
EMBED_MODEL = os.environ.get('EMBED_MODEL', 'all-MiniLM-L6-v2')
EMBED_BATCH = int(os.environ.get('EMBED_BATCH', '64'))
HASHING_FEATURES = int(os.environ.get('HASHING_FEATURES', str(2**18)))  # ~262k dims
PAUSE_BETWEEN_FILES = float(os.environ.get('PAUSE_BETWEEN_FILES', '0.5'))  # seconds

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(INGEST_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ------- Models / transformers (global) -------
vectorizer = HashingVectorizer(n_features=HASHING_FEATURES, alternate_sign=False, norm=None)
tfidf_transformer = TfidfTransformer()
embedder = SentenceTransformer(EMBED_MODEL)

# ------- Helpers -------

def _read_jsonl(path: str) -> List[Dict]:
    """Read a JSONL file into a list of dicts (small batches OK)."""
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # tolerate malformed lines
                continue
    return rows

def _safe_write_npz(out_path: str, embeddings: np.ndarray, meta: List[Dict], sp_data, sp_indices, sp_indptr, n_features: int):
    """Write .npz atomically (write to tmp then rename)."""
    tmp = out_path + ".tmp"
    np.savez_compressed(tmp,
                        embeddings=embeddings,
                        meta=np.array(meta, dtype=object),
                        tf_data=sp_data, tf_indices=sp_indices, tf_indptr=sp_indptr, tf_n_features=n_features)
    os.replace(tmp, out_path)

def process_jsonl_file(jsonl_path: str) -> Optional[str]:
    """
    Process one JSONL file and write features_<ts>.npz in OUT_DIR.
    Returns output filename on success, None on failure.
    """
    try:
        rows = _read_jsonl(jsonl_path)
        if not rows:
            print(f"[SKIP] {jsonl_path} (no rows)")
            return None

        texts = []
        meta = []
        for r in rows:
            texts.append((r.get('content') or '').strip())
            meta.append({
                'id': r.get('id'),
                'date': r.get('date'),
                'likeCount': int(r.get('likeCount') or 0),
                'retweetCount': int(r.get('retweetCount') or 0),
                'source_hashtag': r.get('source_hashtag') or ''
            })

        # TF (hashing) -> TF-IDF
        Xh = vectorizer.transform(texts)
        Xtfidf = tfidf_transformer.fit_transform(Xh)  # transforms counts -> tfidf

        # Embeddings in batches (to avoid large memory)
        emb_list = []
        for start in range(0, len(texts), EMBED_BATCH):
            chunk = texts[start:start+EMBED_BATCH]
            embs = embedder.encode(chunk, batch_size=EMBED_BATCH, show_progress_bar=False)
            emb_list.append(np.asarray(embs, dtype='float32'))
        embeddings = np.vstack(emb_list)

        # Serialize sparse tfidf (csr)
        sp = Xtfidf.tocsr()
        data = sp.data.astype('float32')
        indices = sp.indices.astype('int32')
        indptr = sp.indptr.astype('int64')
        n_features = sp.shape[1]

        ts = int(time.time())
        out_fn = os.path.join(OUT_DIR, f'features_{ts}.npz')
        _safe_write_npz(out_fn, embeddings, meta, data, indices, indptr, n_features)
        print(f"[OK] Wrote features: {out_fn} (rows={len(texts)})")
        return out_fn
    except Exception as e:
        print(f"[ERR] Failed processing {jsonl_path}: {e}")
        return None

def find_and_process_all(move_processed: bool = True, dry_run: bool = False):
    """
    Process all JSONL files found in INGEST_DIR matching batch_*.jsonl and idle_batch_*.jsonl.
    If move_processed: move processed input files to ingest/processed/.
    If dry_run: do not write outputs, just print what would be done.
    """
    patterns = [os.path.join(INGEST_DIR, 'batch_*.jsonl'), os.path.join(INGEST_DIR, 'idle_batch_*.jsonl')]
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)))
    if not files:
        print("[INFO] No JSONL batches found in", INGEST_DIR)
        return

    for fpath in files:
        print("[PROCESS] ", fpath)
        if dry_run:
            print(" - dry-run: would process", fpath)
            continue
        out = process_jsonl_file(fpath)
        if out and move_processed:
            try:
                dest = os.path.join(PROCESSED_DIR, os.path.basename(fpath))
                shutil.move(fpath, dest)
                print(f" - moved {fpath} -> {dest}")
            except Exception as e:
                # if move fails, just warn and continue
                print(f" - warning: failed to move {fpath} to processed/: {e}")
        # small pause to avoid hammering model / disk
        time.sleep(PAUSE_BETWEEN_FILES)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Streaming feature extractor')
    parser.add_argument('--no-move', dest='move', action='store_false', help='Do not move processed JSONL files (leave them in place)')
    parser.add_argument('--dry-run', dest='dry', action='store_true', help='List files to process but do not write outputs')
    args = parser.parse_args()
    find_and_process_all(move_processed=args.move, dry_run=args.dry)
