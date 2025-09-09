# features/jsonl_to_features.py
"""
Convert a big JSONL (one tweet per line) into feature .npz files.

Output per chunk:
  features/features_<timestamp>_<idx>.npz
Contains:
  - embeddings: float32 (N x D)
  - meta: numpy object array (list of dicts with id,date,likeCount,source_hashtag...)
  - tf_data, tf_indices, tf_indptr, tf_n_features: CSR sparse TF-IDF data

Usage:
  python features/jsonl_to_features.py --input all_tweets.jsonl --outdir features --chunk-size 500
"""
import os
import json
import time
import argparse
import tempfile
import traceback
from typing import List, Dict

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sentence_transformers import SentenceTransformer

# -------- Defaults / config ----------
DEFAULT_INPUT = "all_tweets.jsonl"
DEFAULT_OUTDIR = "features"
DEFAULT_CHUNK = 500                # how many tweets per .npz chunk
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_EMBED_BATCH = 64           # internal batch for embedder.encode
DEFAULT_HASHING_FEATURES = 2**18   # ~262k dims (stateless, memory-friendly)

# -------- Helpers ----------
def safe_make_dirs(path: str):
    os.makedirs(path, exist_ok=True)

def write_npz_atomic(out_path: str, embeddings: np.ndarray, meta: List[Dict],
                     tf_data, tf_indices, tf_indptr, tf_n_features):
    """
    Write .npz atomically and robustly on Windows:
      - create temp file in the same directory as out_path (so os.replace is atomic)
      - ensure exceptions are logged and temp cleaned up on failure
    """
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    try:
        # create a named temporary file in the target directory
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_features_", suffix=".npz", dir=out_dir, text=False)
        os.close(fd)  # close low-level fd; numpy will open path
        # write compressed npz to tmp_path
        np.savez_compressed(tmp_path,
                            embeddings=embeddings,
                            meta=np.array(meta, dtype=object),
                            tf_data=tf_data, tf_indices=tf_indices, tf_indptr=tf_indptr, tf_n_features=tf_n_features)
        # double-check file exists before replace
        if not os.path.exists(tmp_path):
            raise FileNotFoundError(f"Temporary file not written: {tmp_path}")
        # atomic replace (works on Windows when same filesystem/dir)
        os.replace(tmp_path, out_path)
    except Exception as e:
        # try to clean up tmp file if present
        try:
            if 'tmp_path' in locals() and tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        tb = traceback.format_exc()
        raise RuntimeError(f"Failed to write npz atomically to {out_path}: {e}\n{tb}")

def process_chunk(texts: List[str], meta: List[Dict],
                  vectorizer: HashingVectorizer, tfidf_transformer: TfidfTransformer,
                  embedder: SentenceTransformer, embed_batch: int, outdir: str, chunk_idx: int) -> str:
    """
    Process a single chunk of texts+meta and write features .npz.
    Returns output filename or empty string on failure.
    """
    if not texts:
        return ""
    try:
        # TF (hashing) then TF-IDF (fit_transform per chunk)
        Xh = vectorizer.transform(texts)               # sparse
        Xtfidf = tfidf_transformer.fit_transform(Xh)   # convert counts -> tfidf

        # embeddings in sub-batches to avoid memory blow
        emb_list = []
        for s in range(0, len(texts), embed_batch):
            chunk = texts[s:s+embed_batch]
            embs = embedder.encode(chunk, batch_size=embed_batch, show_progress_bar=False)
            emb_list.append(np.asarray(embs, dtype='float32'))
        embeddings = np.vstack(emb_list)

        # sparse serialize
        sp = Xtfidf.tocsr()
        data = sp.data.astype('float32')
        indices = sp.indices.astype('int32')
        indptr = sp.indptr.astype('int64')
        n_features = sp.shape[1]

        ts = int(time.time())
        out_fn = os.path.join(outdir, f"features_{ts}_{chunk_idx}.npz")
        write_npz_atomic(out_fn, embeddings, meta, data, indices, indptr, n_features)
        return out_fn
    except Exception as e:
        print(f"[ERR] process_chunk failed: {e}")
        print(traceback.format_exc())
        return ""

# -------- Main ----------
def main(input_path: str, outdir: str, chunk_size: int, embed_model: str, embed_batch: int, hashing_features: int):
    if not os.path.exists(input_path):
        raise SystemExit(f"Input JSONL not found: {input_path}")

    safe_make_dirs(outdir)
    print(f"Using embed model: {embed_model}")
    print(f"Reading: {input_path}")
    print(f"Writing features into: {outdir} (chunk size {chunk_size})")

    # instantiate vectorizer / transformer / embedder
    vectorizer = HashingVectorizer(n_features=hashing_features, alternate_sign=False, norm=None)
    tfidf_transformer = TfidfTransformer()
    embedder = SentenceTransformer(embed_model)

    texts = []
    meta = []
    chunk_idx = 0
    total_rows = 0
    written_files = []

    # stream through the JSONL file
    with open(input_path, 'r', encoding='utf-8') as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                j = json.loads(ln)
            except Exception:
                # tolerate malformed lines
                continue
            # extract content and metadata; fallbacks used to remain robust
            content = (j.get('content') or j.get('text') or "").strip()
            if not content:
                continue
            texts.append(content)
            meta.append({
                'id': j.get('id') or '',
                'date': j.get('date') or j.get('ts') or '',
                'likeCount': int(j.get('likeCount') or 0) if j.get('likeCount') else 0,
                'retweetCount': int(j.get('retweetCount') or 0) if j.get('retweetCount') else 0,
                'source_hashtag': j.get('source_hashtag') or j.get('hashtags') or ''
            })
            total_rows += 1

            # when chunk full, process and write .npz
            if len(texts) >= chunk_size:
                chunk_idx += 1
                print(f"[CHUNK] Processing chunk #{chunk_idx} rows={len(texts)} total={total_rows}")
                out = process_chunk(texts, meta, vectorizer, tfidf_transformer, embedder, embed_batch, outdir, chunk_idx)
                if out:
                    written_files.append(out)
                    print(f"[OK] Wrote {out}")
                else:
                    print(f"[WARN] chunk #{chunk_idx} failed to write")
                texts = []
                meta = []

    # final partial chunk
    if texts:
        chunk_idx += 1
        print(f"[CHUNK] Final chunk #{chunk_idx} rows={len(texts)} total={total_rows}")
        out = process_chunk(texts, meta, vectorizer, tfidf_transformer, embedder, embed_batch, outdir, chunk_idx)
        if out:
            written_files.append(out)
            print(f"[OK] Wrote {out}")
        else:
            print(f"[WARN] final chunk #{chunk_idx} failed to write")

    print(f"\nDone. total_rows={total_rows}. chunks={chunk_idx}. files_written={len(written_files)}")
    return written_files

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert JSONL -> features .npz")
    ap.add_argument("--input", "-i", default=DEFAULT_INPUT, help="Input JSONL file")
    ap.add_argument("--outdir", "-o", default=DEFAULT_OUTDIR, help="Output directory for .npz files")
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK, help="How many tweets per .npz chunk")
    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="SentenceTransformer model name")
    ap.add_argument("--embed-batch", type=int, default=DEFAULT_EMBED_BATCH, help="Batch size when encoding embeddings")
    ap.add_argument("--hashing-features", type=int, default=DEFAULT_HASHING_FEATURES, help="HashingVectorizer n_features")
    args = ap.parse_args()
    main(args.input, args.outdir, args.chunk_size, args.embed_model, args.embed_batch, args.hashing_features)
