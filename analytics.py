# analytics/produce_signals.py
"""
Produce aggregated trading signals from features (.npz) + JSONL with visual trade markers.

Key rule (per your spec):
  - If most tweets in a window are bullish -> TRADE = SHORT
  - If most tweets in a window are bearish -> TRADE = LONG
  - Otherwise -> HOLD

Visual:
  - Top panel: mean composite signal (line) with shaded 95% CI
  - Markers on top panel: SHORT -> red downward triangle, LONG -> green upward triangle, HOLD -> gray circle
  - Bottom panel: tweet counts per window (bars)

Outputs (in outdir):
  - signals_<ts>.csv
  - signals_<ts>.json
  - signals_<ts>.png (plot with trade markers)
"""
import os
import glob
import json
import time
import argparse
import math
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from scipy import stats
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import re
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------- Helpers -----------------
POSITIVE_WORDS = ["buy","long","bull","bullish","rally","up","profit","target","breakout","support"]
NEGATIVE_WORDS = ["sell","short","bear","bearish","down","loss","stoploss","crash","resistance"]

def load_npz_files(pattern: str) -> pd.DataFrame:
    rows = []
    files = sorted(glob.glob(pattern))
    for p in files:
        try:
            d = np.load(p, allow_pickle=True)
        except Exception as e:
            print("Failed to load", p, e)
            continue
        if 'embeddings' not in d or 'meta' not in d:
            print("Skipping missing arrays in", p)
            continue
        embs = d['embeddings']
        meta = list(d['meta'])
        for i, m in enumerate(meta):
            try:
                if isinstance(m, dict):
                    _id = m.get('id') or f"npz_{os.path.basename(p)}_{i}"
                    _date = m.get('date')
                else:
                    # fallback: try to coerce
                    _id = str(m[0]) if len(m)>0 else f"npz_{os.path.basename(p)}_{i}"
                    _date = m[1] if len(m)>1 else None
                rows.append({'id': str(_id), 'date': pd.to_datetime(_date, utc=True, errors='coerce'), 'embedding': embs[i]})
            except Exception:
                rows.append({'id': f"npz_{os.path.basename(p)}_{i}", 'date': pd.NaT, 'embedding': embs[i]})
    df = pd.DataFrame(rows)
    return df

def load_jsonl(jsonl_path: str) -> pd.DataFrame:
    rows = []
    with open(jsonl_path, 'r', encoding='utf-8') as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln: continue
            try:
                j = json.loads(ln)
            except Exception:
                continue
            rows.append({'id': str(j.get('id') or ''), 'date': pd.to_datetime(j.get('date') or j.get('ts') or None, utc=True, errors='coerce'), 'content': (j.get('content') or j.get('text') or '').strip()})
    return pd.DataFrame(rows)

def compute_centroids(embedder: SentenceTransformer, pos_words=POSITIVE_WORDS, neg_words=NEGATIVE_WORDS):
    pos_emb = embedder.encode(pos_words, show_progress_bar=False)
    neg_emb = embedder.encode(neg_words, show_progress_bar=False)
    pos_centroid = np.mean(pos_emb, axis=0)
    neg_centroid = np.mean(neg_emb, axis=0)
    return pos_centroid, neg_centroid

def cosine(a: np.ndarray, b: np.ndarray):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def embedding_sentiment(embeddings: np.ndarray, pos_centroid: np.ndarray, neg_centroid: np.ndarray) -> np.ndarray:
    scores = np.zeros((len(embeddings),), dtype='float32')
    for i, e in enumerate(embeddings):
        scores[i] = cosine(e, pos_centroid) - cosine(e, neg_centroid)
    return scores

def lexical_score(texts: List[str], pos_list=POSITIVE_WORDS, neg_list=NEGATIVE_WORDS) -> np.ndarray:
    pos_re = re.compile(r'\b(' + '|'.join(re.escape(w) for w in pos_list) + r')\b', flags=re.I)
    neg_re = re.compile(r'\b(' + '|'.join(re.escape(w) for w in neg_list) + r')\b', flags=re.I)
    out = np.zeros((len(texts),), dtype='float32')
    for i, t in enumerate(texts):
        if not isinstance(t, str) or not t.strip():
            out[i] = 0.0
            continue
        tokens = re.findall(r'\w+', t)
        n = max(1, len(tokens))
        pos = len(pos_re.findall(t))
        neg = len(neg_re.findall(t))
        out[i] = (pos - neg) / math.sqrt(n)
    return out

def compute_window_stats(series_vals: np.ndarray):
    n = len(series_vals)
    if n == 0:
        return {'n':0,'mean':0.0,'sem':0.0,'ci95_half':0.0}
    mean = float(np.mean(series_vals))
    sem = float(stats.sem(series_vals)) if n > 1 else 0.0
    if n > 1:
        tval = float(stats.t.ppf(0.975, df=n-1))
        ci_half = tval * sem
    else:
        ci_half = 0.0
    return {'n':n,'mean':mean,'sem':sem,'ci95_half':ci_half}

def decide_trade_from_window(proportion_positive: float, proportion_negative: float, threshold=0.6):
    # YOUR RULE: most bullish -> SHORT, most bearish -> LONG
    if proportion_positive >= threshold and proportion_positive > proportion_negative:
        return 'SHORT'
    if proportion_negative >= threshold and proportion_negative > proportion_positive:
        return 'LONG'
    return 'HOLD'

# ----------------- Main processing -----------------
def produce_signals(npz_pattern: str, jsonl_path: str, outdir: str, window: str, min_tweets_per_window:int, embed_model:str, weight_emb:float=0.7, weight_lex:float=0.3, threshold:float=0.6):
    os.makedirs(outdir, exist_ok=True)
    print("Loading embeddings from:", npz_pattern)
    emb_df = load_npz_files(npz_pattern)
    print("Loaded embeddings:", len(emb_df))
    print("Loading JSONL:", jsonl_path)
    txt_df = load_jsonl(jsonl_path)
    print("Loaded texts:", len(txt_df))

    # join
    df = emb_df.merge(txt_df, on='id', how='left', suffixes=('_emb','_txt'))
    df['content'] = df['content'].fillna('')
    df['date'] = df['date_emb'].fillna(df['date_txt'])
    df['date'] = df['date'].fillna(pd.Timestamp.utcnow())
    df = df.drop(columns=['date_emb','date_txt'], errors='ignore')
    df = df[~df['embedding'].isna()].reset_index(drop=True)
    if df.empty:
        raise SystemExit("No joined records (embeddings+text).")

    embeddings = np.vstack(df['embedding'].values)
    embedder = SentenceTransformer(embed_model)
    pos_centroid, neg_centroid = compute_centroids(embedder)

    emb_scores = embedding_sentiment(embeddings, pos_centroid, neg_centroid)  # higher = bullish
    lex_scores = lexical_score(df['content'].tolist())

    scaler = StandardScaler()
    comp_matrix = np.vstack([emb_scores, lex_scores]).T
    comp_scaled = scaler.fit_transform(comp_matrix)
    composite = weight_emb * comp_scaled[:,0] + weight_lex * comp_scaled[:,1]

    df['emb_score'] = emb_scores
    df['lex_score'] = lex_scores
    df['composite'] = composite
    df['timestamp'] = pd.to_datetime(df['date'], utc=True)
    df = df.sort_values('timestamp').set_index('timestamp')

    # window aggregation
    grouped = df['composite'].groupby(pd.Grouper(freq=window))
    pos_frac_series = (df['composite'] > 0).groupby(pd.Grouper(freq=window)).mean().fillna(0)
    neg_frac_series = (df['composite'] < 0).groupby(pd.Grouper(freq=window)).mean().fillna(0)
    counts_series = df['composite'].groupby(pd.Grouper(freq=window)).count().fillna(0).astype(int)

    agg = []
    for ts, vals in grouped:
        vals_arr = vals.values
        stats_ = compute_window_stats(vals_arr)
        n = stats_['n']
        mean = stats_['mean']
        sem = stats_['sem']
        ci = stats_['ci95_half']
        pos_frac = float(pos_frac_series.get(ts, 0.0))
        neg_frac = float(neg_frac_series.get(ts, 0.0))
        count = int(counts_series.get(ts, 0))
        trade_action = decide_trade_from_window(pos_frac, neg_frac, threshold=threshold)
        agg.append({'window_start': ts, 'n': n, 'count': count, 'mean_signal': mean, 'sem': sem, 'ci95_half': ci, 'proportion_positive': pos_frac, 'proportion_negative': neg_frac, 'trade_action': trade_action})

    agg_df = pd.DataFrame(agg).dropna(subset=['window_start']).reset_index(drop=True)
    agg_df['window_start_iso'] = agg_df['window_start'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    agg_df = agg_df[agg_df['count'] >= min_tweets_per_window].reset_index(drop=True)

    # outputs
    tsnow = int(time.time())
    csv_out = os.path.join(outdir, f"signals_{tsnow}.csv")
    json_out = os.path.join(outdir, f"signals_{tsnow}.json")
    png_out = os.path.join(outdir, f"signals_{tsnow}.png")
    agg_df.to_csv(csv_out, index=False)
    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(agg_df.drop(columns=['window_start']).to_dict(orient='records'), f, ensure_ascii=False, indent=2)
    print("Wrote:", csv_out, json_out)

    # Plotting with trade markers
    times = agg_df['window_start']
    means = agg_df['mean_signal'].astype(float)
    cis = agg_df['ci95_half'].astype(float)
    counts = agg_df['count'].astype(int)
    actions = agg_df['trade_action'].astype(str)

    N = len(times)
    max_points = 1200
    stride = max(1, int(math.ceil(N / max_points)))
    sel = list(range(0, N, stride))

    fig, ax = plt.subplots(2,1, figsize=(14,7), gridspec_kw={'height_ratios':[3,1]}, sharex=True)
    sel_times = times.iloc[sel]
    sel_means = means.iloc[sel]
    sel_cis = cis.iloc[sel]

    ax[0].plot(sel_times, sel_means, linewidth=1, label='mean composite')
    ax[0].fill_between(sel_times, sel_means - sel_cis, sel_means + sel_cis, alpha=0.2, label='95% CI')
    ax[0].axhline(0, color='k', linewidth=0.6, linestyle='--')

    # map actions to markers/colors
    marker_map = {'SHORT': ('v','red'), 'LONG': ('^','green'), 'HOLD': ('o','gray')}
    for i in sel:
        t = times.iloc[i]
        m = means.iloc[i]
        a = actions.iloc[i]
        mk, col = marker_map.get(a, ('o','gray'))
        # place marker slightly above/below depending on action for visibility
        y = m
        if a == 'SHORT':
            y = m + max(0.001, abs(m)*0.05)  # offset upward so down-triangle is visible
        elif a == 'LONG':
            y = m - max(0.001, abs(m)*0.05)
        ax[0].scatter([t], [y], marker=mk, color=col, s=60, zorder=5, edgecolors='k', linewidths=0.5)

    # build legend for actions
    handles = []
    for name, (mk, col) in marker_map.items():
        handles.append(plt.Line2D([0],[0], marker=mk, color='w', markerfacecolor=col, markersize=8, markeredgecolor='k', linestyle=''))
    ax[0].legend(handles, ['SHORT','LONG','HOLD'], title='Trade (bullish->SHORT, bearish->LONG)')

    ax[0].set_ylabel("Mean composite signal")

    # counts bar chart
    sel_counts = counts.iloc[sel]
    width = pd.to_timedelta(window).to_pytimedelta()
    ax[1].bar(sel_times, sel_counts, width=width, align='center')
    ax[1].set_ylabel("Tweet count")
    ax[1].set_xlabel("Window start")
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(png_out, dpi=150)
    print("Wrote plot:", png_out)

    overall = {'total_windows': len(agg_df), 'total_tweets_used': int(df.shape[0]), 'last_window': agg_df['window_start_iso'].max() if not agg_df.empty else None, 'trade_action_summary': agg_df['trade_action'].value_counts().to_dict()}
    print("Summary:", json.dumps(overall, indent=2))
    return {'csv':csv_out, 'json':json_out, 'png':png_out, 'summary':overall}

# ----------------- CLI -----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-pattern", default="features/features_*.npz")
    ap.add_argument("--jsonl", default="all_tweets.jsonl")
    ap.add_argument("--outdir", default="analytics")
    ap.add_argument("--window", default="5min")
    ap.add_argument("--min-tweets-per-window", type=int, default=3)
    ap.add_argument("--embed-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--weight-emb", type=float, default=0.7)
    ap.add_argument("--weight-lex", type=float, default=0.3)
    ap.add_argument("--threshold", type=float, default=0.6, help="proportion threshold to consider majority (0..1)")
    args = ap.parse_args()

    produce_signals(args.npz_pattern, args.jsonl, args.outdir, args.window, args.min_tweets_per_window, args.embed_model, args.weight_emb, args.weight_lex, args.threshold)
