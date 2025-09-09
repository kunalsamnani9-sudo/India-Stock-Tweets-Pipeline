# pw_collect_india_stocks_with_state_final.py
"""
Playwright scraper that re-uses an authenticated storage state (state.json).
Collects tweets for hashtags and keywords until a global cap is reached.

Usage:
  python pw_collect_india_stocks_with_state_final.py

Precondition: run save_x_session.py once and have state.json in same folder.
"""

from playwright.sync_api import sync_playwright
import time, csv, hashlib, re, datetime, random, os, sys, logging
from typing import Dict, List

# ------- Config -------
STATE_FILE = "state.json"
OUT_CSV = "india_stock_tweets_pw_auth.csv"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117 Safari/537.36"
HEADLESS = True  # set False to watch the browser (debug)
SCROLL_PAUSE = 0.9
SAVE_EVERY = 50
MAX_SCROLLS_PER_HASHTAG = 3000
GLOBAL_MAX_TWEETS = 2000  # stop after collecting this many total tweets

# Keywords to search. The script will cycle through these until the global cap is reached.
KEYWORDS: List[str] = [
    "#nifty50",
    "#sensex",
    "#intraday",
    "#banknifty",
    "#stockmarkets",
    "#stocks",
    "#trading",
    "#candlestick",
    "#market",
    "#ipo",
    "#investing",
    "#equities",
    "#stockmarketindia",
]

FIELDNAMES = ["id","username","date","content","replyCount","retweetCount","likeCount","quoteCount","mentions","hashtags","url","source_hashtag"]

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pw_auth_scraper")

# ------- Helpers -------
def make_pseudo_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def extract_metrics_from_article(article):
    replies = retweets = likes = quotes = None
    try:
        groups = article.query_selector_all('div[role="group"]')
        for g in groups:
            buttons = g.query_selector_all('*[aria-label]')
            for b in buttons:
                try:
                    label = b.get_attribute('aria-label') or ""
                except Exception:
                    label = ""
                m = re.search(r'([\d,\.]+)\s+([A-Za-z]+)', label)
                if m:
                    try:
                        val = int(m.group(1).replace(',', '').split('.')[0])
                    except Exception:
                        continue
                    kind = m.group(2).lower()
                    if 'like' in kind:
                        likes = val
                    elif 'retweet' in kind:
                        retweets = val
                    elif 'reply' in kind:
                        replies = val
                    elif 'quote' in kind:
                        quotes = val
    except Exception:
        pass
    try:
        txt = article.inner_text() or ""
        for label in ('likes', 'retweets', 'replies', 'quotes'):
            m = re.search(r'([0-9,]+)\s+' + label, txt, flags=re.IGNORECASE)
            if m:
                try:
                    val = int(m.group(1).replace(',', ''))
                except Exception:
                    continue
                if label == 'likes' and likes is None: likes = val
                if label == 'retweets' and retweets is None: retweets = val
                if label == 'replies' and replies is None: replies = val
                if label == 'quotes' and quotes is None: quotes = val
    except Exception:
        pass
    return replies, retweets, likes, quotes

def extract_username_and_time(article):
    username = None
    ts_iso = None
    try:
        time_el = article.query_selector("time")
        if time_el:
            ts_iso = time_el.get_attribute("datetime")
    except Exception:
        pass
    try:
        inner = article.inner_text() or ""
        lines = [ln.strip() for ln in inner.splitlines() if ln.strip()]
        for ln in lines[:6]:
            m = re.search(r'@([A-Za-z0-9_]{1,15})', ln)
            if m:
                username = m.group(1)
                break
        try:
            link_elems = article.query_selector_all("a")
            for a in link_elems:
                try:
                    href = a.get_attribute("href") or ""
                except Exception:
                    href = ""
                if href.startswith("/") and len(href) > 1 and not href.startswith("/i/"):
                    cand = href.split("/")[1]
                    if re.match(r'^[A-Za-z0-9_]{1,15}$', cand):
                        if cand.lower() not in ("status","search","i"):
                            username = cand
                            break
        except Exception:
            pass
    except Exception:
        pass
    return username, ts_iso

# ------- Main run -------
def run():
    if not os.path.exists(STATE_FILE):
        log.error("State file '%s' not found. Run save_x_session.py first and sign in interactively.", STATE_FILE)
        sys.exit(1)

    # prepare CSV
    f_exists = os.path.exists(OUT_CSV)
    out_f = open(OUT_CSV, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
    if not f_exists:
        writer.writeheader()
    out_f.flush()

    collected_global = set()
    per_tag_counts: Dict[str,int] = {tag:0 for tag in KEYWORDS}
    total_written = 0

    # Keep track of total scrolls done per keyword so we don't exceed MAX_SCROLLS_PER_HASHTAG
    scrolls_done: Dict[str,int] = {tag:0 for tag in KEYWORDS}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        context = browser.new_context(storage_state=STATE_FILE, user_agent=USER_AGENT)
        page = context.new_page()

        # We'll cycle through keywords round-robin. For each keyword visit we allow up to VISIT_SCROLLS scrolls (50),
        # and we accumulate scrolls per keyword (not to exceed MAX_SCROLLS_PER_HASHTAG). The run stops when
        # GLOBAL_MAX_TWEETS is reached or all keywords reached their MAX_SCROLLS_PER_HASHTAG.
        VISIT_SCROLLS = 50
        keyword_index = 0

        # Loop until global cap or all keywords exhausted
        while total_written < GLOBAL_MAX_TWEETS and any(scrolls_done[tag] < MAX_SCROLLS_PER_HASHTAG for tag in KEYWORDS):
            tag = KEYWORDS[keyword_index]
            # if this keyword already exhausted, skip to next
            if scrolls_done[tag] >= MAX_SCROLLS_PER_HASHTAG:
                keyword_index = (keyword_index + 1) % len(KEYWORDS)
                continue

            log.info("Visiting keyword %s (round-robin). Total collected so far: %d", tag, total_written)
            collected_this_visit = 0
            visit_scrolls = 0

            # navigate to search page for the keyword
            q = tag
            if not q.startswith("#") and not q.startswith("%23"):
                q = "#" + q
            url = f"https://x.com/search?q={q.replace('#','%23')}&f=live"
            try:
                page.goto(url, wait_until="networkidle", timeout=60000)
            except Exception as e:
                log.warning("goto error for %s: %s", tag, e)
            time.sleep(2 + random.random()*1.3)

            # Perform up to VISIT_SCROLLS scrolls for this keyword or until global cap / per-keyword max reached
            while visit_scrolls < VISIT_SCROLLS and scrolls_done[tag] < MAX_SCROLLS_PER_HASHTAG and total_written < GLOBAL_MAX_TWEETS:
                before_scroll = total_written
                articles = page.query_selector_all("article")
                for art in articles:
                    if total_written >= GLOBAL_MAX_TWEETS:
                        break
                    try:
                        url_el = art.query_selector("a[href*='/status/']")
                        tweet_url = None
                        try:
                            tweet_url = url_el.get_attribute("href") if url_el else None
                        except Exception:
                            tweet_url = None
                        if tweet_url and tweet_url.startswith("/"):
                            tweet_url = "https://x.com" + tweet_url
                        inner_text = art.inner_text().strip()
                        if not inner_text:
                            continue
                        dedupe_key = tweet_url or make_pseudo_id(inner_text[:400])
                        if dedupe_key in collected_global:
                            continue

                        # content heuristics
                        content = None
                        try:
                            el = art.query_selector('div[data-testid="tweetText"]')
                            if el:
                                content = el.inner_text().strip()
                            else:
                                el2 = art.query_selector('div[lang]')
                                if el2:
                                    content = el2.inner_text().strip()
                        except Exception:
                            content = None
                        if not content:
                            content = inner_text

                        username, ts_iso = extract_username_and_time(art)
                        replies, retweets, likes, quotes = extract_metrics_from_article(art)

                        mentions = ",".join(sorted(set(re.findall(r"@([A-Za-z0-9_]{1,15})", content or ""))))
                        hashtags_found = ",".join(sorted(set(h.lower() for h in re.findall(r"#(\w+)", content or ""))))

                        record = {
                            "id": dedupe_key,
                            "username": username or "",
                            "date": ts_iso or datetime.datetime.utcnow().isoformat(),
                            "content": (content or "").replace("\n", " "),
                            "replyCount": replies,
                            "retweetCount": retweets,
                            "likeCount": likes,
                            "quoteCount": quotes,
                            "mentions": mentions,
                            "hashtags": hashtags_found,
                            "url": tweet_url or "",
                            "source_hashtag": tag
                        }

                        writer.writerow(record)
                        total_written += 1
                        collected_this_visit += 1
                        per_tag_counts[tag] += 1
                        collected_global.add(dedupe_key)
                        if total_written % SAVE_EVERY == 0:
                            out_f.flush()
                        if total_written >= GLOBAL_MAX_TWEETS:
                            break
                    except Exception as e:
                        if random.random() < 0.02:
                            log.warning("Per-article parse error (ignored): %s", e)
                        continue

                after_scroll = total_written
                new_collected = after_scroll - before_scroll
                visit_scrolls += 1
                scrolls_done[tag] += 1

                log.info("Keyword %s: after visit-scroll %d (this visit), collected %d new tweets (total %d). Keyword total scrolls=%d",
                         tag, visit_scrolls, new_collected, total_written, scrolls_done[tag])

                # if we've hit GLOBAL_MAX_TWEETS break out
                if total_written >= GLOBAL_MAX_TWEETS:
                    break

                # scroll down to load more
                page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                time.sleep(SCROLL_PAUSE + random.random()*0.6)

            # After up to VISIT_SCROLLS (50) scrolls for this keyword, log a summary for the visit
            log.info("Finished visit for %s: collected %d tweets this visit, total collected %d, total keyword scrolls=%d",
                     tag, collected_this_visit, total_written, scrolls_done[tag])

            # Move to next keyword (round-robin). If we're at the end, wrap to first.
            keyword_index = (keyword_index + 1) % len(KEYWORDS)
            # small pause between keyword switches
            time.sleep(1 + random.random()*1.0)

        browser.close()
    out_f.flush()
    out_f.close()

    total = sum(per_tag_counts.values())
    log.info("=== Run summary ===")
    for tag in KEYWORDS:
        log.info("  %s : collected %d", tag, per_tag_counts.get(tag,0))
    log.info("TOTAL collected: %d (global cap %d)", total, GLOBAL_MAX_TWEETS)
    print("\n[+] Saved file:", OUT_CSV)
    print("[+] Total tweets extracted:", total)
    return total


if __name__ == "__main__":
    print("Using KEYWORDS:")
    for k in KEYWORDS:
        print(f"  {k}")
    print(f"Global cap (GLOBAL_MAX_TWEETS) = {GLOBAL_MAX_TWEETS}")
    run()
