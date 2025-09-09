# save_x_session.py
"""
Interactive helper: open browser, let you sign in to X, then save storage state to 'state.json'.
Run this once in a terminal: python save_x_session.py
"""

from playwright.sync_api import sync_playwright
import time, sys, os

OUT = "state.json"
BROWSER = "chromium"   # can be "chromium", "firefox", "webkit"

def main():
    print("Launching browser. Please sign in to X in the opened window.")
    print("After login completes (and you finish any 2FA), return to this terminal and press Enter.")
    with sync_playwright() as p:
        browser = getattr(p, BROWSER).launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://x.com/login", wait_until="domcontentloaded", timeout=60000)
        # optional helpful message in page
        try:
            print("Browser opened. If a popup appears (like cookies), handle it. Then sign in to X.")
            input("Press Enter in this terminal when you have signed in and are seeing the X home/search page...")
        except KeyboardInterrupt:
            print("\nAborted by user.")
            browser.close()
            sys.exit(1)
        # small delay to let JS settle
        time.sleep(2)
        # save storage state (cookies + localStorage)
        context.storage_state(path=OUT)
        print(f"Saved authenticated state to: {OUT}")
        browser.close()

if __name__ == "__main__":
    main()
