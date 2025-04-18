import os
import re
import csv
import tweepy

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
BEARER_TOKEN = os.getenv("BEARER_TOKEN", "AAAAAAAAAAAAAAAAAAAAAAyt0gEAAAAAgfPg%2BMNlXgJkBf0E%2B36RemaqewU%3DQj2ZXkEPQ4cFo4UkjLC1QKSItd2JIqBjnjslxKarQ5draKRskA")
USERNAME     = "HyperliquidX"

LISTING_RE = re.compile(
    r"By community request, you can now long or short \$([A-Za-z0-9]+)",
    re.IGNORECASE
)

# ─── SETUP CLIENT ────────────────────────────────────────────────────────────────
client = tweepy.Client(
    bearer_token=BEARER_TOKEN,
    wait_on_rate_limit=True,
)

# ─── RESOLVE USER ID ─────────────────────────────────────────────────────────────
user_resp = client.get_user(username=USERNAME)
if not user_resp.data:
    raise SystemExit(f" Couldn’t find user @{USERNAME}")
USER_ID = user_resp.data.id

# ─── SCRAPE & PARSE ──────────────────────────────────────────────────────────────
listings = {}  # token -> earliest YYYY‑MM‑DD

for resp in tweepy.Paginator(
    client.get_users_tweets,
    id=USER_ID,
    tweet_fields=["created_at", "text"],
    max_results=1
):
    if not resp.data:
        continue
    for tweet in resp.data:
        m = LISTING_RE.search(tweet.text)
        if m:
            token = m.group(1).upper()
            date  = tweet.created_at.date().isoformat()
            if token not in listings or date < listings[token]:
                listings[token] = date

# ─── OUTPUT TO CSV ───────────────────────────────────────────────────────────────
with open("listings.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["token", "date"])
    for token in sorted(listings):
        writer.writerow([token, listings[token]])

print(f"Found {len(listings)} listings – saved to listings.csv")
