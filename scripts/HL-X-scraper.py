import snscrape.modules.twitter as sntwitter
import pandas as pd
import re
from datetime import datetime
import os
import certifi

os.environ['SSL_CERT_FILE'] = certifi.where()


# Tokens of interest
tokens = set([
    "BTC","ETH","ATOM","MATIC","DYDX","SOL","AVAX","BNB","APE","OP","LTC","ARB","DOGE","INJ",
    "SUI","kPEPE","CRV","LDO","LINK","STX","RNDR","CFX","FTM","GMX","SNX","XRP","BCH","APT",
    "AAVE","COMP","MKR","WLD","FXS","HPOS","RLB","UNIBOT","YGG","TRX","kSHIB","UNI","SEI",
    "RUNE","OX","FRIEND","SHIA","CYBER","ZRO","BLZ","DOT","BANANA","TRB","FTT","LOOM","OGN",
    "RDNT","ARK","BNT","CANTO","REQ","BIGTIME","KAS","ORBS","BLUR","TIA","BSV","ADA","TON",
    "MINA","POLYX","GAS","PENDLE","STG","FET","STRAX","NEAR","MEME","ORDI","BADGER","NEO",
    "ZEN","FIL","PYTH","SUSHI","ILV","IMX","kBONK","GMT","SUPER","USTC","NFTI","JUP","kLUNC",
    "RSR","GALA","JTO","NTRN","ACE","MAV","WIF","CAKE","PEOPLE","ENS","ETC","XAI","MANTA",
    "UMA","ONDO","ALT","ZETA","DYM","MAVIA","W","PANDORA","STRK","PIXEL","AI","TAO","AR",
    "MYRO","kFLOKI","BOME","ETHFI","ENA","MNT","TNSR","SAGA","MERL","HBAR","POPCAT","OMNI",
    "EIGEN","REZ","NOT","TURBO","BRETT","IO","ZK","BLAST","LISTA","MEW","RENDER","kDOGS",
    "POL","CATI","CELO","HMSTR","SCR","NEIROETH","kNEIRO","GOAT","MOODENG","GRASS","PURR",
    "PNUT","XLM","CHILLGUY","SAND","IOTA","ALGO","HYPE","ME","MOVE","VIRTUAL","PENGU",
    "USUAL","FARTCOIN","AI16Z","AIXBT","ZEREBRO","BIO","GRIFFAIN","SPX","S","MORPHO","TRUMP",
    "MELANIA","ANIME","VINE","VVV","JELLY","BERA","TST","LAYER","IP","OM","KAITO","NIL",
    "PAXG","PROMPT","BABY","WCT"
])

# Search tweets from HyperliquidX with "you can now long or short $"
query = 'from:HyperliquidX "you can now long or short $"'

results = []
for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    text = tweet.content
    date = tweet.date.date()
    match = re.search(r'long or short \$(\w+)', text)
    if match:
        token = match.group(1).upper()
        if token in tokens:
            results.append({
                'token': token,
                'listing_date': date,
                'tweet': text
            })

# Save to CSV
df = pd.DataFrame(results).drop_duplicates(subset="token")
df.to_csv("hyperliquid_listing_dates.csv", index=False)
print(df)
