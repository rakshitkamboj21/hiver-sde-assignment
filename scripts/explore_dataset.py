import pandas as pd

FILE_PATH = "data/raw/twcs.csv"

print("=" * 60)
print("HIVER SDE ASSIGNMENT - DATASET EXPLORATION")
print("=" * 60)

# ---------------------------------------------------------
# 1. Read the first small portion to inspect the structure
# ---------------------------------------------------------

print("\n[1] Reading dataset structure...")

sample = pd.read_csv(FILE_PATH, nrows=10)

print("\nColumns:")
for column in sample.columns:
    print(" -", column)

print("\nFirst 5 rows:")
print(sample.head().to_string())

# ---------------------------------------------------------
# 2. Count total rows efficiently
# ---------------------------------------------------------

print("\n[2] Counting tweets...")

total_rows = 0

for chunk in pd.read_csv(FILE_PATH, chunksize=100_000):
    total_rows += len(chunk)

print(f"Total tweets: {total_rows:,}")

# ---------------------------------------------------------
# 3. Analyze inbound / outbound tweets
# ---------------------------------------------------------

print("\n[3] Analyzing customer vs brand messages...")

inbound_count = 0
outbound_count = 0

# Counts of authors for inbound/outbound tweets
inbound_authors = {}
outbound_authors = {}

for chunk in pd.read_csv(
    FILE_PATH,
    chunksize=100_000,
    usecols=[
        "author_id",
        "inbound",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    ],
):

    # Customer messages
    inbound = chunk[chunk["inbound"] == True]

    # Brand/support messages
    outbound = chunk[chunk["inbound"] == False]

    inbound_count += len(inbound)
    outbound_count += len(outbound)

    # Count authors
    for author in inbound["author_id"].dropna():
        author = str(author)
        inbound_authors[author] = inbound_authors.get(author, 0) + 1

    for author in outbound["author_id"].dropna():
        author = str(author)
        outbound_authors[author] = outbound_authors.get(author, 0) + 1


print(f"Customer/inbound tweets : {inbound_count:,}")
print(f"Brand/outbound tweets   : {outbound_count:,}")

# ---------------------------------------------------------
# 4. Show most active outbound accounts
# ---------------------------------------------------------

print("\n[4] Top accounts sending brand/support replies:")

top_outbound = sorted(
    outbound_authors.items(),
    key=lambda x: x[1],
    reverse=True
)

for author, count in top_outbound[:30]:
    print(f"{author:25} {count:,} tweets")

# ---------------------------------------------------------
# 5. Show most active customer accounts
# ---------------------------------------------------------

print("\n[5] Top customer accounts:")

top_inbound = sorted(
    inbound_authors.items(),
    key=lambda x: x[1],
    reverse=True
)

for author, count in top_inbound[:20]:
    print(f"{author:25} {count:,} tweets")

# ---------------------------------------------------------
# 6. Basic conversation statistics
# ---------------------------------------------------------

print("\n[6] Conversation relationships...")

response_count = 0
reply_to_count = 0

for chunk in pd.read_csv(
    FILE_PATH,
    chunksize=100_000,
    usecols=[
        "response_tweet_id",
        "in_response_to_tweet_id",
    ],
):

    response_count += chunk["response_tweet_id"].notna().sum()
    reply_to_count += chunk["in_response_to_tweet_id"].notna().sum()

print(f"Tweets with response_tweet_id       : {response_count:,}")
print(f"Tweets with in_response_to_tweet_id : {reply_to_count:,}")

print("\n" + "=" * 60)
print("DATASET EXPLORATION COMPLETE")
print("=" * 60)