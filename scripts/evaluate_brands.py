import pandas as pd
from collections import defaultdict

FILE_PATH = "data/raw/twcs.csv"

CANDIDATE_BRANDS = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "Delta",
]

CHUNK_SIZE = 100_000


# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------

stats = {
    brand: {
        "brand_tweets": 0,
        "linked_replies": 0,
        "unique_customers": set(),
        "reply_lengths": [],
    }
    for brand in CANDIDATE_BRANDS
}


# ---------------------------------------------------------
# We first find which customer tweets each brand replied to
# ---------------------------------------------------------

parent_to_brand = defaultdict(list)

print("=" * 70)
print("HIVER - CORRECTED BRAND EVALUATION")
print("=" * 70)

print("\n[1] Finding brand replies and their customer messages...")

for chunk in pd.read_csv(
    FILE_PATH,
    chunksize=CHUNK_SIZE,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "text",
        "in_response_to_tweet_id",
    ],
):

    # Only brand/support messages
    brand_rows = chunk[
        (chunk["inbound"] == False)
        & (chunk["author_id"].isin(CANDIDATE_BRANDS))
    ]

    for _, row in brand_rows.iterrows():

        brand = row["author_id"]

        stats[brand]["brand_tweets"] += 1

        # Every brand reply that has a parent customer tweet
        if pd.notna(row["in_response_to_tweet_id"]):

            parent_id = str(
                int(row["in_response_to_tweet_id"])
            )

            stats[brand]["linked_replies"] += 1

            parent_to_brand[parent_id].append(
                brand
            )

            # Record reply length
            text = str(row["text"])

            stats[brand]["reply_lengths"].append(
                len(text)
            )


print(
    f"Customer tweet IDs referenced by brand replies: "
    f"{len(parent_to_brand):,}"
)


# ---------------------------------------------------------
# Find the corresponding customer tweets
# ---------------------------------------------------------

print("\n[2] Matching customer tweets...")

customer_messages_found = 0

# Keep only examples for later inspection
examples = {
    brand: []
    for brand in CANDIDATE_BRANDS
}

for chunk in pd.read_csv(
    FILE_PATH,
    chunksize=CHUNK_SIZE,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "text",
    ],
):

    inbound_rows = chunk[
        chunk["inbound"] == True
    ]

    for _, row in inbound_rows.iterrows():

        tweet_id = str(row["tweet_id"])

        if tweet_id not in parent_to_brand:
            continue

        customer_messages_found += 1

        customer_id = str(row["author_id"])
        customer_text = str(row["text"])

        brands = parent_to_brand[tweet_id]

        for brand in brands:

            stats[brand]["unique_customers"].add(
                customer_id
            )

            # Save a few real examples
            if len(examples[brand]) < 10:

                examples[brand].append({
                    "customer": customer_text,
                    "customer_id": customer_id,
                })


print(
    f"Matched customer tweets: "
    f"{customer_messages_found:,}"
)


# ---------------------------------------------------------
# Print brand comparison
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("BRAND COMPARISON")
print("=" * 70)

for brand in CANDIDATE_BRANDS:

    s = stats[brand]

    reply_lengths = s["reply_lengths"]

    average_length = (
        sum(reply_lengths) / len(reply_lengths)
        if reply_lengths
        else 0
    )

    median_length = (
        sorted(reply_lengths)[len(reply_lengths) // 2]
        if reply_lengths
        else 0
    )

    print(f"\n{brand}")
    print("-" * 50)

    print(
        f"Brand tweets: "
        f"{s['brand_tweets']:,}"
    )

    print(
        f"Linked customer conversations: "
        f"{s['linked_replies']:,}"
    )

    print(
        f"Unique customers: "
        f"{len(s['unique_customers']):,}"
    )

    print(
        f"Average reply length: "
        f"{average_length:.1f} characters"
    )

    print(
        f"Median reply length: "
        f"{median_length} characters"
    )


# ---------------------------------------------------------
# Print real customer examples
# ---------------------------------------------------------

print("\n\n" + "=" * 70)
print("REAL CUSTOMER MESSAGE EXAMPLES")
print("=" * 70)

for brand in CANDIDATE_BRANDS:

    print(f"\n\n### {brand}")
    print("-" * 70)

    for i, example in enumerate(
        examples[brand][:5],
        start=1
    ):

        print(f"\nExample {i}")
        print("CUSTOMER:")
        print(example["customer"])


print("\n" + "=" * 70)
print("CORRECTED BRAND EVALUATION COMPLETE")
print("=" * 70)