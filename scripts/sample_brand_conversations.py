import pandas as pd
from collections import defaultdict

DATA_PATH = "data/raw/twcs.csv"

BRANDS = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "Delta",
]

CHUNK_SIZE = 100_000
EXAMPLES_PER_BRAND = 10


def main():
    # ---------------------------------------------------------
    # PASS 1:
    # Find brand replies and remember which customer tweet
    # each reply is responding to.
    # ---------------------------------------------------------

    brand_replies = defaultdict(list)
    parent_tweet_ids = set()

    print("Pass 1: collecting brand replies...")

    for chunk in pd.read_csv(
        DATA_PATH,
        chunksize=CHUNK_SIZE,
        usecols=[
            "tweet_id",
            "author_id",
            "inbound",
            "created_at",
            "text",
            "in_response_to_tweet_id",
        ],
    ):
        # Only outbound tweets from our candidate brands
        brand_rows = chunk[
            (chunk["inbound"] == False)
            & (chunk["author_id"].isin(BRANDS))
            & (chunk["in_response_to_tweet_id"].notna())
        ]

        for _, row in brand_rows.iterrows():
            brand = row["author_id"]

            parent_id = str(int(row["in_response_to_tweet_id"]))

            brand_replies[brand].append(
                {
                    "brand_tweet_id": str(int(row["tweet_id"])),
                    "parent_tweet_id": parent_id,
                    "brand_text": str(row["text"]),
                    "brand_created_at": row["created_at"],
                }
            )

            parent_tweet_ids.add(parent_id)

    print(f"Collected {len(parent_tweet_ids):,} customer tweet IDs.")
    print()

    # ---------------------------------------------------------
    # PASS 2:
    # Find the customer tweets that the brand replies were
    # responding to.
    # ---------------------------------------------------------

    customer_tweets = {}

    print("Pass 2: matching customer tweets...")

    for chunk in pd.read_csv(
        DATA_PATH,
        chunksize=CHUNK_SIZE,
        usecols=[
            "tweet_id",
            "author_id",
            "inbound",
            "created_at",
            "text",
        ],
    ):
        customer_rows = chunk[
            (chunk["inbound"] == True)
            & (chunk["tweet_id"].astype(str).isin(parent_tweet_ids))
        ]

        for _, row in customer_rows.iterrows():
            tweet_id = str(int(row["tweet_id"]))

            customer_tweets[tweet_id] = {
                "customer_tweet_id": tweet_id,
                "customer_id": str(row["author_id"]),
                "customer_text": str(row["text"]),
                "customer_created_at": row["created_at"],
            }

    print(f"Matched {len(customer_tweets):,} customer tweets.")
    print()

    # ---------------------------------------------------------
    # Build paired conversations
    # ---------------------------------------------------------

    paired = defaultdict(list)

    for brand, replies in brand_replies.items():
        for reply in replies:
            customer = customer_tweets.get(reply["parent_tweet_id"])

            if customer is None:
                continue

            paired[brand].append(
                {
                    **customer,
                    **reply,
                }
            )

    # ---------------------------------------------------------
    # Print examples
    # ---------------------------------------------------------

    print("=" * 80)
    print("TASK 2C - CUSTOMER → BRAND CONVERSATION SAMPLES")
    print("=" * 80)

    for brand in BRANDS:
        examples = paired[brand]

        print()
        print("#" * 80)
        print(f"BRAND: {brand}")
        print(f"PAIRED CONVERSATIONS: {len(examples):,}")
        print("#" * 80)

        for i, example in enumerate(
            examples[:EXAMPLES_PER_BRAND], start=1
        ):
            print()
            print(f"Example {i}")
            print("-" * 60)

            print("CUSTOMER:")
            print(example["customer_text"])

            print()
            print(f"{brand}:")
            print(example["brand_text"])

            print()
            print(
                f"Customer ID: {example['customer_id']} | "
                f"Date: {example['customer_created_at']}"
            )

    # ---------------------------------------------------------
    # Simple response-quality statistics
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("RESPONSE DIVERSITY SUMMARY")
    print("=" * 80)

    for brand in BRANDS:
        examples = paired[brand]

        if not examples:
            continue

        response_lengths = [
            len(x["brand_text"])
            for x in examples
            if x["brand_text"]
        ]

        generic_dm = sum(
            1
            for x in examples
            if "dm" in x["brand_text"].lower()
        )

        question_responses = sum(
            1
            for x in examples
            if "?" in x["brand_text"]
        )

        print()
        print(f"{brand}")
        print(f"  Paired conversations : {len(examples):,}")
        print(
            f"  Average reply length: "
            f"{sum(response_lengths) / len(response_lengths):.1f}"
        )
        print(
            f"  Median reply length : "
            f"{sorted(response_lengths)[len(response_lengths) // 2]}"
        )
        print(f"  Replies mentioning DM: {generic_dm:,}")
        print(f"  Replies containing ?: {question_responses:,}")


if __name__ == "__main__":
    main()