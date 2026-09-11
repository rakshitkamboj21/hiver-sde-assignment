import pandas as pd
from collections import defaultdict

DATA_PATH = "data/raw/twcs.csv"

BRAND = "SpotifyCares"
CHUNK_SIZE = 100_000

# Number of paired conversations to collect
MAX_EXAMPLES = 1000


def main():
    # ---------------------------------------------------------
    # PASS 1: Find SpotifyCares replies
    # ---------------------------------------------------------

    print("Pass 1: collecting SpotifyCares replies...")

    spotify_replies = []
    parent_ids = set()

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
        rows = chunk[
            (chunk["inbound"] == False)
            & (chunk["author_id"] == BRAND)
            & (chunk["in_response_to_tweet_id"].notna())
        ]

        for _, row in rows.iterrows():
            parent_id = str(int(row["in_response_to_tweet_id"]))

            spotify_replies.append(
                {
                    "brand_tweet_id": str(int(row["tweet_id"])),
                    "parent_tweet_id": parent_id,
                    "brand_text": str(row["text"]),
                    "brand_created_at": row["created_at"],
                }
            )

            parent_ids.add(parent_id)

    print(f"SpotifyCares replies found: {len(spotify_replies):,}")
    print(f"Customer tweet IDs needed: {len(parent_ids):,}")

    # ---------------------------------------------------------
    # PASS 2: Find corresponding customer tweets
    # ---------------------------------------------------------

    print("\nPass 2: matching customer tweets...")

    customers = {}

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
        rows = chunk[
            (chunk["inbound"] == True)
            & (chunk["tweet_id"].astype(str).isin(parent_ids))
        ]

        for _, row in rows.iterrows():
            tweet_id = str(int(row["tweet_id"]))

            customers[tweet_id] = {
                "customer_tweet_id": tweet_id,
                "customer_id": str(row["author_id"]),
                "customer_text": str(row["text"]),
                "customer_created_at": row["created_at"],
            }

    print(f"Customer tweets matched: {len(customers):,}")

    # ---------------------------------------------------------
    # PASS 3: Build customer -> Spotify pairs
    # ---------------------------------------------------------

    conversations = []

    for reply in spotify_replies:
        customer = customers.get(reply["parent_tweet_id"])

        if customer is None:
            continue

        conversations.append(
            {
                **customer,
                **reply,
            }
        )

    print(f"Paired conversations: {len(conversations):,}")

    # ---------------------------------------------------------
    # Save a sample for analysis
    # ---------------------------------------------------------

    sample = conversations[:MAX_EXAMPLES]

    output_path = "data/spotify_intent_sample.csv"

    pd.DataFrame(sample).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nSaved {len(sample):,} conversations to:")
    print(output_path)

    # ---------------------------------------------------------
    # Print conversations to terminal
    # ---------------------------------------------------------

    print("\n" + "=" * 90)
    print("SPOTIFYCARES INTENT ANALYSIS SAMPLE")
    print("=" * 90)

    for i, conversation in enumerate(sample, start=1):

        print(f"\n{'-' * 90}")
        print(f"EXAMPLE {i}")
        print(f"{'-' * 90}")

        print("\nCUSTOMER:")
        print(conversation["customer_text"])

        print("\nSPOTIFYCARES:")
        print(conversation["brand_text"])

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":
    main()