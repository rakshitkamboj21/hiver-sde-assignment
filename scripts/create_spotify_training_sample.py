import pandas as pd
from pathlib import Path


DATA_PATH = Path("data/raw/twcs.csv")
OUTPUT_PATH = Path("data/spotify_training_sample.csv")

BRAND = "SpotifyCares"

CHUNK_SIZE = 100_000
TARGET_EXAMPLES = 5_000


def main():

    # --------------------------------------------------
    # PASS 1:
    # Collect Spotify brand replies and their parent IDs
    # --------------------------------------------------

    brand_replies = []

    print("Pass 1: collecting Spotify replies...")

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

        brand_rows = chunk[
            (chunk["inbound"] == False)
            & (chunk["author_id"] == BRAND)
            & (chunk["in_response_to_tweet_id"].notna())
        ]

        for _, row in brand_rows.iterrows():

            brand_replies.append(
                {
                    "brand_tweet_id": str(int(row["tweet_id"])),
                    "parent_tweet_id": str(
                        int(row["in_response_to_tweet_id"])
                    ),
                    "brand_text": str(row["text"]),
                    "brand_created_at": row["created_at"],
                }
            )

    print(f"Spotify replies collected: {len(brand_replies):,}")

    parent_ids = {
        x["parent_tweet_id"]
        for x in brand_replies
    }

    print(f"Unique customer tweet IDs: {len(parent_ids):,}")


    # --------------------------------------------------
    # PASS 2:
    # Find matching customer tweets
    # --------------------------------------------------

    customer_tweets = {}

    print("\nPass 2: matching customer tweets...")

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
            & (chunk["tweet_id"].astype(str).isin(parent_ids))
        ]

        for _, row in customer_rows.iterrows():

            tweet_id = str(int(row["tweet_id"]))

            customer_tweets[tweet_id] = {
                "customer_tweet_id": tweet_id,
                "customer_id": str(row["author_id"]),
                "customer_text": str(row["text"]),
                "customer_created_at": row["created_at"],
            }

    print(
        f"Matched customer tweets: "
        f"{len(customer_tweets):,}"
    )


    # --------------------------------------------------
    # Build paired conversations
    # --------------------------------------------------

    paired = []

    for reply in brand_replies:

        customer = customer_tweets.get(
            reply["parent_tweet_id"]
        )

        if customer is None:
            continue

        paired.append(
            {
                **customer,
                **reply,
            }
        )


    print(
        f"Paired Spotify conversations: "
        f"{len(paired):,}"
    )


    # --------------------------------------------------
    # Select training sample
    # --------------------------------------------------

    df = pd.DataFrame(paired)

    if len(df) > TARGET_EXAMPLES:

        df = df.sample(
            n=TARGET_EXAMPLES,
            random_state=42,
        )

    df = df.reset_index(drop=True)


    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("=" * 60)
    print("TRAINING SAMPLE CREATED")
    print("=" * 60)

    print(f"Rows saved: {len(df):,}")
    print(f"File: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()