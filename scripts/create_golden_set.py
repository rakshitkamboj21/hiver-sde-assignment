import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------

INPUT_PATH = Path("data/spotify_intent_sample.csv")
OUTPUT_PATH = Path("data/spotify_golden_set.csv")

# Number of examples to manually label
N_SAMPLES = 200

# Fixed seed makes the sample reproducible
RANDOM_STATE = 42


# --------------------------------------------------
# Load conversations
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print(f"Loaded {len(df)} conversations.")


# --------------------------------------------------
# Select a reproducible sample
# --------------------------------------------------

sample = df.sample(
    n=min(N_SAMPLES, len(df)),
    random_state=RANDOM_STATE
).copy()


# --------------------------------------------------
# Keep only the fields needed for labeling
# --------------------------------------------------

golden = sample[
    [
        "customer_tweet_id",
        "customer_text"
    ]
].copy()


# --------------------------------------------------
# Add blank intent column
# --------------------------------------------------

golden["intent"] = ""


# --------------------------------------------------
# Save labeling template
# --------------------------------------------------

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

golden.to_csv(
    OUTPUT_PATH,
    index=False
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print(f"\nCreated golden-set labeling file:")
print(OUTPUT_PATH)

print(f"\nNumber of examples: {len(golden)}")

print("\nColumns:")
print(golden.columns.tolist())

print("\nFirst 10 examples:")
print(golden.head(10).to_string(index=False))

print("\nNext step:")
print("Manually assign one of the 8 intents to every example.")