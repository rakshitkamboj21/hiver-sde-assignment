import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix,
)
import joblib


# --------------------------------------------------
# Paths
# --------------------------------------------------

TRAIN_DATA_PATH = Path("data/spotify_training_sample.csv")
GOLDEN_DATA_PATH = Path("data/spotify_golden_set.csv")

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "spotify_intent_classifier.joblib"


# --------------------------------------------------
# Load training data
# --------------------------------------------------

train_df = pd.read_csv(TRAIN_DATA_PATH)

print(f"Loaded {len(train_df)} historical conversations")


# --------------------------------------------------
# Weak labeling
# --------------------------------------------------
# These labels are generated using deterministic rules.
# They are NOT used as the final evaluation labels.
#
# The final evaluation uses the separately hand-labelled
# 200-example golden set.


def assign_label(text):
    text = str(text).lower().strip()

    # ---------------------------------------------
    # Account & Security
    # ---------------------------------------------
    if any(word in text for word in [
        "password",
        "forgot my password",
        "reset password",
        "login",
        "log in",
        "logged in",
        "can't log in",
        "cannot log in",
        "account hacked",
        "hacked account",
        "stolen account",
        "someone is hijacking",
    ]):
        return "Account & Security"


    # ---------------------------------------------
    # Subscription & Billing
    # ---------------------------------------------
    if any(word in text for word in [
        "premium",
        "subscription",
        "student discount",
        "student premium",
        "payment",
        "paypal",
        "charged",
        "charge",
        "refund",
        "billing",
        "renew",
        "price",
        "cost",
        "gift card",
        "family premium",
        "free trial",
    ]):
        return "Subscription & Billing"


    # ---------------------------------------------
    # Features & Product Feedback
    # ---------------------------------------------
    # Feature/product requests should take priority
    # over words such as playlist, library, song, etc.

    if any(phrase in text for phrase in [
        "feature request",
        "feature",
        "add a feature",
        "add more",
        "can you add",
        "please add",
        "would like",
        "i wish",
        "why not",
        "suggestion",
        "feedback",
        "request",
        "it feels hidden",
        "make it easier",
        "why is there a limit",
        "why not unlimited",
        "when will we be able to",
        "can we have",
        "could you add",
    ]):
        return "Features & Product Feedback"


    # ---------------------------------------------
    # Playlist, Library & Sync
    # ---------------------------------------------
    if any(word in text for word in [
        "playlist",
        "download",
        "downloaded",
        "saved",
        "my library",
        "library",
        "sync",
        "synchron",
        "songs disappeared",
        "songs missing from my library",
    ]):
        return "Playlist, Library & Sync"


    # ---------------------------------------------
    # Playback & Streaming
    # ---------------------------------------------
    if any(word in text for word in [
        "can't play",
        "cannot play",
        "can't playback",
        "won't play",
        "won’t play",
        "playing",
        "playback",
        "pause",
        "paused",
        "shuffle",
        "repeat",
        "buffer",
        "stream",
        "streaming",
        "skip",
        "skipping",
        "sound",
        "audio",
    ]):
        return "Playback & Streaming"


    # ---------------------------------------------
    # App & Web Player Technical
    # ---------------------------------------------
    if any(word in text for word in [
        "crash",
        "crashing",
        "browser",
        "web player",
        "cache",
        "cookies",
        "freeze",
        "frozen",
        "loading",
        "app",
        "desktop app",
        "ios",
        "android",
    ]):
        return "App & Web Player Technical"


    # ---------------------------------------------
    # Content Availability
    # ---------------------------------------------
    if any(word in text for word in [
        "not on spotify",
        "isn't on spotify",
        "isnt on spotify",
        "not available",
        "unavailable",
        "missing album",
        "missing song",
        "missing artist",
        "where is",
        "put this on spotify",
        "add this album",
        "add this song",
        "available in",
        "country",
        "region",
        "license",
        "licensing",
    ]):
        return "Content Availability"


    # ---------------------------------------------
    # Other / Unclear
    # ---------------------------------------------
    return "Other / Unclear"


train_df["intent"] = train_df["customer_text"].apply(assign_label)


print("\nWeak-label distribution:")
print(train_df["intent"].value_counts())


# --------------------------------------------------
# Prepare training data
# --------------------------------------------------

X_train = train_df["customer_text"]
y_train = train_df["intent"]


# --------------------------------------------------
# TF-IDF + Logistic Regression
# --------------------------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
    ),
])


# --------------------------------------------------
# Train
# --------------------------------------------------

print("\nTraining classifier...")

model.fit(X_train, y_train)

print("Training complete.")


# --------------------------------------------------
# Load hand-labelled golden set
# --------------------------------------------------

golden_df = pd.read_csv(GOLDEN_DATA_PATH)

print(f"\nLoaded {len(golden_df)} hand-labelled golden examples")


# --------------------------------------------------
# Golden-set evaluation
# --------------------------------------------------

X_gold = golden_df["customer_text"]
y_gold = golden_df["intent"]


# --------------------------------------------------
# Trivial baseline
# --------------------------------------------------
# Always predict the majority class from the golden set.

majority_class = y_gold.value_counts().idxmax()

baseline_predictions = [majority_class] * len(y_gold)

baseline_accuracy = accuracy_score(
    y_gold,
    baseline_predictions,
)

baseline_macro_f1 = f1_score(
    y_gold,
    baseline_predictions,
    average="macro",
    zero_division=0,
)

baseline_weighted_f1 = f1_score(
    y_gold,
    baseline_predictions,
    average="weighted",
    zero_division=0,
)

print("\n" + "=" * 60)
print("TRIVIAL BASELINE")
print("=" * 60)

print(f"\nMajority class: {majority_class}")
print(f"Accuracy:       {baseline_accuracy:.4f}")
print(f"Macro F1:       {baseline_macro_f1:.4f}")
print(f"Weighted F1:    {baseline_weighted_f1:.4f}")


# --------------------------------------------------
# TF-IDF + Logistic Regression evaluation
# --------------------------------------------------

predictions = model.predict(X_gold)

accuracy = accuracy_score(
    y_gold,
    predictions,
)

macro_f1 = f1_score(
    y_gold,
    predictions,
    average="macro",
    zero_division=0,
)

weighted_f1 = f1_score(
    y_gold,
    predictions,
    average="weighted",
    zero_division=0,
)


print("\n" + "=" * 60)
print("TF-IDF + LOGISTIC REGRESSION")
print("=" * 60)

print(f"\nAccuracy:       {accuracy:.4f}")
print(f"Macro F1:       {macro_f1:.4f}")
print(f"Weighted F1:    {weighted_f1:.4f}")


# --------------------------------------------------
# Per-class metrics
# --------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_gold,
        predictions,
        zero_division=0,
    )
)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

labels = sorted(golden_df["intent"].unique())

cm = confusion_matrix(
    y_gold,
    predictions,
    labels=labels,
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels,
)

print("\nConfusion Matrix:")
print(cm_df)

# --------------------------------------------------
# Error analysis
# --------------------------------------------------

error_df = golden_df.copy()

error_df["predicted_intent"] = predictions

errors = error_df[
    error_df["intent"] != error_df["predicted_intent"]
].copy()

print("\n" + "=" * 60)
print("ERROR ANALYSIS")
print("=" * 60)

print(f"\nTotal errors: {len(errors)} / {len(golden_df)}")

print("\nTop confusion pairs:")

confusion_pairs = (
    errors
    .groupby(["intent", "predicted_intent"])
    .size()
    .sort_values(ascending=False)
)

for (actual, predicted), count in confusion_pairs.head(10).items():
    print(f"{count:2d} | Actual: {actual} -> Predicted: {predicted}")


print("\nRepresentative misclassified examples:")

for i, (_, row) in enumerate(errors.head(20).iterrows(), start=1):

    print("\n" + "-" * 60)
    print(f"Example {i}")
    print(f"Tweet ID: {row['customer_tweet_id']}")
    print(f"Actual: {row['intent']}")
    print(f"Predicted: {row['predicted_intent']}")
    print(f"Customer: {row['customer_text']}")


# --------------------------------------------------
# Save model
# --------------------------------------------------

MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")