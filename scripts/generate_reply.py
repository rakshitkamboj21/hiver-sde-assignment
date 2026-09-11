from pathlib import Path
import re

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TRAIN_DATA_PATH = Path("data/spotify_training_sample.csv")
MODEL_PATH = Path("models/spotify_intent_classifier.joblib")


def load_data():
    df = pd.read_csv(TRAIN_DATA_PATH)

    df = df.dropna(
        subset=["customer_text", "brand_text"]
    ).copy()

    df["customer_text"] = df["customer_text"].astype(str)
    df["brand_text"] = df["brand_text"].astype(str)

    from train_intent_classifier import assign_label

    df["intent"] = df["customer_text"].apply(assign_label)

    return df


def clean_historical_reply(text):
    text = str(text)

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Remove Twitter handles
    text = re.sub(r"@\w+", "", text)

    # Remove agent initials such as /LM, /AU, /JS
    text = re.sub(r"\s*/[A-Z]{2}\b", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def build_retriever(df):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    vectors = vectorizer.fit_transform(
        df["customer_text"]
    )

    return vectorizer, vectors


def retrieve_similar(
    query,
    intent,
    df,
    vectorizer,
    vectors,
    top_k=3,
):
    intent_df = df[df["intent"] == intent].copy()

    if len(intent_df) == 0:
        intent_df = df.copy()

    indices = intent_df.index

    filtered_vectors = vectors[indices]

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        filtered_vectors,
    )[0]

    top_positions = scores.argsort()[::-1][:top_k]

    results = intent_df.iloc[top_positions].copy()
    results["similarity"] = scores[top_positions]

    return results


def is_useful_reply(reply):
    """
    Identify historical replies that contain a useful
    troubleshooting step or concrete support action.
    """

    reply = reply.lower()

    vague_patterns = [
        "we've sent you a bit more info",
        "we have sent you a bit more info",
        "we'll carry on helping",
        "we will carry on helping",
        "more info over dm",
    ]

    for pattern in vague_patterns:
        if pattern in reply:
            return False

    useful_patterns = [
        "can you",
        "could you",
        "let us know",
        "send us",
        "dm us",
        "email",
        "device",
        "operating system",
        "os",
        "version",
        "restart",
        "browser",
        "incognito",
        "cache",
        "cookies",
        "error",
        "username",
    ]

    return any(
        pattern in reply
        for pattern in useful_patterns
    )


def select_best_reply(results):
    """
    Select the most useful historical resolution rather
    than blindly selecting the highest similarity result.
    """

    candidates = []

    for _, row in results.iterrows():
        reply = clean_historical_reply(row["brand_text"])

        if not reply:
            continue

        useful = is_useful_reply(reply)

        candidates.append(
            {
                "reply": reply,
                "similarity": row["similarity"],
                "useful": useful,
            }
        )

    if not candidates:
        return (
            "Hi! Sorry you're having trouble with Spotify. "
            "Could you share a few more details about the issue "
            "so we can look into it?"
        )

    # Prefer useful replies.
    useful_candidates = [
        item for item in candidates
        if item["useful"]
    ]

    if useful_candidates:
        return max(
            useful_candidates,
            key=lambda x: x["similarity"]
        )["reply"]

    # Otherwise use the most similar reply.
    return max(
        candidates,
        key=lambda x: x["similarity"]
    )["reply"]


def generate_reply(query, intent, results):
    return select_best_reply(results)


def main():
    df = load_data()

    model = joblib.load(MODEL_PATH)

    vectorizer, vectors = build_retriever(df)

    test_queries = [
        "Spotify keeps pausing while I listen to music",
        "I was charged for Premium but I still see adverts",
        "I can't log into my Spotify account",
        "My Spotify app keeps crashing on my phone",
    ]

    for query in test_queries:

        intent = model.predict([query])[0]

        results = retrieve_similar(
            query=query,
            intent=intent,
            df=df,
            vectorizer=vectorizer,
            vectors=vectors,
            top_k=3,
        )

        reply = generate_reply(
            query=query,
            intent=intent,
            results=results,
        )

        print("\n" + "=" * 80)
        print(f"CUSTOMER: {query}")
        print(f"INTENT:   {intent}")
        print("=" * 80)

        print("\nRetrieved historical replies:")

        for rank, (_, row) in enumerate(
            results.iterrows(),
            start=1,
        ):
            cleaned = clean_historical_reply(
                row["brand_text"]
            )

            print(f"\n--- Historical example {rank} ---")
            print(f"Similarity: {row['similarity']:.3f}")
            print(f"Customer: {row['customer_text']}")
            print(f"Historical reply: {cleaned}")

        print("\nGENERATED REPLY:")
        print(reply)


if __name__ == "__main__":
    main() 
