from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from train_intent_classifier import assign_label


TRAIN_DATA_PATH = Path("data/spotify_training_sample.csv")
MODEL_PATH = Path("models/spotify_intent_classifier.joblib")


def load_data():
    df = pd.read_csv(TRAIN_DATA_PATH)

    required_columns = [
        "customer_text",
        "brand_text",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["customer_text", "brand_text"]).copy()

    df["customer_text"] = df["customer_text"].astype(str)
    df["brand_text"] = df["brand_text"].astype(str)

    return df


def load_classifier():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Classifier model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def build_retriever(df):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    customer_vectors = vectorizer.fit_transform(
        df["customer_text"]
    )

    return vectorizer, customer_vectors


def predict_intent(model, query):
    return model.predict([query])[0]


def retrieve_similar(
    query,
    intent,
    df,
    vectorizer,
    customer_vectors,
    top_k=3,
):
    # First restrict historical examples to the predicted intent.
    intent_df = df[df["intent"] == intent].copy()

    # If there are no examples for that intent, fall back to all examples.
    if len(intent_df) == 0:
        intent_df = df.copy()

    intent_indices = intent_df.index

    filtered_vectors = customer_vectors[intent_indices]

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        filtered_vectors,
    )[0]

    top_positions = scores.argsort()[::-1][:top_k]

    results = intent_df.iloc[top_positions].copy()
    results["similarity"] = scores[top_positions]

    return results


def main():
    df = load_data()
    model = load_classifier()

    print(f"Loaded {len(df):,} historical Spotify conversations.")

    # The training CSV currently contains customer_text and brand_text.
    # We need the same weak-labeling logic used during classifier training.
    # The saved classifier predicts the intent for the incoming query,
    # while historical conversations need labels for filtering.

    from train_intent_classifier import assign_label

    df["intent"] = df["customer_text"].apply(assign_label)

    vectorizer, customer_vectors = build_retriever(df)

    test_queries = [
        "Spotify keeps pausing while I listen to music",
        "I was charged for Premium but I still see adverts",
        "I can't log into my Spotify account",
    ]

    for query in test_queries:
        intent = predict_intent(model, query)

        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print(f"PREDICTED INTENT: {intent}")
        print("=" * 80)

        results = retrieve_similar(
            query,
            intent,
            df,
            vectorizer,
            customer_vectors,
            top_k=3,
        )

        for rank, (_, row) in enumerate(
            results.iterrows(),
            start=1,
        ):
            print(f"\n--- Result {rank} ---")
            print(f"Similarity: {row['similarity']:.3f}")
            print(f"Intent:     {row['intent']}")
            print(f"Customer:   {row['customer_text']}")
            print(f"Spotify:    {row['brand_text']}")


if __name__ == "__main__":
    main()