import json
import os
import re
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "spotify_training_sample.csv"
MODEL_PATH = BASE_DIR / "models" / "spotify_intent_classifier.joblib"

TOP_K = 3
MODEL_NAME = "gemini-3.6-flash"


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found.\n"
        "Please add GEMINI_API_KEY=your_key to the .env file."
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

print("Creating Gemini client...")

client = genai.Client(api_key=API_KEY)


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

def load_data():
    """Load and validate historical Spotify conversations."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "customer_text",
        "brand_text",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df.dropna(
        subset=["customer_text", "brand_text"]
    ).copy()

    df["customer_text"] = (
        df["customer_text"]
        .astype(str)
        .str.strip()
    )

    df["brand_text"] = (
        df["brand_text"]
        .astype(str)
        .str.strip()
    )

    return df


# ============================================================
# HISTORICAL INTENT LABELING
# ============================================================

def assign_historical_intents(df):
    """
    Assign the same weak labels used by the intent classifier
    to historical conversations.
    """

    # Import here so this script can still load independently.
    import sys

    scripts_dir = str(Path(__file__).resolve().parent)

    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from train_intent_classifier import assign_label

    df = df.copy()

    print("Assigning historical intents...")

    df["intent"] = (
        df["customer_text"]
        .apply(assign_label)
    )

    return df


# ============================================================
# RETRIEVAL INDEX
# ============================================================

def build_retrieval_index(df):
    """Build TF-IDF index over customer messages."""

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    customer_matrix = vectorizer.fit_transform(
        df["customer_text"]
    )

    return vectorizer, customer_matrix


# ============================================================
# INTENT CLASSIFICATION
# ============================================================

def load_intent_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Intent classifier not found:\n{MODEL_PATH}\n\n"
            "Run train_intent_classifier.py first."
        )

    return joblib.load(MODEL_PATH)


def predict_intent(model, customer_text):
    """Predict intent for an incoming customer message."""

    return model.predict(
        [customer_text]
    )[0]


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_historical_reply(text):
    """
    Remove Twitter-specific information from historical replies.
    """

    text = str(text)

    # Remove URLs
    text = re.sub(
        r"https?://\S+",
        "",
        text
    )

    # Remove Twitter handles
    text = re.sub(
        r"@\w+",
        "",
        text
    )

    # Remove agent initials such as /LM, /AU
    text = re.sub(
        r"\s*/[A-Z]{2}\b",
        "",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# SIMILARITY RETRIEVAL
# ============================================================

def retrieve_similar(
    query,
    intent,
    df,
    vectorizer,
    customer_matrix,
    top_k=TOP_K,
):
    """
    Retrieve the most similar historical conversations.

    Retrieval is first restricted to the predicted intent.
    """

    intent_df = df[
        df["intent"] == intent
    ].copy()

    # Fallback if the predicted intent has no historical examples.
    if len(intent_df) == 0:
        intent_df = df.copy()

    intent_indices = intent_df.index.tolist()

    filtered_matrix = customer_matrix[
        intent_indices
    ]

    query_vector = vectorizer.transform(
        [query]
    )

    scores = cosine_similarity(
        query_vector,
        filtered_matrix,
    )[0]

    top_positions = (
        scores.argsort()[::-1][:top_k]
    )

    results = intent_df.iloc[
        top_positions
    ].copy()

    results["similarity"] = (
        scores[top_positions]
    )

    return results


# ============================================================
# GEMINI PROMPT
# ============================================================

def build_prompt(
    customer_text,
    intent,
    examples,
):
    """Build the grounded Gemini prompt."""

    historical_examples = []

    for i, (_, row) in enumerate(
        examples.iterrows(),
        start=1,
    ):

        historical_examples.append(
            f"""
Historical Example {i}

Customer:
{row["customer_text"]}

Spotify response:
{clean_historical_reply(row["brand_text"])}

Similarity:
{row["similarity"]:.3f}
"""
        )

    examples_text = "\n".join(
        historical_examples
    )

    prompt = f"""
You are a professional Spotify customer support agent.

Your job is to analyze the customer message and produce:

1. A concise customer-facing support reply.
2. A decision:
   - AUTO-HANDLE
   - ESCALATE
3. A short reason for that decision.

CUSTOMER MESSAGE:
{customer_text}

PREDICTED INTENT:
{intent}

SIMILAR HISTORICAL SPOTIFY CONVERSATIONS:
{examples_text}

IMPORTANT RULES:

- Ground the response in the historical examples.
- Use the historical responses as guidance, not as text to copy.
- Respond directly to the customer's problem.
- Be polite, professional, and concise.
- Do not mention historical examples.
- Do not mention that you are an AI.
- Do not invent account information.
- Do not claim that you performed an action.
- Do not invent URLs.
- Do not include Twitter handles.
- Do not include internal agent initials.
- Do not copy historical responses word-for-word.
- If account-specific information is required, prefer ESCALATE.
- If the issue requires checking billing, account status,
  authentication, personal information, or another backend
  system that the support agent would need access to,
  prefer ESCALATE.
- Common troubleshooting questions that can be answered from
  the available historical support behavior can be AUTO-HANDLE.
- The reason must be specific to this customer's issue.
- Keep the customer-facing reply concise.

Return ONLY valid JSON in exactly this format:

{{
    "reply": "customer-facing response",
    "decision": "AUTO-HANDLE or ESCALATE",
    "reason": "short explanation"
}}
"""

    return prompt


# ============================================================
# GEMINI RESPONSE PARSING
# ============================================================

def parse_gemini_response(response_text):
    """
    Safely parse Gemini's JSON response.
    """

    text = response_text.strip()

    # Remove markdown code fences if Gemini adds them.
    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Gemini returned invalid JSON:\n"
            f"{response_text}"
        ) from exc

    required_fields = [
        "reply",
        "decision",
        "reason",
    ]

    missing = [
        field
        for field in required_fields
        if field not in data
    ]

    if missing:
        raise ValueError(
            f"Gemini response is missing fields: {missing}"
        )

    decision = str(
        data["decision"]
    ).strip().upper()

    if decision not in {
        "AUTO-HANDLE",
        "ESCALATE",
    }:
        raise ValueError(
            f"Invalid escalation decision: {decision}"
        )

    return {
        "reply": str(
            data["reply"]
        ).strip(),

        "decision": decision,

        "reason": str(
            data["reason"]
        ).strip(),
    }


# ============================================================
# GENERATE RESPONSE
# ============================================================

def generate_reply(
    customer_text,
    intent_model,
    df,
    vectorizer,
    customer_matrix,
):
    """
    Complete pipeline:

    customer message
        ↓
    intent classification
        ↓
    similarity retrieval
        ↓
    Gemini grounded generation
        ↓
    reply + decision + reason
    """

    # --------------------------------------------------------
    # 1. Predict intent
    # --------------------------------------------------------

    intent = predict_intent(
        intent_model,
        customer_text,
    )

    # --------------------------------------------------------
    # 2. Retrieve similar conversations
    # --------------------------------------------------------

    examples = retrieve_similar(
        query=customer_text,
        intent=intent,
        df=df,
        vectorizer=vectorizer,
        customer_matrix=customer_matrix,
        top_k=TOP_K,
    )

    # --------------------------------------------------------
    # 3. Build grounded prompt
    # --------------------------------------------------------

    prompt = build_prompt(
        customer_text,
        intent,
        examples,
    )

    # --------------------------------------------------------
    # 4. Gemini
    # --------------------------------------------------------

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    if not response.text:
        raise ValueError(
            "Gemini returned an empty response."
        )

    generated = parse_gemini_response(
        response.text
    )

    return (
        intent,
        examples,
        generated,
    )


# ============================================================
# DISPLAY
# ============================================================

def display_result(
    customer_text,
    intent_model,
    df,
    vectorizer,
    customer_matrix,
):
    """Display one complete pipeline result."""

    print()
    print("=" * 80)
    print("CUSTOMER:")
    print(customer_text)

    try:

        intent, examples, generated = (
            generate_reply(
                customer_text,
                intent_model,
                df,
                vectorizer,
                customer_matrix,
            )
        )

        # ----------------------------------------------------
        # Intent
        # ----------------------------------------------------

        print()
        print("PREDICTED INTENT:")
        print(intent)

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        print()
        print(
            "RETRIEVED HISTORICAL EXAMPLES:"
        )

        for i, (_, row) in enumerate(
            examples.iterrows(),
            start=1,
        ):

            print()
            print(
                f"{i}. Similarity: "
                f"{row['similarity']:.3f}"
            )

            print(
                f"Customer: "
                f"{row['customer_text']}"
            )

            print(
                f"Spotify: "
                f"{row['brand_text']}"
            )

        # ----------------------------------------------------
        # Reply
        # ----------------------------------------------------

        print()
        print("GEMINI REPLY:")
        print(
            generated["reply"]
        )

        # ----------------------------------------------------
        # Decision
        # ----------------------------------------------------

        print()
        print("HANDLING DECISION:")
        print(
            generated["decision"]
        )

        # ----------------------------------------------------
        # Reason
        # ----------------------------------------------------

        print()
        print("DECISION REASON:")
        print(
            generated["reason"]
        )

    except Exception as exc:

        print()
        print("ERROR:")
        print(
            type(exc).__name__
        )
        print(exc)


# ============================================================
# TEST QUERIES
# ============================================================

TEST_QUERIES = [

    "Spotify keeps pausing while I listen to music.",

    "I was charged for Premium but I still see adverts.",

    "I can't log into my Spotify account.",

    "Spotify keeps crashing when I open the app.",

]


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading intent classifier...")

    intent_model = load_intent_model()

    print("Loading historical Spotify conversations...")

    df = load_data()

    print(
        f"Loaded {len(df):,} historical conversations"
    )

    df = assign_historical_intents(df)

    print("Building retrieval index...")

    vectorizer, customer_matrix = (
        build_retrieval_index(df)
    )

    print("Retrieval index ready.")

    for query in TEST_QUERIES:

        display_result(
            query,
            intent_model,
            df,
            vectorizer,
            customer_matrix,
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()