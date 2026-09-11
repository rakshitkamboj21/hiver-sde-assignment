import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. "
        "Make sure it is present in your .env file."
    )

# Keep the same model used by the rest of the project.
MODEL_NAME = "gemini-3.6-flash"

GOLDEN_SET_PATH = Path(
    "data/spotify_golden_set.csv"
)

CALIBRATION_PATH = Path(
    "data/reply_judge_calibration.csv"
)

# Keep this small because the Gemini free tier has a
# limited request quota.
#
# Each calibration example uses:
#   1 request -> generate reply
#   1 request -> judge reply
#
# 8 examples = up to 16 Gemini requests.
CALIBRATION_SIZE = 8

# Retry only temporary service errors.
# Do NOT retry 429 quota exhaustion.
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 10


# ============================================================
# GEMINI CLIENT
# ============================================================

print("Creating Gemini client...")

client = genai.Client(
    api_key=API_KEY
)


# ============================================================
# GEMINI REQUEST
# ============================================================

def call_gemini(prompt):
    """
    Call Gemini.

    503 / UNAVAILABLE:
        Retry with exponential backoff.

    429 / RESOURCE_EXHAUSTED:
        Do NOT retry. The API quota has been exhausted and
        waiting a few seconds will not solve a daily quota limit.
    """

    delay = INITIAL_RETRY_DELAY

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            print(
                f"Calling Gemini "
                f"(attempt {attempt}/{MAX_RETRIES})..."
            )

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )

            if not response.text:
                raise ValueError(
                    "Gemini returned an empty response."
                )

            return response.text.strip()

        except Exception as e:

            error_text = str(e)

            # ------------------------------------------------
            # QUOTA EXHAUSTED
            # ------------------------------------------------
            #
            # Do NOT retry 429.
            #
            # The Gemini API may say something such as:
            #
            # 429 RESOURCE_EXHAUSTED
            # You exceeded your current quota
            #
            # Retrying immediately only wastes requests.
            # ------------------------------------------------

            if (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
            ):

                print()
                print(
                    "Gemini API quota exhausted (429)."
                )

                print(
                    "This is a quota limitation, "
                    "not a code error."
                )

                print(
                    "No additional retry will be attempted."
                )

                raise

            # ------------------------------------------------
            # TEMPORARY SERVICE ERROR
            # ------------------------------------------------

            temporary_error = (
                "503" in error_text
                or "UNAVAILABLE" in error_text
            )

            if not temporary_error:
                raise

            if attempt == MAX_RETRIES:
                print()
                print(
                    "Gemini remained unavailable after "
                    f"{MAX_RETRIES} attempts."
                )
                raise

            print()
            print(
                f"Temporary Gemini service error "
                f"(attempt {attempt}/{MAX_RETRIES})."
            )

            print(
                f"Waiting {delay} seconds before retry..."
            )

            time.sleep(delay)

            delay = min(
                delay * 2,
                120
            )

    raise RuntimeError(
        "Gemini request failed after retries."
    )


# ============================================================
# LOAD GOLDEN SET
# ============================================================

def load_golden_set():

    if not GOLDEN_SET_PATH.exists():
        raise FileNotFoundError(
            f"Golden set not found: {GOLDEN_SET_PATH}"
        )

    df = pd.read_csv(
        GOLDEN_SET_PATH
    )

    possible_text_columns = [
        "customer_text",
        "customer",
        "text",
        "tweet",
        "message",
    ]

    text_column = None

    for column in possible_text_columns:

        if column in df.columns:
            text_column = column
            break

    if text_column is None:
        raise ValueError(
            "Could not find customer text column. "
            f"Available columns: {list(df.columns)}"
        )

    return df, text_column


# ============================================================
# REPLY QUALITY RUBRIC
# ============================================================

RUBRIC = """
Evaluate the quality of a customer-support reply on a 1-5 scale.

5 = Excellent
- Directly addresses the customer's problem.
- Helpful and actionable.
- Appropriate Spotify support tone.
- Does not invent account information or actions.
- Does not make unsupported claims.
- Asks for additional information only when genuinely necessary.

4 = Good
- Correct and useful.
- Minor omissions or wording issues.
- Still appropriate for customer support.

3 = Acceptable
- Generally relevant.
- Somewhat generic or incomplete.
- Would probably require another support interaction.

2 = Poor
- Weakly addresses the problem.
- Missing important troubleshooting/support guidance.
- Too generic, confusing, or partially inappropriate.

1 = Unacceptable
- Irrelevant or incorrect.
- Unsafe.
- Invents actions/account information.
- Fails to address the customer's problem.

Return ONLY valid JSON:

{
  "score": 1,
  "reason": "short explanation"
}
"""


# ============================================================
# JUDGE REPLY
# ============================================================

def judge_reply(customer, reply):

    prompt = f"""
You are evaluating a Spotify customer-support agent.

CUSTOMER:
{customer}

AGENT REPLY:
{reply}

{RUBRIC}
"""

    text = call_gemini(
        prompt
    )

    # Remove accidental markdown fences.
    text = re.sub(
        r"```json",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"```",
        "",
        text,
    )

    text = text.strip()

    # Extract JSON if Gemini added surrounding text.
    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL,
    )

    if not match:
        raise ValueError(
            "Could not parse judge response:\n"
            f"{text}"
        )

    result = json.loads(
        match.group(0)
    )

    score = int(
        result["score"]
    )

    reason = str(
        result["reason"]
    )

    if score < 1 or score > 5:
        raise ValueError(
            f"Invalid judge score: {score}"
        )

    return score, reason


# ============================================================
# GENERATE REPLY
# ============================================================

def generate_reply(customer):

    prompt = f"""
You are a professional Spotify customer-support agent.

Customer message:
{customer}

Write a concise customer-facing support response.

Rules:

1. Directly address the customer's issue.
2. Be polite and professional.
3. Give useful troubleshooting when appropriate.
4. Ask for device/app information when needed for technical issues.
5. For account, billing, payment, refund, login, or security issues,
   ask the customer to continue through DM and provide appropriate
   account-identifying information.
6. Never claim to have accessed an account.
7. Never claim to have performed a backend action.
8. Never invent URLs.
9. Do not mention that you are an AI.
10. Write only the customer-facing reply.
"""

    return call_gemini(
        prompt
    )


# ============================================================
# SAVE PROGRESS
# ============================================================

def save_results(rows):

    CALIBRATION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df = pd.DataFrame(
        rows
    )

    result_df.to_csv(
        CALIBRATION_PATH,
        index=False,
    )


# ============================================================
# LOAD EXISTING PROGRESS
# ============================================================

def load_existing_results():

    if not CALIBRATION_PATH.exists():
        return []

    try:

        result_df = pd.read_csv(
            CALIBRATION_PATH
        )

        if result_df.empty:
            return []

        return result_df.to_dict(
            orient="records"
        )

    except Exception:

        return []


# ============================================================
# CREATE CALIBRATION SET
# ============================================================

def create_calibration_set(
    df,
    text_column,
):

    print()
    print("=" * 80)
    print("CREATING LLM-JUDGE CALIBRATION SET")
    print("=" * 80)

    existing_rows = load_existing_results()

    rows = existing_rows.copy()

    completed = len(rows)

    print()
    print(
        f"Already completed: "
        f"{completed}/{CALIBRATION_SIZE}"
    )

    if completed >= CALIBRATION_SIZE:

        print()
        print(
            "Calibration set is already complete."
        )

        return rows

    # --------------------------------------------------------
    # Deterministic sample.
    # --------------------------------------------------------

    sample = df.sample(
        n=min(
            CALIBRATION_SIZE,
            len(df),
        ),
        random_state=42,
    ).copy()

    # --------------------------------------------------------
    # Continue from previous progress.
    # --------------------------------------------------------

    for index in range(
        completed,
        len(sample),
    ):

        row = sample.iloc[index]

        customer = str(
            row[text_column]
        )

        print()
        print("=" * 80)
        print(
            f"[{index + 1}/{len(sample)}]"
        )

        print()
        print("Customer:")
        print(customer)

        # ----------------------------------------------------
        # Generate reply.
        # ----------------------------------------------------

        print()
        print("Generating reply...")

        try:

            reply = generate_reply(
                customer
            )

        except Exception as e:

            print()
            print(
                "Could not generate reply."
            )

            print(
                f"{type(e).__name__}: {e}"
            )

            print()
            print(
                "Stopping evaluation so the "
                "completed progress remains saved."
            )

            break

        print()
        print("Generated reply:")
        print(reply)

        # ----------------------------------------------------
        # Judge reply.
        # ----------------------------------------------------

        print()
        print("Judging reply...")

        try:

            score, reason = judge_reply(
                customer,
                reply,
            )

        except Exception as e:

            print()
            print(
                "Could not judge reply."
            )

            print(
                f"{type(e).__name__}: {e}"
            )

            print()
            print(
                "Stopping evaluation so the "
                "completed progress remains saved."
            )

            break

        print()
        print(
            f"LLM judge score: "
            f"{score}/5"
        )

        print(
            f"Judge reason: "
            f"{reason}"
        )

        # ----------------------------------------------------
        # Save immediately.
        # ----------------------------------------------------

        rows.append(
            {
                "customer_text": customer,
                "generated_reply": reply,
                "llm_judge_score": score,
                "llm_judge_reason": reason,
                "human_score": "",
                "human_notes": "",
            }
        )

        save_results(
            rows
        )

        print()
        print(
            "Progress saved."
        )

    print()
    print("=" * 80)
    print("CALIBRATION SET STATUS")
    print("=" * 80)

    print()
    print(
        f"Completed: "
        f"{len(rows)}/{CALIBRATION_SIZE}"
    )

    print()
    print(
        f"Saved to: "
        f"{CALIBRATION_PATH}"
    )

    if len(rows) >= CALIBRATION_SIZE:

        print()
        print(
            "Calibration generation complete."
        )

        print()
        print(
            "Next step:"
        )

        print(
            "Open the CSV and fill the "
            "human_score column with 1-5 ratings."
        )

    else:

        print()
        print(
            "Calibration is incomplete."
        )

        print(
            "Run the script again after the "
            "Gemini quota/service issue is resolved."
        )

    return rows


# ============================================================
# HUMAN / LLM AGREEMENT
# ============================================================

def calculate_agreement():

    if not CALIBRATION_PATH.exists():

        print()
        print(
            "Calibration file does not exist."
        )

        return

    df = pd.read_csv(
        CALIBRATION_PATH
    )

    required_columns = [
        "human_score",
        "llm_judge_score",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        print()
        print(
            "Calibration file is missing columns:"
        )

        print(
            missing
        )

        return

    human = pd.to_numeric(
        df["human_score"],
        errors="coerce",
    )

    judge = pd.to_numeric(
        df["llm_judge_score"],
        errors="coerce",
    )

    valid = (
        human.notna()
        & judge.notna()
    )

    if valid.sum() == 0:

        print()
        print(
            "No human scores entered yet."
        )

        print()
        print(
            "Please fill the human_score column "
            "in data/reply_judge_calibration.csv."
        )

        return

    human = human[valid]
    judge = judge[valid]

    # --------------------------------------------------------
    # Exact agreement.
    # --------------------------------------------------------

    exact_agreement = (
        (human == judge).mean()
        * 100
    )

    # --------------------------------------------------------
    # Agreement within one point.
    # --------------------------------------------------------

    within_one = (
        (
            abs(human - judge)
            <= 1
        ).mean()
        * 100
    )

    # --------------------------------------------------------
    # Mean absolute difference.
    # --------------------------------------------------------

    mean_absolute_difference = (
        abs(human - judge)
    ).mean()

    print()
    print("=" * 80)
    print("LLM JUDGE / HUMAN AGREEMENT")
    print("=" * 80)

    print()

    print(
        f"Human-labelled examples: "
        f"{len(human)}"
    )

    print(
        f"Exact agreement: "
        f"{exact_agreement:.2f}%"
    )

    print(
        f"Agreement within ±1: "
        f"{within_one:.2f}%"
    )

    print(
        f"Mean absolute score difference: "
        f"{mean_absolute_difference:.2f}"
    )

    print()
    print(
        "Exact agreement measures whether Gemini "
        "gave exactly the same score as the human."
    )

    print()
    print(
        "Agreement within ±1 measures broader "
        "practical agreement."
    )

    print()
    print(
        "Mean absolute difference shows how far "
        "the judge typically differs from the human."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("SPOTIFY REPLY QUALITY EVALUATION")
    print("=" * 80)

    print()
    print(
        f"Gemini model: {MODEL_NAME}"
    )

    df, text_column = load_golden_set()

    print()
    print(
        f"Loaded {len(df)} golden-set examples."
    )

    print(
        f"Customer column: {text_column}"
    )

    create_calibration_set(
        df,
        text_column,
    )

    calculate_agreement()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()