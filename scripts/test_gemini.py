import os

from dotenv import load_dotenv
from google import genai


load_dotenv()


api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found."
    )


print("API key loaded.")
print("API key length:", len(api_key))

client = genai.Client(
    api_key=api_key
)


print()
print("Testing Gemini 3.6 Flash...")


try:

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=(
            "Reply with exactly: "
            "Gemini API connection successful."
        ),
    )

    print()
    print("SUCCESS")

    print()
    print("Gemini response:")

    print(response.text)


except Exception as exc:

    print()
    print("ERROR")

    print(
        type(exc).__name__
    )

    print(exc)