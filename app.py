import os
import base64
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import re

# Load .env file
load_dotenv()

# Read environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash"  # or another supported model

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found. Check your .env file.")

app = FastAPI()

# ===== CORS: allow any origin =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return "This is ga3-q2-tds-q-multimodel-image"


def clean_answer(text: str) -> str:
    """
    Clean the model's answer to:
    - Always return a string.
    - For numeric answers, return only the number, e.g. "4089.35".
    - Remove units, currency symbols, and extra text where possible.
    """
    text = text.strip()

    # Remove common prefixes like "The total is", "Answer:", etc.
    text = re.sub(r"^(the total|total|answer|value|max|maximum)[:\s]*", "", text.lower())

    # Remove units and currency symbols
    text = re.sub(r"\s*(usd|dollars|rs|₹|%|€|£|inr)\b", "", text, flags=re.IGNORECASE)

    # If the answer is clearly a category name (for pie chart), return it as-is
    # But still strip leading/trailing punctuation and "the " prefix
    text = text.strip(".,:;!?\"'")
    text = re.sub(r"^the\s+", "", text, flags=re.IGNORECASE)

    # If it looks numeric, clean it to just digits and '.'
    # Check if it contains digits
    if re.search(r"\d", text):
        # Keep only digits, '.', commas (for thousand separators), and minus
        cleaned = re.sub(r"[^0-9.,\-]", "", text)
        # Remove commas used as thousand separators
        cleaned = cleaned.replace(",", "")
        # Strip leading/trailing dots
        cleaned = cleaned.strip(".")
        if cleaned:
            return cleaned

    # Otherwise, return the cleaned text (for non-numeric answers like category names)
    return text.strip()


@app.post("/answer-image")
def answer_image(request: dict):
    """
    Request:
    {
      "image_base64": "iVBORw0KG...",
      "question": "What is the total?"
    }

    Response:
    {
      "answer": "4089.35"
    }
    """
    image_base64 = request["image_base64"]
    question = request["question"].strip()

    # Build a strong prompt that matches the sample image types:
    # - pie_chart: category name
    # - invoice: numeric total
    # - data_table: numeric value (e.g. max score)
    prompt = (
        f"Look at this image and answer the following question. "
        f"Question: {question}\n\n"
        f"Rules for your answer:\n"
        f"1. Always return your answer as a string.\n"
        f"2. If the answer is a number (e.g. total amount, max score), return ONLY the number "
        f"with no currency symbol, no units, and no extra text. Example: 4089.35, 95, 12.5.\n"
        f"3. If the answer is a text label (e.g. category name like 'Housing'), return ONLY that label "
        f"without 'the' or extra words. Example: Housing, Food, Transport.\n"
        f"4. Do not include any explanation, just the raw answer value."
    )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_base64
                        }
                    },
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    headers = {"Content-Type": "application/json"}

    resp = requests.post(url, headers=headers, json=payload)
    resp_json = resp.json()

    # Extract text answer from Gemini
    try:
        text_answer = resp_json["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        text_answer = "Error: could not parse model response."

    answer = clean_answer(text_answer)
    if not answer:
        answer = text_answer.strip()  # fallback to raw text if cleaning removes everything

    return {"answer": answer}
