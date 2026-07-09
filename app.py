import os
import base64
import imghdr
import re
import requests

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ==========================================================
# Load Environment Variables
# ==========================================================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash"

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env file.")

# ==========================================================
# FastAPI App
# ==========================================================
app = FastAPI(title="Multimodal Image QA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# Request Schema
# ==========================================================
class ImageRequest(BaseModel):
    image_base64: str
    question: str


# ==========================================================
# Root Endpoint
# ==========================================================
@app.get("/")
def home():
    return {
        "message": "Multimodal Image Question Answering API is running."
    }


# ==========================================================
# Clean Gemini Output
# ==========================================================
def clean_answer(text: str) -> str:
    """
    Cleans Gemini response.

    Numeric answers:
        "$4,089.35"
        -> "4089.35"

    Text answers:
        "The answer is Housing."
        -> "Housing"
    """

    text = text.strip()

    # Remove common prefixes (case insensitive)
    text = re.sub(
        r"^(the total|total|answer|value|max|maximum|the answer is)[:\s]*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove currency and units
    text = re.sub(
        r"(USD|INR|Rs\.?|₹|\$|€|£|dollars?|rupees?|%)",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.strip(" .,:;!?\"'")

    # Numeric answer
    if re.search(r"\d", text):

        cleaned = re.sub(r"[^0-9.,\-]", "", text)

        cleaned = cleaned.replace(",", "")

        cleaned = cleaned.strip(".")

        return cleaned

    return text


# ==========================================================
# Main Endpoint
# ==========================================================
@app.post("/answer-image")
def answer_image(request: ImageRequest):

    # -----------------------------------
    # Validate Base64
    # -----------------------------------
    try:
        image_bytes = base64.b64decode(request.image_base64)
    except Exception:
        return {"answer": ""}

    # -----------------------------------
    # Detect MIME Type
    # -----------------------------------
    image_type = imghdr.what(None, image_bytes)

    mime = "image/png"

    if image_type in ["jpeg", "jpg"]:
        mime = "image/jpeg"

    elif image_type == "webp":
        mime = "image/webp"

    elif image_type == "gif":
        mime = "image/gif"

    question = request.question.strip()

    prompt = f"""
You are an OCR and document understanding assistant.

Answer ONLY the user's question.

Question:
{question}

Rules:
- Return ONLY the answer.
- Do NOT explain.
- If the answer is numeric, return only the number.
- Remove currency symbols and units.
- Preserve capitalization for text answers.
- Do not add punctuation.
"""

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime,
                            "data": request.image_base64,
                        }
                    },
                    {
                        "text": prompt,
                    },
                ]
            }
        ]
    }

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )

    try:

        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )

        response.raise_for_status()

        result = response.json()

    except requests.RequestException:
        return {"answer": ""}

    # -----------------------------------
    # Parse Gemini Response
    # -----------------------------------
    try:
        text_answer = (
            result["candidates"][0]["content"]["parts"][0]["text"]
        )

    except Exception:
        return {"answer": ""}

    answer = clean_answer(text_answer)

    return {"answer": str(answer)}
