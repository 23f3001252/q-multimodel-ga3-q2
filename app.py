import os
import base64
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


def clean_answer(text: str) -> str:
    """
    Clean Gemini response while preserving text answers.

    Numeric examples:
        "$4,089.35" -> "4089.35"
        "95%" -> "95"

    Text examples:
        "Housing" -> "Housing"
        "Office Supplies" -> "Office Supplies"
    """
    text = text.strip().strip('"').strip("'")

    # Remove common prefixes
    text = re.sub(
        r"^(Answer:|The answer is|Total:|Grand Total:)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # If the whole response is basically a number (with optional currency/unit)
    match = re.fullmatch(
        r"\s*[$₹€£]?\s*(-?\d[\d,]*(?:\.\d+)?)\s*[%A-Za-z]*\s*",
        text,
    )

    if match:
        return match.group(1).replace(",", "")

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


    question = request.question.strip()

    prompt = f"""
    You are an expert OCR and document understanding assistant.
    
    Read the image carefully and answer ONLY the user's question.
    
    Question:
    {question}
    
    Rules:
    1. Return ONLY the answer.
    2. Do NOT explain your reasoning.
    3. Do NOT write complete sentences.
    4. If the answer is numeric:
       - Return only the number.
       - Remove currency symbols.
       - Remove commas used as thousand separators.
       - Remove units.
       Example:
       4089.35
    5. If the answer is text:
       - Return only the exact text from the image.
       - Preserve capitalization and spelling.
       - Do not add punctuation.
    """

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": image/png,
                            "data": request.image_base64,
                        }
                    },
                    {
                        "text": prompt,
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "topP": 0.1,
            "topK": 1,
            "maxOutputTokens": 32,
        },
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
            result["candidates"][0]["content"]["parts"][0]["text"].strip()
        )
    except (KeyError, IndexError, TypeError):
        return {"answer": ""}

    answer = clean_answer(text_answer)

    return {"answer": str(answer)}
