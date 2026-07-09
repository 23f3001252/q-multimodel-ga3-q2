import os
import base64
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Read environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash"  # or another supported model

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found. Check your .env file.")

app = FastAPI()

@app.get("/")
def home():
    return "This is ga3-q2-tds-q-multimodel-image"

import re

def clean_answer(text: str) -> str:
    # Remove common prefixes
    text = text.strip()
    text = re.sub(r"^(the total|total|answer|value)[:\s]*", "", text.lower())
    text = re.sub(r"[^0-9.]", "", text)  # keep only digits and '.'
    text = text.strip(".")
    return text

@app.post("/answer-image")
def answer_image(request: dict):
    image_base64 = request["image_base64"]
    question = request["question"]

    strong_question = (
        f"Extract the requested value from this image. Return only the number, "
        f"with no currency symbol, no units, and no extra text. For example: 4089.35. "
        f"Question: {question}"
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
                        "text": strong_question
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

    try:
        text_answer = resp_json["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        text_answer = "Error: could not parse model response."

    answer = clean_answer(text_answer)
    if not answer:
        answer = text_answer.strip()  # fallback to raw text if cleaning removes everything

    return {"answer": answer}
