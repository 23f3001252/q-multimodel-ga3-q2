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
    question = request["question"]

    # Prepare Gemini payload
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
                        "text": question
                    }
                ]
            }
        ]
    }

    # Call Gemini API
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    headers = {"Content-Type": "application/json"}

    resp = requests.post(url, headers=headers, json=payload)
    resp_json = resp.json()

    # Extract text answer
    try:
        text_answer = resp_json["candidates"]["content"]["parts"]["text"]
    except Exception:
        text_answer = "Error: could not parse model response."

    answer = text_answer.strip()

    return {"answer": answer}
