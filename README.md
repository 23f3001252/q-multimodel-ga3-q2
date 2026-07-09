# Multimodal Image QA API

A FastAPI service that accepts a base64-encoded image and a question, and returns an answer extracted from the image using a multimodal LLM (e.g., Google Gemini).

## Features

- Endpoint: `POST /answer-image`
- Request format:
  ```json
  {
    "image_base64": "iVBORw0KG...",
    "question": "What is the total?"
  }
  ```
- Response format:
  ```json
  {
    "answer": "4089.35"
  }
  ```
- CORS enabled for cross-origin requests.
- Environment variables managed via `.env` (using `python-dotenv`).

## Prerequisites

- Python 3.8+
- Gemini API key (from Google AI Studio)

## Setup (Local)

1. Clone or copy this project into a folder.
2. Create a `.env` file in the project root with:
   ```env
   GEMINI_API_KEY=your_actual_gemini_api_key
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the server:
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

The API will be available at `http://localhost:8000`.

## Testing

Example request:

```bash
curl -X POST http://localhost:8000/answer-image \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "iVBORw0KGgoPNG...",
    "question": "What is the total?"
  }'
```

## Deployment

This project can be deployed to:

- Render (recommended)
- Hugging Face Spaces
- Cloudflare Tunnel (for local hosting)

When deploying, ensure:
- `.env` is configured on the deployment platform (or use platform secrets).
- The start command is:
  ```bash
  uvicorn app:app --host 0.0.0.0 --port $PORT
  ```

## Files

- `app.py` – FastAPI application with `/answer-image` endpoint.
- `requirements.txt` – Python dependencies.
- `.env` – Environment variables (do not commit to Git).
- `README.md` – This file.
- 
