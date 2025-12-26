# app.py
# Flask backend for NLP Question-and-Answering System
# Python 3.12 - Flask Web GUI

import os
import sqlite3
import json
import re
import string
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, g
import requests
import time


# Load environment variables
load_dotenv()

APP_DIR = Path(__file__).parent
DB_PATH = APP_DIR / "queries.db"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["DATABASE"] = str(DB_PATH)
print("GEMINI_API_KEY =", GEMINI_API_KEY)


# -----------------------------
# DATABASE HELPERS
# -----------------------------

def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(app.config["DATABASE"])
        db.row_factory = sqlite3.Row
    return db


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()


def init_db():
    db = get_db()
    cursor = db.cursor()
    
    # Create table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            processed_question TEXT,
            answer TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Check if processed_question column exists, if not add it
    cursor.execute("PRAGMA table_info(queries)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if "processed_question" not in columns:
        try:
            cursor.execute("ALTER TABLE queries ADD COLUMN processed_question TEXT")
        except Exception as e:
            print(f"Migration note: {e}")
    
    db.commit()


# Initialize DB on startup
with app.app_context():
    init_db()


# -------- TEXT PREPROCESSING --------

class TextPreprocessor:
    """Text preprocessing utilities"""
    
    @staticmethod
    def lowercase(text: str) -> str:
        """Convert to lowercase"""
        return text.lower()
    
    @staticmethod
    def remove_punctuation(text: str) -> str:
        """Remove punctuation"""
        return text.translate(str.maketrans('', '', string.punctuation))
    
    @staticmethod
    def tokenize(text: str) -> list:
        """Simple tokenization"""
        return text.split()
    
    @staticmethod
    def preprocess(text: str) -> str:
        """Full preprocessing pipeline: lowercase -> remove punctuation -> tokenize -> rejoin"""
        text = TextPreprocessor.lowercase(text)
        text = TextPreprocessor.remove_punctuation(text)
        tokens = TextPreprocessor.tokenize(text)
        return " ".join(tokens)


# -----------------------------
# GEMINI API HELPER
# -----------------------------

def clean_markdown(text: str) -> str:
    """Remove common markdown formatting like **bold**, *italic*, etc."""
    import re
    # Remove bold: **text** -> text
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    # Remove italic: *text* -> text
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    # Remove bold (alt): __text__ -> text
    text = re.sub(r'__(.+?)__', r'\1', text)
    # Remove italic (alt): _text_ -> text
    text = re.sub(r'_(.+?)_', r'\1', text)
    # Remove code backticks: `text` -> text
    text = re.sub(r'`([^`]+)`', r'\1', text)
    return text


def call_gemini_api(prompt_text: str) -> str:
    """
    Call Gemini API with retry logic and automatic fallback.
    - Retries on 503 (overloaded) with exponential backoff
    - Falls back to gemini-2.0-flash if primary model fails
    - Tries multiple endpoint versions
    """
    if not GEMINI_API_KEY:
        return "ERROR: GEMINI_API_KEY not set."

    primary_model = GEMINI_MODEL  # e.g. "gemini-2.5-flash"
    fallback_model = "gemini-2.5-pro"
    
    # List of models to try in order
    models_to_try = [primary_model, fallback_model]

    def _make_request(model: str, url: str, payload: dict, headers: dict) -> tuple:
        """
        Make a single request with retry logic on 503.
        Returns (status_code, data_dict or error_message).
        """
        max_retries = 3
        retry_delay = 1  # Start with 1 second
        
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    url,
                    params={"key": GEMINI_API_KEY} if GEMINI_API_KEY else None,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                # If 503 (service unavailable), retry with exponential backoff
                if resp.status_code == 503 and attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # 1s, 2s, 4s
                    print(f"[Retry {attempt + 1}/{max_retries}] Model overloaded (503). Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue

                # Handle rate limit responses with server-provided retry-after when present
                if resp.status_code == 429 and attempt < max_retries - 1:
                    retry_after = resp.headers.get("Retry-After", "2")
                    print(f"[Retry {attempt + 1}/{max_retries}] Rate limited (429). Waiting {retry_after}s...")
                    try:
                        time.sleep(float(retry_after))
                    except ValueError:
                        time.sleep(2)
                    continue
                
                return resp.status_code, resp
                
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"[Retry {attempt + 1}/{max_retries}] Network error. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                return None, str(e)
        
        return resp.status_code, resp

    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 512},
    }

    headers = {"Content-Type": "application/json"}

    def _extract_text(data: dict) -> str | None:
        if not isinstance(data, dict):
            return None
        # Modern shape: candidates -> content -> parts[].text
        if "candidates" in data and data["candidates"]:
            cand = data["candidates"][0]
            if isinstance(cand, dict):
                content = cand.get("content") or cand.get("message")
                if isinstance(content, dict):
                    parts = content.get("parts") or []
                    if parts and isinstance(parts[0], dict) and "text" in parts[0]:
                        return parts[0]["text"]
                if "text" in cand and isinstance(cand["text"], str):
                    return cand["text"]
                if "content" in cand and isinstance(cand["content"], str):
                    return cand["content"]

        # Older shape: output -> list with text/content
        if "output" in data and isinstance(data["output"], list):
            collected = []
            for item in data["output"]:
                if isinstance(item, dict):
                    if "content" in item and isinstance(item["content"], str):
                        collected.append(item["content"])
                    if "text" in item and isinstance(item["text"], str):
                        collected.append(item["text"])
            if collected:
                return "\n".join(collected).strip()

        # Fallback keys
        for key in ("response", "responses"):
            if key in data:
                val = data[key]
                if isinstance(val, str):
                    return val
                if isinstance(val, dict) and "content" in val and isinstance(val["content"], str):
                    return val["content"]

        return None

    # Try each model with multiple endpoint versions
    last_error = None
    for model in models_to_try:
        endpoints = [
            f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent",
        ]
        
        for url in endpoints:
            status, result = _make_request(model, url, payload, headers)
            
            if status is None:
                last_error = f"Network error contacting {url}: {result}"
                continue

            resp = result

            if status == 404:
                last_error = f"404 from {url}: endpoint not found"
                break  # Skip to next model

            # 503 after all retries exhausted, try next model
            if status == 503:
                try:
                    err_body = resp.text
                except Exception:
                    err_body = "(unable to read response body)"
                last_error = f"503 (model overloaded) from {url}: {err_body}"
                # Continue to next model
                break  # Break to try next model

            # Other non-200 errors
            if status != 200:
                try:
                    body = resp.text
                except Exception:
                    body = "(unable to read response body)"
                return f"ERROR: Gemini API returned {status} from {url}: {body}"

            # 200 -> attempt to parse
            try:
                data = resp.json()
            except Exception as e:
                return f"ERROR: Could not decode JSON response from {url}: {e}"

            text = _extract_text(data)
            if text:
                return clean_markdown(text)

            # Return raw JSON if parsing didn't yield text
            return json.dumps(data)

    # If all models and endpoints failed
    if last_error:
        return f"ERROR: All Gemini models/endpoints failed. Last error: {last_error}"
    return "ERROR: All Gemini models/endpoints failed with unknown errors."



# -----------------------------
# ROUTES
# -----------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json() or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"ok": False, "error": "No question provided."}), 400

    # Preprocess the question
    processed_question = TextPreprocessor.preprocess(question)
    
    # Query Gemini with processed question
    raw_api_response = call_gemini_api(processed_question)
    
    # For now, the answer is the same as raw response
    # (In production, you might clean it up further here)
    answer = raw_api_response

    # Save to SQLite with both original and processed question
    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        "INSERT INTO queries (question, processed_question, answer) VALUES (?, ?, ?)",
        (question, processed_question, answer)
    )
    db.commit()

    return jsonify({
        "ok": True,
        "original_question": question,
        "processed_question": processed_question,
        "raw_api_response": raw_api_response,
        "answer": answer
    })


# -----------------------------
# DEBUG LOCAL SERVER
# -----------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
