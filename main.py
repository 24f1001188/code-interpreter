from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
from io import StringIO
import traceback
import httpx
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Enable CORS (required for testing from browser / other origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request / Response models ---

class CodeRequest(BaseModel):
    code: str

class CodeResponse(BaseModel):
    error: List[int]
    result: str

# For structured AI output
class ErrorAnalysis(BaseModel):
    error_lines: List[int]


# --- Tool function: execute Python code ---

def execute_python_code(code: str) -> dict:
    """
    Execute Python code and return exact output.

    Returns:
        {
            "success": bool,
            "output": str  # Exact stdout or traceback
        }
    """
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        # Create a clean globals dict for exec
        exec_globals = {}
        exec(code, exec_globals)
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}

    except Exception:
        output = traceback.format_exc()
        return {"success": False, "output": output}

    finally:
        sys.stdout = old_stdout


# --- AI error analysis using Gemini via AI Pipe (OpenRouter) ---

AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
if not AIPIPE_TOKEN:
    raise RuntimeError("AIPIPE_TOKEN not found in environment / .env file")

OPENROUTER_BASE_URL = "https://aipipe.org/openrouter/v1"


async def analyze_error_with_ai(code: str, traceback_text: str) -> List[int]:
    """
    Use LLM with structured JSON output to identify error line numbers.
    """
    prompt = f"""You are a helpful assistant that analyzes Python error tracebacks.

Your task:
- Look at the Python code and the traceback.
- Identify the line number(s) in the CODE where the error occurred.
- Return a JSON object with a single key "error_lines" which is a list of integers.

Rules:
- Line numbers start from 1.
- Only include lines that directly caused the error (e.g., the line that raised the exception).
- If multiple lines are involved, list all relevant line numbers.
- Do NOT include any explanation text, only valid JSON.

CODE:
{code}

TRACEBACK:
{traceback_text}

Respond ONLY with a JSON object like:
{{"error_lines": [3]}}
"""

    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
    }

    # Use a widely-supported model via OpenRouter/AI Pipe
    payload = {
        "model": "openai/gpt-4.1-nano",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.0,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, headers=headers, json=payload)
        # If this fails, we want to see why in the terminal
        if resp.status_code != 200:
            print("AI Pipe error status:", resp.status_code)
            print("AI Pipe error body:", resp.text)
            resp.raise_for_status()
        data = resp.json()

    content = data["choices"][0]["message"]["content"]

    # Parse JSON response
    try:
        parsed = ErrorAnalysis.model_validate_json(content)
    except Exception:
        import json
        raw = json.loads(content)
        parsed = ErrorAnalysis(**raw)

    return parsed.error_lines

# --- FastAPI endpoint ---

@app.post("/code-interpreter", response_model=CodeResponse)
async def code_interpreter(request: CodeRequest) -> CodeResponse:
    code = request.code

    # 1) Run the code
    exec_result = execute_python_code(code)

    # 2) If success, return immediately
    if exec_result["success"]:
        return CodeResponse(error=[], result=exec_result["output"])

    # 3) If error, call AI to analyze traceback
    traceback_text = exec_result["output"]
    error_lines = await analyze_error_with_ai(code, traceback_text)

    return CodeResponse(error=error_lines, result=traceback_text)
