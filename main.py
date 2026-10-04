import os
import sys
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI()


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Request/response models
# ---------------------------------------------------------

class CodeRequest(BaseModel):
    code: str


class CodeResponse(BaseModel):
    error: List[int]
    result: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


# ---------------------------------------------------------
# Python execution tool
# ---------------------------------------------------------

def execute_python_code(code: str) -> dict:
    """
    Execute Python code and return its output.

    Returns:
        {
            "success": bool,
            "output": str
        }
    """

    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        # Execute the supplied Python code
        exec(code)

        # Get exactly what was printed
        output = sys.stdout.getvalue()

        return {
            "success": True,
            "output": output
        }

    except Exception:
        # Get the complete Python traceback
        output = traceback.format_exc()

        return {
            "success": False,
            "output": output
        }

    finally:
        # Always restore normal stdout
        sys.stdout = old_stdout


# ---------------------------------------------------------
# AI error analysis
# ---------------------------------------------------------

def analyze_error_with_ai(code: str, error_traceback: str) -> List[int]:
    """
    Ask the AI to identify the Python line(s) responsible
    for the error.
    """

    token = os.environ.get("AIPIPE_TOKEN")

    if not token:
        raise RuntimeError("AIPIPE_TOKEN environment variable is not set")

    client = OpenAI(
        api_key=token,
        base_url="https://aipipe.org/openrouter/v1"
    )

    prompt = f"""
Analyze the Python code and traceback below.

Your job is to identify the exact source-code line number
or line numbers where the error occurred.

CODE:
{code}

TRACEBACK:
{error_traceback}

Return only the line number(s) responsible for the error.
"""

    response = client.chat.completions.create(
        model="google/gemini-2.0-flash-lite-001",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "error_analysis",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "error_lines": {
                            "type": "array",
                            "items": {
                                "type": "integer"
                            }
                        }
                    },
                    "required": ["error_lines"],
                    "additionalProperties": False
                }
            }
        }
    )

    result_text = response.choices[0].message.content

    result = ErrorAnalysis.model_validate_json(result_text)

    return result.error_lines


# ---------------------------------------------------------
# Main API endpoint
# ---------------------------------------------------------

@app.post("/code-interpreter", response_model=CodeResponse)
def code_interpreter(request: CodeRequest):

    # 1. Execute the Python code
    execution = execute_python_code(request.code)

    # 2. If successful, DO NOT call the AI
    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"]
        }

    # 3. The code failed, so ask AI to analyze the traceback
    error_lines = analyze_error_with_ai(
        request.code,
        execution["output"]
    )

    # 4. Return the AI analysis and the ORIGINAL traceback
    return {
        "error": error_lines,
        "result": execution["output"]
    }

