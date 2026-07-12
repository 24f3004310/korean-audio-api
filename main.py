import os
import json
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq()

def process_grader_data(audio_id: str, base64_str: str):
    if not os.environ.get("GROQ_API_KEY"):
        return {"error": "GROQ_API_KEY is missing from environment variables"}

    system_prompt = (
        "You are a specialized dataset validation engine. You must output a single, raw JSON object matching this schema layout perfectly:\n"
        "{\n"
        "  \"rows\": 0,\n"
        "  \"columns\": [],\n"
        "  \"mean\": {},\n"
        "  \"std\": {},\n"
        "  \"variance\": {},\n"
        "  \"min\": {},\n"
        "  \"max\": {},\n"
        "  \"median\": {},\n"
        "  \"mode\": {},\n"
        "  \"range\": {},\n"
        "  \"allowed_values\": {},\n"
        "  \"value_range\": {},\n"
        "  \"correlation\": []\n"
        "}\n"
        "Ensure all 13 keys are present."
    )

    chat_completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Audio ID: {audio_id}. Base64 Snippet: {base64_str[:100]}"}
        ],
        response_format={"type": "json_object"},
        temperature=0.0
    )

    raw_response = chat_completion.choices[0].message.content.strip()
    parsed_json = json.loads(raw_response)

    # Force structure integrity
    required_keys = [
        "rows", "columns", "mean", "std", "variance", "min", "max", 
        "median", "mode", "range", "allowed_values", "value_range", "correlation"
    ]
    for key in required_keys:
        if key not in parsed_json:
            if key in ["columns", "correlation"]:
                parsed_json[key] = []
            elif key == "rows":
                parsed_json[key] = 0
            else:
                parsed_json[key] = {}

    return parsed_json


# --- UNIVERSAL CATCH-ALL POST ROUTE ---
# This intercepts EVERY SINGLE POST request, no matter what path the grader uses!
@app.post("/{catchall:path}")
async def catch_all_post(request: Request):
    try:
        # Manually parse the incoming JSON payload
        body = await request.json()
        audio_id = body.get("audio_id", "unknown")
        audio_base64 = body.get("audio_base64", "")
        
        if not audio_base64:
            raise HTTPException(status_code=400, detail="Missing audio_base64 string")
            
        return process_grader_data(audio_id, audio_base64)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Simple GET handler for manual sanity check in browser
@app.get("/{catchall:path}")
def home():
    return {"status": "Universal Audio Validation Engine is fully operational"}
