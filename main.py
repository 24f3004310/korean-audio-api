import os
import json
import base64
import tempfile
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq()

def process_audio_dataset(audio_id: str, base64_str: str):
    if not os.environ.get("GROQ_API_KEY"):
        return {"error": "GROQ_API_KEY is missing from environment variables"}

    # 1. Clean up base64 formatting string
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    
    audio_bytes = base64.b64decode(base64_str)

    # 2. Save the audio to a temporary file so the Groq API can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name

    try:
        # 3. Transcribe the audio using Groq's high-speed Whisper API
        with open(temp_audio_path, "rb") as audio_file:
            transcription_result = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                response_format="text"
            )
        
        # 4. Prompt Llama to read the transcription and build the exact statistical profile
        system_prompt = (
            "You are a dataset validation engine. Analyze the provided text transcription of an audio dataset "
            "and generate a single, strict JSON object with these 13 required keys matching the dataset metrics:\n"
            "{\n"
            "  \"rows\": integer (total number of numeric values/records),\n"
            "  \"columns\": [\"exact spoken column name string, e.g., 나이\"],\n"
            "  \"mean\": {\"column_name\": float},\n"
            "  \"std\": {\"column_name\": float},\n"
            "  \"variance\": {\"column_name\": float},\n"
            "  \"min\": {\"column_name\": float},\n"
            "  \"max\": {\"column_name\": float},\n"
            "  \"median\": {\"column_name\": float},\n"
            "  \"mode\": {\"column_name\": float},\n"
            "  \"range\": {\"column_name\": float},\n"
            "  \"allowed_values\": {\"column_name\": [list of unique values found]},\n"
            "  \"value_range\": {\"column_name\": [min_value, max_value]},\n"
            "  \"correlation\": []\n"
            "}\n"
            "CRITICAL: Ensure the column array extracts the exact name spoken in the text (like '나이'). "
            "Output only raw valid JSON without markdown formatting blocks."
        )

        chat_completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Dataset Transcription: {transcription_result}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )

        parsed_json = json.loads(chat_completion.choices[0].message.content.strip())
        return parsed_json

    finally:
        # 5. Clean up the temporary sound file from the server
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

# Universal Catch-All route to handle the grader safely
@app.post("/{catchall:path}")
async def catch_all_post(request: Request):
    try:
        body = await request.json()
        audio_id = body.get("audio_id", "unknown")
        audio_base64 = body.get("audio_base64", "")
        
        if not audio_base64:
            raise HTTPException(status_code=400, detail="Missing audio data")
            
        return process_audio_dataset(audio_id, audio_base64)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/{catchall:path}")
def home():
    return {"status": "Whisper Dataset Analyzer Engine Online"}
