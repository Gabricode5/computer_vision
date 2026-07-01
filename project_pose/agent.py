from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import requests

app = FastAPI()

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

class AnalysisRequest(BaseModel):
    user_question: str
    pose_data: str  # Ta liste de points convertie en string CSV

@app.post("/analyze")
async def analyze_pose(request: AnalysisRequest):
    # On construit un prompt très direct pour forcer BOB à analyser les chiffres
    prompt_complet = (
        f"Système: Tu es BOB, un physicien expert en biomécanique. "
        f"Voici les coordonnées (id, x, y, z) des 33 points clés du corps détectés par MediaPipe :\n"
        f"{request.pose_data}\n\n"
        f"L'utilisateur demande : {request.user_question}\n"
        f"Réponds avec précision scientifique sur la posture."
    )

    payload = {
        "model": "llama3.2:3b",
        "prompt": prompt_complet,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))