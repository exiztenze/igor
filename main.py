from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import json
from vosk import Model, KaldiRecognizer
from gtts import gTTS
from io import BytesIO

app = FastAPI()

# VOSK modelini yükle (Türkçe için küçük model)
vosk_model = Model("vosk-model-small-tr")

class TextRequest(BaseModel):
    text: str

@app.post("/ses_oku")
async def text_to_speech(request: TextRequest):
    """Metni sese çevir (TTS)"""
    try:
        tts = gTTS(text=request.text, lang='tr')
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        return {"audio": audio_bytes.getvalue()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ses_anla")
async def speech_to_text(audio_file: UploadFile = File(...)):
    """Sesi metne çevir (STT)"""
    try:
        # Ses dosyasını geçici olarak kaydet
        temp_file = "temp_audio.ogg"
        with open(temp_file, "wb") as f:
            f.write(await audio_file.read())
        
        # VOSK ile transkripsiyon
        rec = KaldiRecognizer(vosk_model, 16000)
        with open(temp_file, "rb") as f:
            while True:
                data = f.read(4000)
                if len(data) == 0:
                    break
                rec.AcceptWaveform(data)
        
        result = json.loads(rec.FinalResult())
        os.remove(temp_file)  # Geçici dosyayı sil
        
        return {"text": result.get("text", "")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def health_check():
    return {"status": "IGOR çalışıyor!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
