from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import os
import json
from vosk import Model, KaldiRecognizer
from gtts import gTTS
from io import BytesIO
import logging

# Log ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="IGOR Voice Assistant API")

# Model yükleniyor (YENİ YAPI)
MODEL_PATH = "vosk-model-tr"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model klasörü bulunamadı: {MODEL_PATH}")

try:
    model = Model(MODEL_PATH)
    logger.info("VOSK model başarıyla yüklendi!")
except Exception as e:
    logger.error(f"Model yükleme hatası: {str(e)}")
    raise

class TextRequest(BaseModel):
    text: str

@app.post("/ses_oku", summary="Metni sese çevir")
async def text_to_speech(request: TextRequest):
    """
    Örnek istek:
    ```json
    {"text": "Merhaba dünya"}
    ```
    """
    try:
        tts = gTTS(text=request.text, lang='tr', slow=False)
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        
        return FileResponse(
            audio_bytes,
            media_type="audio/mpeg",
            filename="response.mp3"
        )
    except Exception as e:
        logger.error(f"TTS hatası: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ses_anla", summary="Sesi metne çevir")
async def speech_to_text(audio_file: UploadFile = File(...)):
    """
    Desteklenen formatlar: WAV, OGG (16kHz, mono)
    """
    try:
        # Geçici dosya oluştur
        temp_file = "temp_audio.ogg"
        with open(temp_file, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        
        # VOSK ile transkripsiyon
        rec = KaldiRecognizer(model, 16000)
        rec.SetWords(True)
        
        with open(temp_file, "rb") as f:
            while True:
                data = f.read(4000)
                if len(data) == 0:
                    break
                rec.AcceptWaveform(data)
        
        result = json.loads(rec.FinalResult())
        os.remove(temp_file)
        
        return {"text": result.get("text", ""), "confidence": result.get("confidence", 0)}
    
    except Exception as e:
        logger.error(f"STT hatası: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", include_in_schema=False)
async def health_check():
    return {"status": "IGOR aktif", "model": "VOSK-tr"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
