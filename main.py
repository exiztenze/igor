from fastapi import FastAPI, UploadFile
from transformers import pipeline
# main.py'de değişiklik:
# Whisper yerine VOSK kullanın (10x daha hafif)
import vosk
model = vosk.Model("vosk-model-small-tr")
from gtts import gTTS
import os

app = FastAPI()

# 1. Sohbet modeli (DialoGPT)
igor_sohbet = pipeline("text-generation", model="microsoft/DialoGPT-small")

# 2. Ses tanıma modeli (Whisper)
igor_kulak = whisper.load_model("tiny")

@app.post("/sor")
async def sor(ses_dosyasi: UploadFile = None, metin: str = None):
    # Eğer ses geldiyse, metne çevir
    if ses_dosyasi:
        ses_icerik = await ses_dosyasi.read()
        with open("gelen_ses.ogg", "wb") as f:
            f.write(ses_icerik)
        metin = igor_kulak.transcribe("gelen_ses.ogg")["text"]
    
    # IGOR yanıt versin
    yanit = igor_sohbet(metin, max_length=50)[0]["generated_text"]
    
    # Yanıtı sese çevir (TTS)
    tts = gTTS(yanit, lang="tr")
    tts.save("yanit.mp3")
    
    return {"metin": yanit, "ses": "yanit.mp3"}
