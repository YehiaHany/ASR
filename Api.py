from fastapi import FastAPI, File, UploadFile, HTTPException
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
from fastapi.responses import FileResponse
import os
import librosa

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],  # Allows access from any origin. Restrict this in production
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# Load models and processors for both languages
MODEL_PATH_FUSHA = "./wav2vec2-arabic-model"
PROCESSOR_PATH_FUSHA = "./wav2vec2-arabic-processor"

MODEL_PATH_MSA = "./wav2vec2-arabic-masry-model"
PROCESSOR_PATH_MSA = "./wav2vec2-arabic-masry-processor"

# Load model and processor for Fusha (Classical Arabic)
try:
    model_fusha = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH_FUSHA)
    processor_fusha = Wav2Vec2Processor.from_pretrained(PROCESSOR_PATH_FUSHA)
    model_fusha.eval()
    print("Fusha model and processor loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load Fusha model and processor: {e}")

# Load model and processor for MSA (Modern Standard Arabic)
try:
    model_msa = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH_MSA)
    processor_msa = Wav2Vec2Processor.from_pretrained(PROCESSOR_PATH_MSA)
    model_msa.eval()
    print("MSA model and processor loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load MSA model and processor: {e}")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_fusha.to(device)
model_msa.to(device)

@app.post("/transcribe_fusha/")
async def transcribe_fusha(file: UploadFile = File(...)):
    """
    Endpoint to transcribe an uploaded audio file in Fusha (Classical Arabic).
    """
    if not file.filename.lower().endswith(".wav"):
        # Rename the file to .wav if it's not a .wav file
        file_name_without_extension = os.path.splitext(file.filename)[0]
        file_path = f"temp_{file_name_without_extension}.wav"
    else:
        file_path = f"temp_{file.filename}"

    try:
        # Load the audio file
        audio_bytes = await file.read()
        with open(file_path, "wb") as f:
            f.write(audio_bytes)

        # Load and preprocess audio
        speech_array, sampling_rate = librosa.load(file_path, sr=16000)
        input_values = processor_fusha(speech_array, sampling_rate=16000, return_tensors="pt").input_values
        input_values = input_values.to(device)

        # Perform inference
        with torch.no_grad():
            logits = model_fusha(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)

        # Decode the predicted transcript
        transcription = processor_fusha.batch_decode(predicted_ids)[0]

        # Return the transcription
        return {"transcription": transcription}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    finally:
        # Clean up the temporary file after processing
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/")
async def read_index():
    # Serve the index.html file from the static folder
    return FileResponse(os.path.join("static", "index.html"))

@app.post("/transcribe_msa/")
async def transcribe_msa(file: UploadFile = File(...)):
    """
    Endpoint to transcribe an uploaded audio file in MSA (Modern Standard Arabic).
    """
    if not file.filename.lower().endswith(".wav"):
        # Rename the file to .wav if it's not a .wav file
        file_name_without_extension = os.path.splitext(file.filename)[0]
        file_path = f"temp_{file_name_without_extension}.wav"
    else:
        file_path = f"temp_{file.filename}"

    try:
        # Load the audio file
        audio_bytes = await file.read()
        with open(file_path, "wb") as f:
            f.write(audio_bytes)

        # Load and preprocess audio
        speech_array, sampling_rate = librosa.load(file_path, sr=16000)
        input_values = processor_msa(speech_array, sampling_rate=16000, return_tensors="pt").input_values
        input_values = input_values.to(device)

        # Perform inference
        with torch.no_grad():
            logits = model_msa(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)

        # Decode the predicted transcript
        transcription = processor_msa.batch_decode(predicted_ids)[0]

        # Return the transcription
        return {"transcription": transcription}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    finally:
        # Clean up the temporary file after processing
        if os.path.exists(file_path):
            os.remove(file_path)
