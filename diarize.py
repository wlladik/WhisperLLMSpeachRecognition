from pyannote.audio import Pipeline
from dotenv import load_dotenv
import os

load_dotenv()  # Loading .env.example file
hf_token = os.getenv("HUGGINGFACE_TOKEN")


def run_diarization(audio_path):
    print("Token:", hf_token)

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=hf_token
    )

    diarization = pipeline(audio_path)
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/diarization.rttm", "w") as f:
        diarization.write_rttm(f)


run_diarization("data/audio/audio.wav")
