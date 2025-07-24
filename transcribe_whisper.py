import os
import shutil
import tempfile
import json
from pyannote.audio import Pipeline
from pydub import AudioSegment
import whisper

# ---------- Settings ----------
AUDIO_PATH = "data/audio/audio.wav"
OUTPUT_PATH = "outputs/whisper/segments_transcribed.json"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
WHISPER_MODEL = "large"  # "medium", "large"
LANGUAGE = "pl"
# -----------------------------------


def diarize_audio(audio_path):
    print("🔍 Diarization...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HUGGINGFACE_TOKEN)
    diarization = pipeline(audio_path)
    return diarization


def extract_segments(diarization, audio_path, tmp_dir):
    print("✂️ Cutting on segments...")
    audio = AudioSegment.from_wav(audio_path)
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_ms = int(turn.start * 1000)
        end_ms = int(turn.end * 1000)
        segment_audio = audio[start_ms:end_ms]
        segment_path = os.path.join(tmp_dir, f"{speaker}_{start_ms}_{end_ms}.wav")
        segment_audio.export(segment_path, format="wav")
        segments.append({
            "path": segment_path,
            "speaker": speaker,
            "start": turn.start,
            "end": turn.end
        })
    return segments


def transcribe_segments(segments, model_name, language):
    print("🧠 Loading Whisper...")
    model = whisper.load_model(model_name)
    results = []
    for seg in segments:
        result = model.transcribe(seg["path"], language=language)
        results.append({
            "speaker": seg["speaker"],
            "start": seg["start"],
            "end": seg["end"],
            "text": result["text"].strip()
        })
    return results


def save_results_json(results, output_path):
    print(f"💾 Saving results in {output_path} (JSON)...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    with tempfile.TemporaryDirectory() as tmp_dir:
        diarization = diarize_audio(AUDIO_PATH)
        segments = extract_segments(diarization, AUDIO_PATH, tmp_dir)
        results = transcribe_segments(segments, WHISPER_MODEL, LANGUAGE)
        save_results_json(results, OUTPUT_PATH)
    print("✅ Done!")


if __name__ == "__main__":
    main()
