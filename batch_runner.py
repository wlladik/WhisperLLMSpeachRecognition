# batch_runner.py
import os
import json
import tempfile
from pyannote.audio import Pipeline
from pydub import AudioSegment
import whisper

# ---------- Налаштування ----------
INPUT_FOLDER = "data/audio/"
OUTPUT_FOLDER = "outputs/whisper_batch/"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
WHISPER_MODEL = "large"
LANGUAGE = "pl"
# ----------------------------------


def diarize_audio(audio_path, token):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=token)
    return pipeline(audio_path)


def extract_segments(diarization, audio_path, tmp_dir):
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


def transcribe_segments(segments, model, language):
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


def save_results(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def process_file(file_path, output_path, token, model_name, language):
    print(f"🔊 Processing: {file_path}")
    model = whisper.load_model(model_name)
    with tempfile.TemporaryDirectory() as tmp_dir:
        diarization = diarize_audio(file_path, token)
        segments = extract_segments(diarization, file_path, tmp_dir)
        results = transcribe_segments(segments, model, language)
        save_results(results, output_path)


def main():
    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".wav")]
    for file in files:
        input_path = os.path.join(INPUT_FOLDER, file)
        output_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(file)[0]}.json")
        process_file(input_path, output_path, HUGGINGFACE_TOKEN, WHISPER_MODEL, LANGUAGE)
    print("✅ Done.")


if __name__ == "__main__":
    main()
