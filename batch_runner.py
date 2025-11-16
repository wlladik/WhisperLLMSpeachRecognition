import os
import json
import tempfile
from pyannote.audio import Pipeline
from pydub import AudioSegment
import whisper
from datetime import datetime
from tqdm import tqdm  # Progress

# ---------- Settings ----------
INPUT_FOLDER = "data/audio/"
OUTPUT_FOLDER = "outputs/whisper_batch/"
HUGGINGFACE_TOKEN = "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"
WHISPER_MODEL = "large"
LANGUAGE = "pl"
# ----------------------------------


def convert_to_wav(input_path, tmp_dir):
    audio = AudioSegment.from_file(input_path)
    wav_path = os.path.join(tmp_dir, "converted.wav")
    audio.export(wav_path, format="wav")
    return wav_path


def diarize_audio(audio_path, token):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=token)
    pipeline._embedding.batch_size = 1
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
    for seg in tqdm(segments, desc="Transcribing segments"):
        result = model.transcribe(seg["path"], language=language)
        results.append({
            "speaker": seg["speaker"],
            "start": seg["start"],
            "end": seg["end"],
            "text": result["text"].strip()
        })
    return results


def save_results(results, output_path, start_time, end_time):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "results": results,
            "start_time": start_time,
            "end_time": end_time
        }, f, ensure_ascii=False, indent=2)


def process_file(file_path, output_path, token, model_name, language):
    print(f"\nProcessing: {file_path}")
    if not token:
        raise ValueError("HUGGINGFACE_TOKEN is not set. Please set it in .env or system variables.")

    start_time = datetime.now()
    model = whisper.load_model(model_name)

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = convert_to_wav(file_path, tmp_dir)
        diarization = diarize_audio(wav_path, token)
        segments = extract_segments(diarization, wav_path, tmp_dir)
        results = transcribe_segments(segments, model, language)
        end_time = datetime.now()

        save_results(results, output_path, start_time.strftime('%Y-%m-%d %H:%M:%S'),
                     end_time.strftime('%Y-%m-%d %H:%M:%S'))

    print(f"Done: {file_path} | {start_time.strftime('%H:%M:%S')} → {end_time.strftime('%H:%M:%S')}")


def main():
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith((".mp3", ".wav"))]
    if not files:
        print("No audio files found in input folder.")
        return

    for file in tqdm(files, desc="Processing files"):
        try:
            input_path = os.path.join(INPUT_FOLDER, file)
            output_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(file)[0]}.json")
            process_file(input_path, output_path, HUGGINGFACE_TOKEN, WHISPER_MODEL, LANGUAGE)
        except Exception as e:
            print(f"Error processing {file}: {e}")
    print("All done!")


if __name__ == "__main__":
    main()
