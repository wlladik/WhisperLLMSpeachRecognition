from faster_whisper import WhisperModel
import json
import os


def transcribe_fast_whisper(audio_path, output_path, model_size="base"):
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, language="pl", beam_size=5)
    results = []
    for segment in segments:
        results.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })

    # Checking the directory (exists or not)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Saving JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Transcription saved to {output_path}")


if __name__ == "__main__":
    transcribe_fast_whisper(
        audio_path="data/audio/audio.wav",
        output_path="outputs/fastwhisper/segments_transcribed.json",
        model_size="base"  # "medium", "large-v2"
    )