# compare_and_merge_llm.py
import json

WHISPER_PATH = "outputs/whisper/segments_transcribed.json"
FAST_WHISPER_PATH = "outputs/fastwhisper/segments_transcribed.json"
MERGED_OUTPUT = "outputs/final_merged/merged_result.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_segments(whisper_data, fast_data):
    merged = []
    for i in range(min(len(whisper_data), len(fast_data))):
        merged.append({
            "start_whisper": whisper_data[i]["start"],
            "end_whisper": whisper_data[i]["end"],
            "speaker": whisper_data[i].get("speaker", "Unknown"),
            "text_whisper": whisper_data[i]["text"],
            "text_fast": fast_data[i]["text"],
        })
    return merged


def save_json(data, path):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    whisper_data = load_json(WHISPER_PATH)
    fast_data = load_json(FAST_WHISPER_PATH)
    merged = merge_segments(whisper_data, fast_data)
    save_json(merged, MERGED_OUTPUT)
    print(f"✅ Saved to {MERGED_OUTPUT}")


if __name__ == "__main__":
    main()
