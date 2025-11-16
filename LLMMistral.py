import os
import json
import requests
from tqdm import tqdm
import re

# Settings
WHISPER_BATCH_DIR = "outputs/whisper_batch"
OUTPUT_DIR = "outputs/whisper_batch_corrected"
LLM_API_URL = "http://localhost:1234/v1/chat/completions"
CHECKPOINT_FILE = "checkpoint.log"

# Prompt
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are an assistant that improves transcribed text from audio recordings in Polish.\n"
        "Each input contains:\n"
        "- speaker label (e.g., SPEAKER_00)\n"
        "- start and end time\n"
        "- text in Polish (possibly with errors)\n\n"
        "Fix grammar, clarity, and naturalness **in Polish only**.\n"
        "**Do not translate to English**. Do not change speaker or timestamps.\n"
        "**Return ONLY the corrected JSON**, without explanations or introductions.\n"
    )
}

def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def save_checkpoint(filename):
    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
        f.write(filename + "\n")

def correct_text_with_llm(segment):
    input_json = json.dumps(segment, ensure_ascii=False, indent=2)

    payload = {
        "model": "mistralai/mistral-7b-instruct-v0.3:2",
        "messages": [
            SYSTEM_PROMPT,
            {
                "role": "user",
                "content": f"Please correct the following transcript segment:\n\n{input_json}"
            }
        ],
        "temperature": 0.3,
        "stream": False
    }

    raw = None

    try:
        response = requests.post(LLM_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        raw = response.json()

        content = raw.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        if not content:
            raise ValueError("Empty response from LLM")

        cleaned = re.sub(r"^Assistant:\s*", "", content)

        corrected = json.loads(cleaned)
        return corrected

    except Exception as e:
        print(f"Error: {e}")
        if raw:
            print("Raw response:\n", json.dumps(raw, ensure_ascii=False, indent=2))
        else:
            print("No raw response (API might have failed)")

        with open("error_segments.log", "a", encoding="utf-8") as log:
            log.write(json.dumps(segment, ensure_ascii=False) + "\n\n")
        return segment

def process_file(filepath, output_path):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    corrected_segments = []
    print(f"Processing: {os.path.basename(filepath)}")
    for item in tqdm(data["results"], desc="Correcting segments"):
        corrected_segment = correct_text_with_llm(item)
        corrected_segments.append(corrected_segment)

    corrected_data = {
        "speaker": data.get("speaker"),
        "results": corrected_segments,
        "start_time": data.get("start_time"),
        "end_time": data.get("end_time")
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corrected_data, f, ensure_ascii=False, indent=2)
    print(f"Saved: {output_path}")


def main():
    completed_files = load_checkpoint()
    files = [f for f in os.listdir(WHISPER_BATCH_DIR) if f.endswith(".json")]

    if not files:
        print("No JSON files found in input directory.")
        return

    for filename in files:
        if filename in completed_files:
            print(f"Skipped (already done): {filename}")
            continue

        in_path = os.path.join(WHISPER_BATCH_DIR, filename)
        out_path = os.path.join(OUTPUT_DIR, filename)

        try:
            process_file(in_path, out_path)
            save_checkpoint(filename)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    main()
