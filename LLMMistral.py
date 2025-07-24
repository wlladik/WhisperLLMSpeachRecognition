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
        "Return corrected JSON with the same structure."
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

    try:
        response = requests.post(LLM_API_URL, json=payload)
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]

        # 🔧 Delete the prefixes as 'Assistant:\n'
        cleaned = re.sub(r"^Assistant:\s*", "", raw.strip())

        # Trying to parse JSON
        corrected = json.loads(cleaned)
        return corrected

    except Exception as e:
        print(f"⚠️ Error: {e}")
        print("⚠️ Raw response:\n", raw)
        with open("error_segments.log", "a", encoding="utf-8") as log:
            log.write(json.dumps(segment, ensure_ascii=False) + "\n\n")
        return segment


def process_file(filepath, output_path):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    corrected_data = []
    print(f"🧠 Processing: {os.path.basename(filepath)}")
    for item in tqdm(data):
        corrected_segment = correct_text_with_llm(item)
        corrected_data.append(corrected_segment)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corrected_data, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved: {output_path}")


def main():
    completed_files = load_checkpoint()
    files = [f for f in os.listdir(WHISPER_BATCH_DIR) if f.endswith(".json")]

    if not files:
        print("❌ We don't have any JSON in this directory.")
        return

    for filename in files:
        if filename in completed_files:
            print(f"⏩ Skipped (already done): {filename}")
            continue

        in_path = os.path.join(WHISPER_BATCH_DIR, filename)
        out_path = os.path.join(OUTPUT_DIR, filename)

        try:
            process_file(in_path, out_path)
            save_checkpoint(filename)
        except Exception as e:
            print(f"❌ Error in progressing the {filename}: {e}")


if __name__ == "__main__":
    main()
