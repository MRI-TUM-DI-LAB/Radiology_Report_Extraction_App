import os
import json

json_file_path = "parrot.json"  
german_dir = "free_text_parrot_german"
english_dir = "free_text_parrot_english"

os.makedirs(german_dir, exist_ok=True)
os.makedirs(english_dir, exist_ok=True)

with open(json_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for entry in data:
    no = entry.get("no", "unknown")  

    report_text = entry.get("report", "")
    report_filename = f"{no}.txt"
    report_path = os.path.join(german_dir, report_filename)
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(report_text)

    translation_text = entry.get("translation", "")
    translation_filename = f"{no}.txt"
    translation_path = os.path.join(english_dir, translation_filename)
    with open(translation_path, "w", encoding="utf-8") as translation_file:
        translation_file.write(translation_text)

print("Files have been created successfully.")
