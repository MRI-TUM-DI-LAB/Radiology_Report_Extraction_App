import json
import pandas as pd
import langid
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def detect_language(text):
    lang, _ = langid.classify(text)
    return lang

def process_and_evaluate(json_path, output_csv=None):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = []

    for i, entry in enumerate(data):
        report = entry.get("report", "").strip()
        translation = entry.get("translation", "").strip()

        if report:
            pred_lang = detect_language(report)
            records.append({
                "source": "report",
                "text": report,
                "ground_truth_lang": "de",
                "predicted_lang": pred_lang
            })

        if translation:
            pred_lang = detect_language(translation)
            records.append({
                "source": "translation",
                "text": translation,
                "ground_truth_lang": "en",
                "predicted_lang": pred_lang
            })

    df = pd.DataFrame(records)

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Results saved to: {output_csv}")

    y_true = df["ground_truth_lang"]
    y_pred = df["predicted_lang"]
    accuracy = accuracy_score(y_true, y_pred)

    print(f"\n LangID Accuracy: {accuracy * 100:.2f}% ({len(y_true)} samples)\n")

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=["de", "en"]))

    print("Classification Report:")
    print(classification_report(y_true, y_pred, labels=["de", "en"]))

    return df

if __name__ == "__main__":
    input_json = "parrot.json"    
    output_csv = "langid_results.csv"
    df = process_and_evaluate(input_json, output_csv)

    mistakes = df[df["ground_truth_lang"] != df["predicted_lang"]]

    print(f"Found {len(mistakes)} misclassified sample(s):\n")
    print(mistakes[["source", "ground_truth_lang", "predicted_lang", "text"]].to_string(index=False))
