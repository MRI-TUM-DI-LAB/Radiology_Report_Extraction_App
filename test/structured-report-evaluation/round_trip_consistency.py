import json
import argparse
from bert_score import score

# could be reused for multiligual evaluation
def json_to_text_recursive(data, parent_key=""):
    lines = []
    if isinstance(data, dict):
        for k, v in data.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            lines.extend(json_to_text_recursive(v, new_key))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            new_key = f"{parent_key}[{i}]"
            lines.extend(json_to_text_recursive(item, new_key))
    else:
        lines.append(f"{parent_key}: {data}")
    return lines

def structured_report_to_text(report):
    lines = json_to_text_recursive(report)
    return "\n".join(lines)

def main(args):
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)  

    references = []
    candidates = []

    freetext = data['freetext']
    structured = data['structured']

    reconstructed_text = structured_report_to_text(structured)

    references.append(freetext)
    candidates.append(reconstructed_text)

    P, R, F1 = score(candidates, references, lang='en', verbose=True)
    for i, (p, r, f) in enumerate(zip(P, R, F1)):
        print(f"Sample {i}: Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f}")
    print(f"Average Precision: {P.mean():.4f}")
    print(f"Average Recall: {R.mean():.4f}")
    print(f"Average F1: {F1.mean():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate structured report by BERTScore with JSON flattening")
    parser.add_argument("--input", type=str, required=True,
                        help="Input file path, standard JSON array format, possibly multi-line")
    args = parser.parse_args()

    main(args)
