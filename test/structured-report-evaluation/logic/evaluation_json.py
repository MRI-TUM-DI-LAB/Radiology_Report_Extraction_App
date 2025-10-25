import json
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
import torch
from report_structuring import ReportStructuringProcessor
from template_selection_test import test_template_mapping
import os
from tqdm import tqdm

def load_json(file_path):
    """Load a JSON file and return its content."""
    with open(file_path, 'r') as file:
        return json.load(file)

def legal_json(file_path):
    """Check if a JSON file is well-formed."""
    try:
        with open(file_path, 'r') as file:
            json.load(file)
        return True
    except json.JSONDecodeError as e:
        print(f"JSON format is not legal: {e}")
        return False

def compare_json_fields(gt_json, pred_json, path=''):
    total = 0
    match = 0
    field_matches = []
    visited_keys = set()

    tp = 0  # correct prediction
    fp = 0  # prediction has extra field
    fn = 0  # ground-truth field missing or wrong

    for key in gt_json:
        new_path = f"{path}.{key}" if path else key
        visited_keys.add(key)

        if key not in pred_json:
            print(f"[MISSING] {new_path} not found in prediction.")
            field_matches.append((new_path, False))
            total += 1
            fn += 1
            continue

        if isinstance(gt_json[key], dict) and isinstance(pred_json[key], dict):
            sub_match, sub_total, sub_fields, sub_tp, sub_fp, sub_fn = compare_json_fields(gt_json[key], pred_json[key], new_path)
            match += sub_match
            total += sub_total
            tp += sub_tp
            fp += sub_fp
            fn += sub_fn
            field_matches.extend(sub_fields)
        else:
            total += 1
            if gt_json[key] == pred_json[key]:
                print(f"[MATCH] {new_path}")
                match += 1
                tp += 1
                field_matches.append((new_path, True))
            else:
                print(f"[MISMATCH] {new_path}")
                print(f"  GT  : {gt_json[key]}")
                print(f"  Pred: {pred_json[key]}")
                fn += 1
                field_matches.append((new_path, False))

    # Check for extra fields in prediction
    for key in pred_json:
        if key in visited_keys:
            continue
        new_path = f"{path}.{key}" if path else key
        if isinstance(pred_json[key], dict):
            _, _, sub_fields, _, sub_fp, _ = compare_json_fields({}, pred_json[key], new_path)
            fp += sub_fp
            field_matches.extend(sub_fields)
        else:
            print(f"[EXTRA] {new_path} not found in GT.")
            field_matches.append((new_path, False))
            fp += 1
            total += 1

    return match, total, field_matches, tp, fp, fn


def flatten_json(json_obj, parent_key=''):
    items = {}
    for k, v in json_obj.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_json(v, new_key))
        else:
            items[new_key] = str(v)
    return items

def semantic_evaluate(gt_report, gen_report):
    flat_gt = flatten_json(gt_report)
    flat_gen = flatten_json(gen_report)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    scores = {}
    total_score = 0
    count = 0

    for key, gt_text in flat_gt.items():
        gen_text = flat_gen.get(key)
        if gen_text is None:
            score = 0.0
        else:
            emb_gt = model.encode(gt_text, convert_to_tensor=True)
            emb_gen = model.encode(gen_text, convert_to_tensor=True)
            score = util.cos_sim(emb_gt, emb_gen).item()

        scores[key] = round(score, 4)
        total_score += score
        count += 1

    macro_similarity = round(total_score / count, 4) if count > 0 else 0

    return {
        "per_field_scores": scores,
        "macro_similarity": macro_similarity
    }



if __name__ == "__main__":

    input_folder = '../free_text'
    output_folder = '../output_sequential'


    os.makedirs(output_folder, exist_ok=True)
    
    processor = ReportStructuringProcessor()
    template_assignment = {}
    
    for filename in tqdm(os.listdir(input_folder)):
        # check if it is already processed
        if filename.endswith(".txt"):
            if os.path.exists(os.path.join(output_folder, os.path.splitext(filename)[0] + ".json")):
                print(f"Skipping {filename}, already processed.")
                continue
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + ".json"
            output_path = os.path.join(output_folder, output_filename)
            
            try:
                with open(input_path, "r", encoding="utf-8") as f:
                    freetext = f.read()
                    structured_json, template_name = processor.process_freetext(freetext)
                    # template_name = processor.process_freetext(freetext)
                    template_assignment[filename] = template_name

                # Save to individual JSON file
                with open(output_path, "w", encoding="utf-8") as out_f:
                    json.dump(structured_json, out_f, indent=2, ensure_ascii=False)
                print(f"✅ Processed {filename} -> {output_filename}")
                
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

    # print(template_assignment)
    # template_assignment = {'MR_HIP_2.txt': 'MR_HIP.json', 'MR_SHOULDER_7.txt': 'MR_SHOULDER.json', 'CT_SINUSES_10.txt': 'CT_SINUS.json', 'CT_NECK_3.txt': 'CT_NECK.json', 'CT_SPINE_2.txt': 'CT_SPINE.json', 'MR_KNEE_5.txt': 'MR_KNEE.json', 'MR_ABDOMEN_2.txt': 'MR_ABDOMEN.json', 'MR_ABDOMEN_3.txt': 'MR_ABDOMEN.json', 'MR_KNEE_4.txt': 'MR_KNEE.json', 'CT_SPINE_3.txt': 'CT_SPINE.json', 'CT_NECK_2.txt': 'CT_NECK.json', 'MR_SHOULDER_6.txt': 'MR_SHOULDER.json', 'CT_ANGIOGRAPHY_1.txt': 'CT_ANGIOGRAPHY_ABDOMEN_EXTREMITIES.json', 'CT_PANKREAS_1.txt': 'CT_PANKREAS_TUMOR.json', 'MR_HIP_3.txt': 'MR_HIP.json', 'MR_HIP_1.txt': 'MR_HIP.json', 'CT_ANGIOGRAPHY_3.txt': 'CT_ANGIOGRAPHY_ABDOMEN.json', 'MR_NECK_4.txt': 'MR_NECK.json', 'MR_SHOULDER_4.txt': 'MR_SHOULDER.json', 'CT_SPINE_1.txt': 'CT_SPINE.json', 'MR_ABDOMEN_1.txt': 'MR_ABDOMEN.json', 'MR_KNEE_6.txt': 'MR_KNEE.json', 'MR_KNEE_7.txt': 'MR_KNEE.json', 'CT_NECK_1.txt': 'MR_NECK.json', 'MR_SHOULDER_5.txt': 'MR_SHOULDER.json', 'MR_NECK_5.txt': 'MR_NECK.json', 'CT_ANGIOGRAPHY_2.txt': 'CT_ANGIOGRAPHY_ABDOMEN_EXTREMITIES.json', 'MR_NECK_1.txt': 'MR_NECK.json', 'MR_HIP_4.txt': 'MR_HIP.json', 'MR_SHOULDER_1.txt': 'MR_SHOULDER.json', 'CT_SPINE_4.txt': 'CT_SPINE.json', 'CT_NECK_5.txt': 'CT_HEAD_AND_NECK.json', 'MR_SPINE_9.txt': 'MR_LUMBAR_THORACIC_SPINE_DEGENERATIVE_CHANGES.json', 'CT_ABDOMEN_8.txt': 'CT_ABDOMEN_AND_PELVIS.json', 'MR_ABDOMEN_4.txt': 'MR_ABDOMEN.json', 'MR_KNEE_3.txt': 'MR_KNEE.json', 'MR_KNEE_2.txt': 'MR_KNEE.json', 'MR_ABDOMEN_5.txt': 'MR_ABDOMEN.json', 'CT_ABDOMEN_9.txt': 'CT_ABDOMEN_AND_PELVIS.json', 'MR_SPINE_8.txt': 'MR_COMPLETE_SPINE_DEGENERATIVE_CHANGES.json', 'CT_NECK_4.txt': 'MR_NECK.json', 'CT_SPINE_5.txt': 'CT_SPINE.json', 'MR_HIP_5.txt': 'MR_HIP.json', 'MR_NECK_2.txt': 'MR_NECK.json', 'CT_ANGIOGRAPHY_5.txt': 'CT_ANGIOGRAPHY_ABDOMEN_EXTREMITIES.json', 'MR_SHOULDER_2.txt': 'MR_SHOULDER.json', 'CT_SPINE_7.txt': 'CT_SPINE.json', 'CT_HEAD_10.txt': 'CT_HEAD_WITHOUT_CONTRAST.json', 'CT_HEAD_9.txt': 'CT_HEAD_WITHOUT_CONTRAST.json', 'MR_ABDOMEN_7.txt': 'MR_ABDOMEN.json', 'MR_ABDOMEN_6.txt': 'MR_ABDOMEN.json', 'MR_KNEE_1.txt': 'MR_KNEE.json', 'CT_HEAD_8.txt': 'CT_HEAD_WITH_CONTRAST.json', 'CT_HEAD_11.txt': 'CT_HEAD_WITHOUT_CONTRAST.json', 'CT_SPINE_6.txt': 'CT_SPINE.json', 'MR_SHOULDER_3.txt': 'MR_SHOULDER.json', 'CT_ANGIOGRAPHY_4.txt': 'CT_ANGIOGRAPHY_ABDOMEN_EXTREMITIES.json', 'MR_NECK_3.txt': 'MR_NECK.json', 'CT_CHEST_6.txt': 'CT_CHEST.json', 'MR_KNEE_10.txt': 'MR_KNEE.json', 'MR_ANGIOGRAPHY_1.txt': 'MR_ANGIOGRAPHY_THORAX.json', 'CT_SINUSES_5.txt': 'CT_HEAD_WITHOUT_CONTRAST.json', 'MR_PROSTATE_3.txt': 'MR_PROSTATE.json', 'MR_WHOLEBODY_5.txt': 'MR_WHOLE_BODY.json', 'MR_WHOLEBODY_4.txt': 'MR_WHOLE_BODY.json', 'MR_PROSTATE_2.txt': 'MR_ABDOMEN.json', 'CT_SINUSES_4.txt': 'CT_SINUS.json', 'CT_CHEST_7.txt': 'CT_CHEST.json', 'CT_CHEST_5.txt': 'CT_CHEST.json', 'CT_SINUSES_6.txt': 'CT_SINUS.json', 'MR_ANGIOGRAPHY_2.txt': 'MR_ANGIOGRAPHY_ABDOMEN.json', 'MR_WHOLEBODY_6.txt': 'MR_WHOLE_BODY.json', 'CTA_ABDOMENPELVIS_8.txt': 'CT_ABDOMEN_AND_PELVIS.json', 'MR_BRAIN_9.txt': 'MR_ADULT_BRAIN.json', 'MR_BRAIN_8.txt': 'MR_BRAIN_STROKE_OR_TIA.json', 'CTA_ABDOMENPELVIS_9.txt': 'CT_ABDOMEN_AND_PELVIS.json', 'MR_WHOLEBODY_7.txt': 'MR_WHOLE_BODY.json', 'MR_PROSTATE_1.txt': 'MR_ABDOMEN.json', 'MR_ANGIOGRAPHY_3.txt': 'MR_ANGIOGRAPHY_ABDOMEN.json', 'CT_SINUSES_7.txt': 'CT_HEAD_WITHOUT_CONTRAST.json', 'CTA_ABDOMENPELVIS_10.txt': 'CT_ABDOMEN_AND_PELVIS.json', 'CT_CHEST_4.txt': 'CT_CHEST.json', 'CT_SINUSES_3.txt': 'CT_SINUS.json', 'MR_PROSTATE_5.txt': 'MR_PROSTATE.json', 'MR_WHOLEBODY_3.txt': 'MR_WHOLE_BODY.json', 'MR_WHOLEBODY_2.txt': 'MR_WHOLE_BODY.json', 'MR_PROSTATE_4.txt': 'MR_PROSTATE.json', 'CT_SINUSES_2.txt': 'CT_SINUS.json', 'CT_CHEST_1.txt': 'CT_CHEST.json', 'MR_SHOULDER_11.txt': 'MR_SHOULDER.json', 'CT_CHEST_3.txt': 'CT_CHEST.json', 'MR_ANGIOGRAPHY_4.txt': 'MR_ANGIOGRAPHY_EXTREMITIES.json', 'MR_WHOLEBODY_1.txt': 'MR_WHOLE_BODY.json', 'CT_SINUSES_1.txt': 'CT_SINUS.json', 'MR_ANGIOGRAPHY_5.txt': 'MR_ANGIOGRAPHY_ABDOMEN.json', 'CT_CHEST_2.txt': 'CT_CHEST.json', 'MR_WHOLEBODY_10.txt': 'MR_HIP.json', 'MR_SHOULDER_10.txt': 'MR_SHOULDER.json', 'CT_ABDOMEN_10.txt': 'CT_ABDOMEN_AND_PELVIS.json', 'MR_BRAIN_3.txt': 'MR_ADULT_BRAIN.json', 'CTA_ABDOMENPELVIS_2.txt': 'CT_ABDOMEN_AND_PELVIS.json', 'MR_SPINE_10.txt': 'MR_LUMBAR_THORACIC_SPINE_DEGENERATIVE_CHANGES.json', 'CTA_ABDOMENPELVIS_3.txt': 'CT_ABDOMEN_AND_PELVIS.json', 'MR_BRAIN_2.txt': 'MR_BRAIN_STROKE_OR_TIA.json', 'CTA_ABDOMENPELVIS_1.txt': 'CT_ABDOMEN_AND_PELVIS.json', 'CT_CHEST_10.txt': 'CT_CHEST.json', 'CT_CHEST_11.txt': 'CT_CHEST.json', 'MR_BRAIN_1.txt': 'MR_ADULT_BRAIN.json', 'MR_BRAIN_11.txt': 'MR_ADULT_BRAIN.json', 'CT_CHEST_9.txt': 'CT_CHEST.json', 'MR_BRAIN_5.txt': 'MR_ADULT_BRAIN.json', 'CTA_ABDOMENPELVIS_4.txt': 'CT_ABDOMEN_AND_PELVIS.json', 'CTA_ABDOMENPELVIS_5.txt': 'CT_ABDOMEN_AND_PELVIS.json', 'MR_BRAIN_4.txt': 'MR_ADULT_BRAIN.json', 'CT_CHEST_8.txt': 'CT_CHEST.json', 'MR_BRAIN_10.txt': 'MR_ADULT_BRAIN.json', 'CT_SINUSES_9.txt': 'CT_SINUS.json', 'MR_WHOLEBODY_9.txt': 'MR_ABDOMEN.json', 'MR_BRAIN_6.txt': 'MR_ADULT_BRAIN.json', 'CTA_ABDOMENPELVIS_7.txt': 'CT_ABDOMEN_AND_PELVIS.json', 'CTA_ABDOMENPELVIS_6.txt': 'CT_ABDOMEN_AND_PELVIS.json', 'MR_BRAIN_7.txt': 'MR_ADULT_BRAIN.json', 'MR_WHOLEBODY_8.txt': 'MR_WHOLE_BODY.json', 'CT_SINUSES_8.txt': 'CT_SINUS.json', 'MR_ELBOW_3.txt': 'MR_ELBOW.json', 'MR_SPINE_6.txt': 'MR_LUMBAR_SPINE_LOW_BACK_PAIN.json', 'MR_WRIST_3.txt': 'MR_WRIST.json', 'CT_ABDOMEN_7.txt': 'CT_ABDOMEN_AND_PELVIS.json', 'CT_HEAD_5.txt': 'CT_HEAD_WITH_CONTRAST.json', 'CT_HEAD_4.txt': 'CT_HEAD_WITH_CONTRAST.json', 'CT_ABDOMEN_6.txt': 'CT_ABDOMEN_AND_PELVIS.json', 'MR_WRIST_2.txt': 'MR_WRIST.json', 'MR_SPINE_7.txt': 'MR_LUMBAR_SPINE_LOW_BACK_PAIN.json', 'MR_ELBOW_2.txt': 'MR_WRIST.json', 'CT_SPINE_10.txt': 'CT_SPINE.json', 'CT_SPINE_8.txt': 'CT_SPINE.json', 'MR_SPINE_5.txt': 'MR_COMPLETE_SPINE_DEGENERATIVE_CHANGES.json', 'MR_ANKLE_4.txt': 'MR_ANKLE.json', 'CT_ABDOMEN_4.txt': 'CT_ABDOMEN_AND_PELVIS.json', 'CT_HEAD_6.txt': 'CT_HEAD_WITHOUT_CONTRAST.json', 'MR_ABDOMEN_8.txt': 'MR_ABDOMEN.json', 'MR_ABDOMEN_9.txt': 'MR_ABDOMEN.json', 'CT_HEAD_7.txt': 'CT_HEAD_WITH_CONTRAST.json', 'CT_ABDOMEN_5.txt': 'CT_ABDOMEN_AND_PELVIS.json', 'MR_ANKLE_5.txt': 'MR_ANKLE.json', 'MR_WRIST_1.txt': 'MR_WRIST.json', 'MR_SPINE_4.txt': 'MR_COMPLETE_SPINE_DEGENERATIVE_CHANGES.json', 'CT_SPINE_9.txt': 'CT_SPINE.json', 'MR_ELBOW_1.txt': 'MR_WRIST.json', 'MR_CARDIAC_1.txt': 'MR_ABDOMEN.json', 'MR_SHOULDER_8.txt': 'MR_SHOULDER.json', 'MR_ELBOW_5.txt': 'MR_ELBOW.json', 'MR_WRIST_5.txt': 'MR_WRIST.json', 'MR_ANKLE_1.txt': 'MR_ANKLE.json', 'CT_ABDOMEN_1.txt': 'CT_ABDOMEN_AND_PELVIS.json', 'CT_HEAD_3.txt': 'CT_HEAD_WITHOUT_CONTRAST.json', 'CT_HEAD_2.txt': 'CT_HEAD_STROKE.json', 'MR_WRIST_4.txt': 'MR_WRIST.json', 'MR_SPINE_1.txt': 'MR_LUMBAR_THORACIC_SPINE_DEGENERATIVE_CHANGES.json', 'MR_ELBOW_4.txt': 'MR_ELBOW.json', 'MR_SHOULDER_9.txt': 'MR_SHOULDER.json', 'MR_CARDIAC_2.txt': 'MR_CARDIAC_FUNCTION_AND_VIABILITY.json', 'MR_SPINE_3.txt': 'MR_LUMBAR_THORACIC_SPINE_DEGENERATIVE_CHANGES.json', 'MR_ANKLE_2.txt': 'MR_ANKLE.json', 'CT_ABDOMEN_2.txt': 'CT_ABDOMEN_AND_PELVIS.json', 'MR_KNEE_9.txt': 'MR_KNEE.json', 'CT_HEAD_1.txt': 'CT_HEAD_WITH_CONTRAST.json', 'MR_KNEE_8.txt': 'MR_KNEE.json', 'CT_ABDOMEN_3.txt': 'CT_ABDOMEN_AND_PELVIS.json', 'MR_ANKLE_3.txt': 'MR_ANKLE.json', 'MR_SPINE_2.txt': 'MR_LUMBAR_THORACIC_SPINE_DEGENERATIVE_CHANGES.json'}
    # test_template_mapping(template_assignment)

    # in template assignment, remove those .json and the number of reports, only keep the template name
    
    gt_folder = '../ground_truth'
    n_total = [f for f in os.listdir(output_folder) if f.endswith('.json')]

    count_legal_json = 0

    for filename in tqdm(os.listdir(output_folder)):
        if filename.endswith(".json"):
            # gt_file = os.path.join(gt_folder, filename)
            generated_file = os.path.join(output_folder, filename)

            if not os.path.exists(gt_file):
                print(f"Skipping {filename}, no ground_truth file found.")
                continue

            print(f"\nEvaluating {filename}...")

            '''# 1. Check if the generated JSON files are well-formed
            if not legal_json(generated_file):
                print("The generated report has illegal JSON format.")
                continue
            else:
                count_legal_json += 1

            gt_report = load_json(gt_file)'''
            gen_report = load_json(generated_file)

            # 2. Check the template selection


            '''# 3. Compare the JSON fields
            match, total, field_matches, tp, fp, fn = compare_json_fields(gt_report, gen_report)
            print(f"Total fields: {total}, Matches: {match}, TP: {tp}, FP: {fp}, FN: {fn}")
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

            # 4. Evaluate the reports
            semantic_scores = semantic_evaluate(gt_report, gen_report)
            print("Semantic Evaluation Scores:")
            for field, score in semantic_scores['per_field_scores'].items():
                print(f"{field}: {score}")
            print(f"Macro similarity: {semantic_scores['macro_similarity']:.4f}")'''

            # 3. Flatten the JSON and compare the bert score similarity with freetext
    
    #print(f"\nTotal legal JSON files processed: {count_legal_json}/{len(n_total)}")  