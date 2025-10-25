import warnings
import numpy as np
import pandas as pd
import simple_icd_10_cm as cm 
import os
from collections import defaultdict, Counter
import json

def parse_icd_levels(code: str) -> dict:
    """
    Parses an ICD code into its hierarchical levels.
    Levels:
        - chapter: first character
        - block: first two characters
        - category: first three characters
        - subcategory: up to the first period (.)
        - extension1: up to 1 character after the period
        - extension2: the full code
    """
    levels = {}

    if len(code) >= 1:
        levels['chapter'] = code[:1]
    if len(code) >= 2:
        levels['block'] = code[:2]
    if len(code) >= 3:
        levels['category'] = code[:3]
    if len(code) >= 4:
        levels['subcategory'] = code[:4]
    if len(code) >= 5:
        levels['extension1'] = code[:5]
    if len(code) >= 6: # This assumes 'extension2' is the full code if it's 6 or more characters
        levels['extension2'] = code
    elif len(code) >= 1 and len(code) < 6:
        pass

    return levels


def compute_icd_hierarchy_metrics(df_gs: pd.DataFrame, df_pred: pd.DataFrame) -> dict:
    """
    Computes micro-averaged precision, recall, and F1 score for each ICD code level.

    Args:
        df_gs (pd.DataFrame): Gold standard DataFrame with 'clinical_case' and 'code'.
        df_pred (pd.DataFrame): Predictions DataFrame with 'clinical_case' and 'code'.

    Returns:
        dict: Dictionary with metrics for each level.
    """
    gs_grouped = df_gs.groupby('clinical_case')['code'].apply(set).to_dict()
    pred_grouped = df_pred.groupby('clinical_case')['code'].apply(set).to_dict()
    all_cases = set(gs_grouped) | set(pred_grouped)

    levels = ['chapter', 'block', 'category', 'subcategory', 'extension1', 'extension2']
    metrics = {lvl: {'TP': 0, 'FP': 0, 'FN': 0} for lvl in levels}
    
    for case in all_cases:
        gs_codes = gs_grouped.get(case, set())
        pred_codes = pred_grouped.get(case, set())

        gs_levels = defaultdict(list)
        pred_levels = defaultdict(list)

        for code in gs_codes:
            parsed = parse_icd_levels(code)
            for lvl_name, lvl_value in parsed.items(): 
                gs_levels[lvl_name].append(lvl_value)

        for code in pred_codes:
            parsed = parse_icd_levels(code)
            for lvl_name, lvl_value in parsed.items(): 
                pred_levels[lvl_name].append(lvl_value)
    
        # Compare at each level
        for lvl in levels:
            gs_counter = Counter(gs_levels[lvl])
            pred_counter = Counter(pred_levels[lvl])

            tp = 0
            for item in gs_counter:
                tp += min(gs_counter[item], pred_counter[item])

            fp = 0
            for item in pred_counter:
                fp += max(0, pred_counter[item] - gs_counter[item])

            fn = 0
            for item in gs_counter:
                fn += max(0, gs_counter[item] - pred_counter[item])

            metrics[lvl]['TP'] += tp
            metrics[lvl]['FP'] += fp
            metrics[lvl]['FN'] += fn

    # Compute micro-averaged precision, recall, F1
    results = {}
    for lvl in levels:
        TP = metrics[lvl]['TP']
        FP = metrics[lvl]['FP']
        FN = metrics[lvl]['FN']
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall + 1e-10)
        results[f'{lvl}_precision'] = precision
        results[f'{lvl}_recall'] = recall
        results[f'{lvl}_f1_score'] = f1

    return results

def read_gs(gs_path: str, valid_codes: set) -> pd.DataFrame:
    """Read the gold-standard labels and filter to select the set of valid codes."""
    gs_data = pd.read_csv(gs_path, sep="\t", names=['clinical_case', 'code'], dtype={'clinical_case': object, 'code': object})
    gs_data.code = gs_data.code.str.lower()
    return gs_data

def read_run(pred_path: str, valid_codes: set) -> pd.DataFrame:
    run_data = pd.read_csv(pred_path, sep="\t", names=['clinical_case', 'code'], dtype={'clinical_case': object, 'code': object})
    run_data.code = run_data.code.str.lower()
    if run_data.shape[0] == 0:
        warnings.warn('None of the predicted codes are considered valid codes')
    return run_data

def calculate_metrics(df_gs: pd.DataFrame, df_pred: pd.DataFrame) -> tuple[float]:
    pred_per_cc = df_pred.drop_duplicates(subset=['clinical_case', "code"]).groupby("clinical_case")["code"].count()
    Pred_Pos = df_pred.drop_duplicates(subset=['clinical_case', "code"]).shape[0]
    true_per_cc = df_gs.drop_duplicates(subset=['clinical_case', "code"]).groupby("clinical_case")["code"].count()
    GS_Pos = df_gs.drop_duplicates(subset=['clinical_case', "code"]).shape[0]
    cc = set(df_gs.clinical_case.tolist())
    TP_per_cc = pd.Series(dtype=float)
    for c in cc:
        pred = set(df_pred.loc[df_pred['clinical_case'] == c, 'code'].values)
        gs = set(df_gs.loc[df_gs['clinical_case'] == c, 'code'].values)
        TP_per_cc[c] = len(pred.intersection(gs))
    TP = sum(TP_per_cc.values)
    precision_per_cc = TP_per_cc / pred_per_cc
    recall_per_cc = TP_per_cc / true_per_cc
    f1_score_per_cc = (2 * precision_per_cc * recall_per_cc) / (precision_per_cc + recall_per_cc + 1e-10)
    precision = TP / Pred_Pos
    recall = TP / GS_Pos
    f1_score = (2 * precision * recall) / (precision + recall + 1e-10)
    return precision_per_cc, precision, recall_per_cc, recall, f1_score_per_cc, f1_score

def compute_macro_averaged_scores(df_gs: pd.DataFrame, df_run: pd.DataFrame) -> tuple[float]:
    codes = set(df_gs.code)
    precisions, recalls, f1_scores = [], [], []
    for code in codes:
        true_cases = df_gs[df_gs.code == code]
        pred_cases = df_run[df_run.code == code]
        true_positive_count = len(set(pred_cases.clinical_case).intersection(set(true_cases.clinical_case)))
        precision = true_positive_count / len(pred_cases) if len(pred_cases) > 0 else 0 
        recall = true_positive_count / len(true_cases) if len(true_cases) > 0 else 0 
        f1_score = 2 * precision * recall / (precision + recall) if precision > 0 and recall > 0 else 0
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    # Handle empty lists to avoid warnings/errors if no codes result in non-zero metrics
    macro_precision = np.mean(precisions) if precisions else 0.0
    macro_recall = np.mean(recalls) if recalls else 0.0
    macro_f1 = np.mean(f1_scores) if f1_scores else 0.0
    return macro_precision, macro_recall, macro_f1


def run_evaluation(dataset, experiment_name, language):

    df_gs = pd.read_csv(f"data/gt_codes/{dataset}.tsv", sep="\t")
    df_gs.to_csv("data/gt_test.tsv", sep="\t", index=False) 

    try:
        valid_codes = set([x.lower() for x in cm.get_all_codes() if cm.is_leaf(x)])
    except AttributeError:
        warnings.warn("simple_icd_10_cm might not be fully initialized or installed. Proceeding without `is_leaf` filter.")
        valid_codes = set([x.lower() for x in cm.get_all_codes()])
    except Exception as e:
        warnings.warn(f"Could not load valid codes from simple_icd_10_cm: {e}. Proceeding without valid codes.")
        valid_codes = set() 

    gs_path = "data/gt_test.tsv"
    pred_path = f"data/output/experiment_{experiment_name}/{dataset}/{language}/icd_predictions.tsv"
    df_gs = read_gs(gs_path, valid_codes)
    df_run = read_run(pred_path, valid_codes)

    # compute metrics
    precision_per_cc, precision, recall_per_cc, recall, f1_per_cc, f1_score = calculate_metrics(df_gs, df_run)
    macro_precision, macro_recall, macro_f1 = compute_macro_averaged_scores(df_gs, df_run)
    icd_hierarchy_results = compute_icd_hierarchy_metrics(df_gs, df_run)

    output_data = {}

    # MICRO-AVERAGE STATISTICS
    output_data['micro_average_statistics'] = {
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'f1_score': round(f1_score, 3)
    }

    # MACRO-AVERAGE STATISTICS
    output_data['macro_average_statistics'] = {
        'precision': round(macro_precision, 3),
        'recall': round(macro_recall, 3),
        'f1_score': round(macro_f1, 3)
    }

    # # PER CLINICAL CASE STATISTICS
    # output_data['per_clinical_case_statistics'] = {}
    # for cc in df_gs.clinical_case.unique():
    #     output_data['per_clinical_case_statistics'][cc] = {
    #         'precision': round(precision_per_cc[cc], 3),
    #         'recall': round(recall_per_cc[cc], 3),
    #         'f1_score': round(f1_per_cc[cc], 3)
    #     }

    # ICD HIERARCHY MICRO-AVERAGE METRICS
    output_data['icd_hierarchy_micro_average_metrics'] = {}
    for level in ['chapter', 'block', 'category', 'subcategory', 'extension1', 'extension2']:
        output_data['icd_hierarchy_micro_average_metrics'][level] = {
            'precision': round(icd_hierarchy_results[f'{level}_precision'], 3),
            'recall': round(icd_hierarchy_results[f'{level}_recall'], 3),
            'f1_score': round(icd_hierarchy_results[f'{level}_f1_score'], 3)
        }

    output_dir = f"data/output/experiment_{experiment_name}/{dataset}/{language}"
    output_filename = os.path.join(output_dir, "eval_metrics.json")

    os.makedirs(output_dir, exist_ok=True)

    # Save the data to a JSON file
    with open(output_filename, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Metrics saved to: {output_filename}")

    # Clear the temporary gt tsv file
    os.remove("data/gt_test.tsv")
