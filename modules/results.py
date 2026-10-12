import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from tqdm.auto import tqdm

def extract_and_store_metrics(y_test, y_pred_class, y_pred_cont, temp_results, window_idx):
    y_test = np.asarray(y_test)
    y_pred_class = np.asarray(y_pred_class)
    y_pred_cont = np.asarray(y_pred_cont)

    accuracy = (y_test == y_pred_class).mean()

    tp = ((y_test == 1) & (y_pred_class == 1)).sum()
    fp = ((y_test == 0) & (y_pred_class == 1)).sum()
    fn = ((y_test == 1) & (y_pred_class == 0)).sum()
    tn = ((y_test == 0) & (y_pred_class == 0)).sum()

    precision_1 = tp / (tp + fp + 1e-8)
    recall_1 = tp / (tp + fn + 1e-8)
    f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1 + 1e-8)

    precision_0 = tn / (tn + fn + 1e-8)
    recall_0 = tn / (tn + fp + 1e-8)
    f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0 + 1e-8)

    try:
        auc = roc_auc_score(y_test, y_pred_cont) if not np.any(np.isnan(y_pred_cont)) else 0.0
    except ValueError:
        auc = 0.5

    temp_results.append({
        "Idx": window_idx,
        "Accuracy": accuracy,
        "Precision_1": precision_1,
        "Recall_1": recall_1,
        "F1_1": f1_1,
        "Precision_0": precision_0,
        "Recall_0": recall_0,
        "F1_0": f1_0,
        "AUC": auc
    })


def display_fold_results(model_label, fold_idx, test_idx, y_test, y_pred_class, y_pred_cont):
    if hasattr(y_test, 'cpu'):
        y_test = y_test.cpu().numpy()
    if hasattr(y_pred_class, 'cpu'):
        y_pred_class = y_pred_class.cpu().numpy()
    if hasattr(y_pred_cont, 'cpu'):
        y_pred_cont = y_pred_cont.cpu().numpy()

    model_name = f"FOLD {fold_idx} | {test_idx}"
    tqdm.write(f"\n## [ {model_name} ]")

    report_text = classification_report(y_test, y_pred_class, digits=4)
    tqdm.write("\n### Classification Report\n" + report_text)

    cm = confusion_matrix(y_test, y_pred_class)
    tqdm.write("\n### Confusion Matrix")
    for row in cm:
        tqdm.write('\t'.join(map(str, row)))

    try:
        auc = roc_auc_score(y_test, y_pred_cont)
        tqdm.write(f"\n### ROC AUC\n{auc:.4f}")
    except ValueError:
        tqdm.write(f"\n### ROC AUC\n[WARNING] AUC not defined")

def summarize_kfold_results(temp_results, model_name):
    temp_df = pd.DataFrame(temp_results)
    if temp_df.empty:
        return pd.DataFrame([{'Model': model_name}])
    summary = temp_df.mean(numeric_only=True)
    final_result = {
        'Model': model_name,
        'Accuracy': summary.get('Accuracy'),
        'Precision_1': summary.get('Precision_1'),
        'Recall_1': summary.get('Recall_1'),
        'F1_1': summary.get('F1_1'),
        'Precision_0': summary.get('Precision_0'),
        'Recall_0': summary.get('Recall_0'),
        'F1_0': summary.get('F1_0'),
        'AUC': summary.get('AUC'),
    }
    return pd.DataFrame([final_result])


