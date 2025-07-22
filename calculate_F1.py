import os
import uproot
import numpy as np
import awkward as ak
from weaver.utils.nn.metrics import evaluate_metrics
import datetime
from sklearn.metrics import f1_score
import subprocess
from sklearn.metrics import roc_curve, auc


def calculate_F1_best_threshold():
    root_prediction_path = f'optuna_outputs_2/optuna_model_0_predictions.root'

    with uproot.open(root_prediction_path) as file:
        events = file["Events"]
        array = events.arrays(library='ak')

        target = ak.to_numpy(array['_label_']).astype(int)

        # get probabilities 
        pred_probs = ak.to_numpy(array['score_label_sample1'])

        # roc curve
        fpr, tpr, thresholds = roc_curve(target, pred_probs)
        roc_auc = auc(fpr, tpr)

        P = np.sum(target == 1)
        N = np.sum(target == 0)

        TP = tpr * P
        FP = fpr * N

        precision = np.divide(TP, (TP + FP), out=np.zeros_like(TP), where=(TP + FP) != 0)
        recall = tpr  # Recall is the same as TPR

        F1_scores = np.divide(2 * precision[1:] * recall[1:],
                       precision[1:] + recall[1:], out=np.zeros_like(precision[1:]),
                        where=(precision[1:] + recall[1:]) != 0)

        print(f'F1 scores: {F1_scores[:5]}')
        print(f'Precision: {precision[:5]}')
        print(f'Recall: {recall[:5]}')
        print(f'Thresholds: {thresholds[:5]}')

        best_idx = np.argmax(F1_scores) # best index in the sliced array
        best_threshold = thresholds[best_idx + 1] # bcz we sliced the array
        best_f1_score = F1_scores[best_idx]

        print(f'Best F1 Score: {best_f1_score}, Best Threshold: {best_threshold}')
        print(f'ROC AUC: {roc_auc}')

        return best_f1_score, roc_auc, best_threshold

calculate_F1_best_threshold()
