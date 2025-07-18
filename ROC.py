import uproot
import awkward as ak
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

prediction_file = 'optuna_outputs/optuna_model_3_predictions.root'

def calculate_test_scores():
    pass

def calculate_roc():

    with uproot.open(prediction_file) as file:
        events = file["Events"]
        array = events.arrays(library='ak')
        target = ak.to_numpy(array['_label_']).astype(int)

        # get probabilities 
        pred_probs = ak.to_numpy(array['score_label_sample1']) # they go from 0.5 to 1

        # roc curve
        fpr, tpr, thresholds = roc_curve(target, pred_probs)
        print(f'Number of thresholds: {len(thresholds)}')
        print(f'Number of false positive rates: {len(fpr)}')
        print(f'Number of true positive rates: {len(tpr)}')
        print(f'First 5 thresholds: {thresholds[:5]}')
        print(f'First 5 false positive rates: {fpr[:5]}')
        print(f'First 5 true positive rates: {tpr[:5]}')
        roc_auc = auc(fpr, tpr)
        best_idx = np.argmax(tpr - fpr)
        best_threshold = thresholds[best_idx]

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('roc_curve_optuna_model_3.png')
        print(f'ROC AUC: {roc_auc:.2f}')
        print(f'Best threshold: {best_threshold}')

calculate_roc()