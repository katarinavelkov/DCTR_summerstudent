import uproot
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import awkward as ak

file1 = r'optuna_outputs/optuna_model_0_predictions.root'
file2 = r'optuna_outputs/optuna_model_1_predictions.root'

def load_predictions(filename):
    with uproot.open(filename) as f:
        events = f['Events']
        array = events.arrays(library='ak')
        predictions = ak.to_numpy(array['label_sample1'])
        scores = ak.to_numpy(array['score_label_sample1'])
        target = ak.to_numpy(array['_label_']).astype(int)

        predictions_2 = ak.to_numpy(array['label_sample2'])
        scores_2 = ak.to_numpy(array['score_label_sample2'])

        print('Predicted label_sample_1:', predictions[:10])
        print('Predicted score_label_sample_1:', scores[:10])
        print('Predicted label_sample_2:', predictions_2[:10])
        print('Predicted score_label_sample_2:', scores_2[:10])
        print('True labels:', target[:10])

for file in [file1, file2]:
    print(f'Loading predictions from {file}')
    load_predictions(file)

