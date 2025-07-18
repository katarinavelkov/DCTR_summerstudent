import uproot
import numpy as np
from weaver.utils.nn.metrics import evaluate_metrics
import awkward as ak

#filename = 'predict_output/output.root'
filename = 'optuna_outputs/optuna_model_1_predictions.root'

with uproot.open(filename) as f:
    events = f['Events']
    array = events.arrays(library='ak')

    target = ak.to_numpy(array['_label_']).astype(int)

    # get probabilities
    pred_probs = ak.to_numpy(array['score_label_sample1'])

    # If multi-class predictions, otherwise skip this step
    # pred_probs = np.vstack([1 - pred_probs, pred_probs]).T 
    
    # Convert probabilities to binary predictions
    pred_classes = pred_probs > 0.63
    pred_classes = pred_classes.astype(int)

    # To check stuff
    '''for t, p in zip(target[:10], pred_classes[:10]):
        print(f"True: {t}, Pred: {p}")
    
    # Check shapes
    print('probs, target, classes:', pred_probs.shape, target.shape, pred_classes.shape)'''

    # Calculate metrics, probs are not calculated correctly, as the span goes from 0.5 to 1!!!
    metrics_probs = evaluate_metrics(y_true=target, y_score=pred_probs, eval_metrics=['roc_auc_score', 'average_precision_score'])
    metrics_classes = evaluate_metrics(y_true=target, y_score=pred_classes, eval_metrics=['accuracy_score', 'f1_score', 'confusion_matrix'])

    print("Metrics from probabilities:", metrics_probs)
    print("Metrics from classes:", metrics_classes)