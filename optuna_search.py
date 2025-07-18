import os
import optuna
import uproot
import numpy as np
import awkward as ak
from weaver.utils.nn.metrics import evaluate_metrics
import datetime
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from weaver.nn.model.ParticleNet import ParticleNet
import subprocess
from sklearn.metrics import roc_curve, auc

TRAIN_PATH = r'/eos/cms/store/group/cmst3/user/sesanche/SummerStudent/train_0p01/*.root'
TEST_PATH = r'/eos/cms/store/group/cmst3/user/sesanche/SummerStudent/test_0p01/*.root'

os.makedirs('optuna_models_2', exist_ok=True)
os.makedirs('optuna_logs_2', exist_ok=True)
os.makedirs('optuna_outputs_2', exist_ok=True)


def EdgeConv_params(trial):
    num_ec_blocks = trial.suggest_int('num_ec_blocks', 1, 3)
    ec_k = trial.suggest_int('ec_k', 6, 18, step=6)
    num_neurons = [32, 64, 128]

    ec_params = []
    for block in range(num_ec_blocks):
        ec_c = trial.suggest_categorical(f'ec_c_{block}', num_neurons)
        ec_params.append((ec_k, (ec_c, ec_c, ec_c)))

    return ec_params

def FullyConnected_params(trial):
    num_fc_layers = trial.suggest_int('num_fc_layers', 1, 2)
    num_neurons = [64, 128]
    fc_drop = trial.suggest_float('fc_p', 0.1, 0.5, step=0.1)
    fc_neurons = trial.suggest_categorical('fc_c', num_neurons)

    fc_params = []
    for layer in range(num_fc_layers):
        fc_params.append((fc_neurons, fc_drop))
    
    return fc_params


def write_temporary_model(ec_params, fc_params, model_iter):
    # Write a variation of the particle_net.py model with varied parameters for Optuna to use
    code_to_write = f"""
import torch
import torch.nn as nn
from weaver.nn.model.ParticleNet import ParticleNet

def get_model(data_config, **kwargs):

    conv_params = {repr(ec_params)}
    fc_params = {repr(fc_params)}
    
    use_fusion = True
    features_dims = len(data_config.input_dicts['features'])
    num_classes = len(data_config.label_value)

    model = ParticleNet(features_dims, num_classes, 
                        conv_params, fc_params,
                        use_fusion=use_fusion,
                        use_fts_bn=kwargs.get('use_fts_bn', False),
                        use_counts=kwargs.get('use_counts', True),
                        for_inference=kwargs.get('for_inference', False)
                        )
    model_info = {{
        'input_names': list(data_config.input_names),
        'input_shapes': {{k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()}},
        'output_names': ['softmax'],
        'dynamic_axes': {{**{{k: {{0: 'N', 2: 'n_' + k.split('_')[0]}} for k in data_config.input_names}}, **{{'softmax': {{0: 'N'}}}}}},
    }}

    return model, model_info
"""

    '''format_code_to_write = code_to_write.format(
        conv_params=repr(ec_params),
        fc_params=repr(fc_params))'''

    with open(f'particle_net_temp_2_{model_iter}.py', 'w') as f:
        f.write(code_to_write)


def run_training(model_iter):
    # Run the training script with the temporary model
    prompt = [
        'weaver',
        '--data-train', TRAIN_PATH,
        '--data-test', TEST_PATH,
        '--data-config', 'config.yaml',
        '--network-config', f'particle_net_temp_2_{model_iter}.py', # change this - move to a folder
        '--model-prefix', f'optuna_models_2/optuna_model_{model_iter}',
        '--gpus', '0',
        '--batch-size', '256',
        '--start-lr', '5e-3',
        '--num-epochs', '25',
        '--optimizer', 'ranger',
        '--log', f'optuna_logs_2/optuna_model_{model_iter}.log',
    ]

    result = subprocess.run(prompt, capture_output=True, text=True)

    if result.returncode != 0:
        print(f'Training for {model_iter} failed!')
        print('stderr output:')
        print(result.stderr)
        raise RuntimeError(f'Training {model_iter} failed')

    print(result.stdout)


def run_prediction(model_iter):
    # Run the prediction script with the trained model
    prompt = [
        'weaver',
        '--predict',
        '--data-train', TRAIN_PATH,
        '--data-test', TEST_PATH,
        '--data-config', 'config.yaml',
        '--network-config', f'particle_net_temp_2_{model_iter}.py',
        '--model-prefix', f'optuna_models_2/optuna_model_{model_iter}',
        '--gpus', '0',
        '--batch-size', '256',
        '--log', f'optuna_logs_2/optuna_model_{model_iter}_pred.log',
        '--predict-output', f'optuna_outputs_2/optuna_model_{model_iter}_predictions.root',
    ]

    result = subprocess.run(prompt, capture_output=True, text=True)

    if result.returncode != 0:
        print(f'Prediction for {model_iter} failed!')
        print('stderr output:')
        print(result.stderr)
        print('result return code:', result.returncode)
        raise RuntimeError(f'Prediction {model_iter} failed')

    print(result.stdout)


def calculate_test_scores(model_iter):
    # Calculate test metrics from the predictions stored in root files
    root_prediction_path = f'optuna_outputs_2/optuna_model_{model_iter}_predictions.root'

    with uproot.open(root_prediction_path) as file:
        events = file["Events"]
        array = events.arrays(library='ak')

        target = ak.to_numpy(array['_label_']).astype(int)

        # get probabilities 
        pred_probs = ak.to_numpy(array['score_label_sample1'])

        # roc curve
        fpr, tpr, thresholds = roc_curve(target, pred_probs)
        roc_auc = auc(fpr, tpr)
        best_idx = np.argmax(tpr - fpr)
        best_threshold = thresholds[best_idx]

        # update the predictions with the best threshold
        pred_classes = (pred_probs > best_threshold).astype(int)

        metrics_classes = evaluate_metrics(y_true=target, y_score=pred_classes, eval_metrics=['accuracy_score', 'f1_score', 'confusion_matrix'])

        f1_score_weaver = metrics_classes['f1_score']

        print(f"Trial {model_iter}, threshold = {best_threshold}, F1 Score: {f1_score_weaver}, ROC AUC: {roc_auc}")

        # fallback to 0.0 if f1_score is None
        if f1_score_weaver is None:
            f1_score_weaver = 0.0

        return f1_score_weaver, roc_auc, best_threshold


def objective(trial):
    # Define the hyperparameters for the model
    ec_params = EdgeConv_params(trial)
    fc_params = FullyConnected_params(trial)

    model_iter = trial.number
    write_temporary_model(ec_params, fc_params, model_iter)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{now}] Running trial number: {model_iter}")

    try:
        run_training(model_iter)
        run_prediction(model_iter)
        f1_score, roc_auc, threshold = calculate_test_scores(model_iter)

    except Exception as e:
        print(f"Error during training or prediction for trial {model_iter}: {e}")
        f1_score = 0.0
        threshold = 0.0
        roc_auc = 0.0
    
    # Append the best trial into csv file
    path_csv_results = 'optuna_logs_2/optuna_best_trial_results.csv'
    file_exists = os.path.isfile(path_csv_results)

    columns = ['trial_number', 'f1_score', 'ec_params', 'fc_params', 'threshold', 'timestamp']

    row = {
        'trial_number': model_iter,
        'f1_score': f1_score,
        'roc_auc': roc_auc,
        'ec_params': str(ec_params),
        'fc_params': str(fc_params),
        'threshold': threshold,
        'timestamp': now
    }

    with open(path_csv_results, 'a', buffering=1) as f:
        if not file_exists:
            f.write(','.join(columns) + '\n')
        f.write(','.join(str(row[col]) for col in columns) + '\n')
        f.flush()

    return f1_score, roc_auc


def main():
    study = optuna.create_study(
        directions=['maximize', 'maximize'], 
        study_name='ParticleNet_Optuna_Search')
    
    study.optimize(objective, n_trials=10, n_jobs=1)

    best_trial = study.best_trial

    # save the best trial to csv
    with open("optuna_logs_2/optuna_best_trial_summary.csv", "w") as f:
        f.write(f"Best trial number: {best_trial.number}\n")
        f.write(f"Best trial value (F1 score): {best_trial.value}\n")
        f.write(f"Best trial params: {best_trial.params}\n")

    # save all trials to csv
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df.to_csv("optuna_logs_2/optuna_trials_summary.csv", index=False)

if __name__ == "__main__":
    main()