import os
import optuna
import uproot
import re
import numpy as np
import awkward as ak
from weaver.utils.nn.metrics import evaluate_metrics
import datetime
from sklearn.metrics import f1_score, log_loss
import torch
import torch.nn as nn
from weaver.nn.model.ParticleNet import ParticleNet
import subprocess
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd

TRAIN_PATH = r'/eos/cms/store/group/cmst3/user/sesanche/SummerStudent/train_0p01/*.root'
TEST_PATH = r'/eos/cms/store/group/cmst3/user/sesanche/SummerStudent/test_0p01/*.root'

optuna_version = input("Enter number of the Optuna version: ")

os.makedirs(f'optuna_models_{optuna_version}', exist_ok=True)
os.makedirs(f'optuna_logs_{optuna_version}', exist_ok=True)
os.makedirs(f'optuna_outputs_{optuna_version}', exist_ok=True)
os.makedirs(f'optuna_plots_{optuna_version}', exist_ok=True)


def EdgeConv_params(trial):
    num_ec_blocks = trial.suggest_int('num_ec_blocks', 2, 3)
    ec_k = trial.suggest_int('ec_k', 4, 20, step=2)

    num_base_neurons = [16, 32, 64]
    base_channels = trial.suggest_categorical('ec_base_channels', num_base_neurons)

    ec_params = []
    current_channels = base_channels
    for block in range(num_ec_blocks):
        current_block = (current_channels, current_channels, current_channels)
        ec_params.append((ec_k, current_block))
        current_channels *= 2  # double the channels for the next block, creating a funnel

    final_output_channels = current_channels // 2

    return ec_params, final_output_channels

def FullyConnected_params(trial, final_output_channels):
    num_fc_layers = trial.suggest_int('num_fc_layers', 1, 2)
    fc_drop = trial.suggest_float('fc_p', 0.2, 0.5, step=0.1)

    '''num_neurons = [16, 32, 64, 128]
    fc_drop = trial.suggest_float('fc_p', 0.1, 0.5, step=0.1)
    fc_neurons = trial.suggest_categorical('fc_c', num_neurons)

    fc_params = []
    for layer in range(num_fc_layers):
        fc_params.append((fc_neurons, fc_drop))'''

    fc_params = [(final_output_channels, fc_drop)]

    if num_fc_layers == 2:
        second_fc_channels = final_output_channels // 2
        fc_params = [(final_output_channels, fc_drop), (second_fc_channels, fc_drop)]

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

    with open(f'temporary_net_files/particle_net_temp_{optuna_version}_{model_iter}.py', 'w') as f:
        f.write(code_to_write)


def run_training(model_iter, num_epochs, batch_size, lr_scheduler_name, optimizer_name, steps_per_epoch, training_params):
    # Run the training script with the temporary model
    prompt = [
        'weaver',
        '--data-train', TRAIN_PATH,
        # '--data-test', TEST_PATH,
        '--data-config', 'config.yaml',
        '--network-config', f'temporary_net_files/particle_net_temp_{optuna_version}_{model_iter}.py', 
        '--model-prefix', f'optuna_models_{optuna_version}/optuna_model_{model_iter}',
        '--gpus', '0',
        '--batch-size', str(batch_size),
        '--use-amp',
        '--num-epochs', str(num_epochs),
        '--lr-scheduler', lr_scheduler_name,
        '--optimizer', optimizer_name,
        '--steps-per-epoch', str(steps_per_epoch),
        '--log', f'optuna_logs_{optuna_version}/optuna_model_{model_iter}.log',
    ]

    for param, value in training_params.items():
        if value is not None:
            prompt.append(param)
            prompt.append(str(value))

    result = subprocess.run(prompt, capture_output=True, text=True)

    if result.returncode != 0:
        print(f'Training for {model_iter} failed!')
        print('stderr output:')
        print(result.stderr)
        raise RuntimeError(f'Training {model_iter} failed')

    print(result.stdout)


def run_prediction(model_iter, batch_size):
    # Run the prediction script with the trained model
    prompt = [
        'weaver',
        '--predict',
        #'--data-train', TRAIN_PATH,
        '--data-test', TEST_PATH,
        '--data-config', 'config.yaml',
        '--network-config', f'temporary_net_files/particle_net_temp_{optuna_version}_{model_iter}.py',
        '--model-prefix', f'optuna_models_{optuna_version}/optuna_model_{model_iter}',
        '--gpus', '0',
        '--batch-size', str(batch_size),
        '--log', f'optuna_logs_{optuna_version}/optuna_model_{model_iter}_pred.log',
        '--predict-output', f'optuna_outputs_{optuna_version}/optuna_model_{model_iter}_predictions.root',
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
    root_prediction_path = f'optuna_outputs_{optuna_version}/optuna_model_{model_iter}_predictions.root'

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

        ce_loss = log_loss(target, pred_probs)

        print(f"Trial {model_iter}, threshold = {best_threshold}, F1 Score: {f1_score_weaver}, ROC AUC: {roc_auc}")

        # fallback to 0.0 if f1_score is None
        if f1_score_weaver is None:
            f1_score_weaver = 0.0

        return f1_score_weaver, roc_auc, best_threshold, ce_loss

def calculate_F1_best_threshold(model_iter):
    root_prediction_path = f'optuna_outputs_{optuna_version}/optuna_model_{model_iter}_predictions.root'

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

        F1_scores = np.divide(2 * precision[:1] * recall[:1],
                       precision[:1] + recall[:1], out=np.zeros_like(precision[:1]),
                        where=(precision[:1] + recall[:1]) != 0)

        best_idx = np.argmax(F1_scores)
        best_threshold = thresholds[best_idx + 1] # bcz we sliced the array
        best_f1_score = F1_scores[best_idx]

        return best_f1_score, roc_auc, best_threshold


def calculate_cross_entropy(model_iter):
    root_prediction_path = f'optuna_outputs_{optuna_version}/optuna_model_{model_iter}_predictions.root'
    with uproot.open(root_prediction_path) as file:
        events = file["Events"]
        array = events.arrays(library='ak')
        target = ak.to_numpy(array['_label_']).astype(int)
        pred_probs = ak.to_numpy(array['score_label_sample1'])
        return ce_loss


def parse_validation_metrics(log_path, model_iter):
    roc_auc_validation_history = []
    ce_loss_validation_history = []
    roc_auc_match = '- roc_auc_score_matrix:'
    ce_loss_match = '- roc_auc_score:'

    with open(log_path, 'r') as f:
        lines = f.readlines()

        for i in range(len(lines)):
            if roc_auc_match in lines[i]:
                if i+1 < len(lines):
                    next_line = lines[i+1]
                    val_metric = float(next_line.strip())
                    roc_auc_validation_history.append(val_metric)
            if ce_loss_match in lines[i]:
                if i+1 < len(lines):
                    next_line = lines[i+1]
                    line = float(next_line.strip())
                    ce_loss_validation_history.append(line)

    print('ROC AUC Validation History:', roc_auc_validation_history)
    print('Cross Entropy Loss Validation History:', ce_loss_validation_history)

    df = pd.DataFrame({
        'roc_auc': roc_auc_validation_history,
        'cross_entropy': ce_loss_validation_history
    })

    # save df
    df.to_csv(f'optuna_plots_{optuna_version}/validation_history_{model_iter}.csv', index=False)

    return df


def plot_validation_history(history, model_iter):
    roc_auc_plot_path = f'optuna_plots_{optuna_version}/trial_{model_iter}_roc_auc_validation_history.png'
    ce_loss_plot_path = f'optuna_plots_{optuna_version}/trial_{model_iter}_cross_entropy_validation_history.png'

    df = pd.DataFrame(history, columns=['roc_auc', 'cross_entropy'])
    roc_auc_history = df['roc_auc'].tolist()
    ce_loss_history = df['cross_entropy'].tolist()

    epochs = range(1, len(roc_auc_history) + 1)
    plt.rcParams['font.size'] = 18

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, roc_auc_history, 'ro-', label='ROC AUC')
    plt.title(f'Trial {model_iter}: ROC AUC vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('ROC AUC Score')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(roc_auc_plot_path)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, ce_loss_history, 'ro-')
    plt.title(f'Trial {model_iter}: Cross Entropy vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ce_loss_plot_path)
    plt.close()


def objective(trial):
    # Define the hyperparameters for the model
    ec_params, base_channels = EdgeConv_params(trial)
    fc_params = FullyConnected_params(trial, base_channels)

    # Number of epochs
    num_epochs = 25

    # Schedulers
    lr_scheduler_name = trial.suggest_categorical('lr_scheduler',
        ['steps', 'flat+decay', 'flat+linear', 'flat+cos', 'one-cycle'])

    # Optimizers
    optimizer_name = trial.suggest_categorical('optimizer', ['radam', 'ranger'])

    # Batch size
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])

    # Number of steps per epoch
    num_train_samples = 580352
    steps_per_epoch = num_train_samples // batch_size

    # Add other training parameters depending on the scheduler
    other_training_params = {}
    # depending on the scheduler, set the learning rate and warmup steps
    if lr_scheduler_name == 'one-cycle': # one-cycle does not use warmup
        start_lr = trial.suggest_float('start_lr', 1e-5, 1e-3, log=True)
        other_training_params['--start-lr'] = start_lr
    else:
        start_lr = trial.suggest_float('start_lr', 1e-6, 1e-4, log=True)
        other_training_params['--start-lr'] = start_lr
        if lr_scheduler_name in ['flat+decay', 'flat+linear', 'flat+cos', 'steps']:
            warmup_epochs = trial.suggest_int('warmup_epochs', 0, 4, step=1)
            other_training_params['--warmup-steps'] = warmup_epochs


    # Write the temporary model file with the current hyperparameters
    model_iter = trial.number
    write_temporary_model(ec_params, fc_params, model_iter)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{now}] Running trial number: {model_iter}")

    try:
        run_training(model_iter, num_epochs, batch_size, lr_scheduler_name, optimizer_name, steps_per_epoch, other_training_params)

        #HISTORY
        # Define the path to the log file
        log_path = f'optuna_logs_{optuna_version}/optuna_model_{model_iter}.log'

        # Call the parser to read the file and extract the history
        history_df = parse_validation_metrics(log_path, model_iter)
        best_val_roc_auc = max(history_df['roc_auc'])
        best_val_ce_loss = min(history_df['cross_entropy'])

        # Call the plotter to save a visualization of the history
        plot_validation_history(history_df, model_iter)

        # RUN PREDICTION
        run_prediction(model_iter, batch_size)


    except Exception as e:
        print(f"Error during training or prediction for trial {model_iter}: {e}")
        best_val_roc_auc = 0.0
        best_val_ce_loss = float('inf')

    # Append the best trial into csv file
    path_csv_results = f'optuna_logs_{optuna_version}/optuna_best_trial_results.csv'
    file_exists = os.path.isfile(path_csv_results)

    columns = ['trial_number', 'batch_size', 'roc_auc', 'cross_entropy', 'ec_params', 'fc_params', 'optimizer', 'lr_scheduler', 'start_lr', 'timestamp']

    row = {
        'timestamp': now,
        'trial_number': model_iter,
        'batch_size': batch_size,
        'roc_auc': best_val_roc_auc,
        'cross_entropy': best_val_ce_loss,
        'ec_params': str(ec_params),
        'fc_params': str(fc_params),
        'optimizer': optimizer_name,
        'lr_scheduler': lr_scheduler_name,
        'start_lr': other_training_params.get('--start-lr'),
    }
    # add warmup steps if they exist
    for key, value in other_training_params.items():
        if key == '--warmup-steps':
            row['warmup_steps'] = value


    with open(path_csv_results, 'a', buffering=1) as f:
        if not file_exists:
            f.write(','.join(columns) + '\n')
        f.write(','.join(str(row[col]) for col in columns) + '\n')
        f.flush()

    return best_val_roc_auc, best_val_ce_loss


def main():
    study = optuna.create_study(
        directions=['maximize', 'minimize'], 
        study_name='ParticleNet_Optuna_Search')
    
    study.optimize(objective, n_trials=20, n_jobs=1)


if __name__ == "__main__":
    main()