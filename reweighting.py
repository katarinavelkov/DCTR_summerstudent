import uproot
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

optuna_version = '32'  # Change this to the version of optuna run
number_of_models = 1  # Total number of models trained in that optuna run
sample_number = 1  # Change this to the sample number you want to plot, 1 or 2

for num in range(number_of_models):
    print('-----------------------------------------------------------')
    print(f"Processing model {num} from optuna version {optuna_version}")
    root_file_path = f"optuna_outputs_{optuna_version}/optuna_model_{num}_predictions.root"

    # Open the ROOT file and access the Events tree
    with uproot.open(root_file_path) as file:
        tree = file["Events"]
        
        # Load the desired variable sample
        score = tree[f"score_label_sample{sample_number}"].array(library="np")
        label = tree[f"label_sample{sample_number}"].array(library="np") # target value
        print(score[:5], score.shape)
        print(label[:5], label.shape)

        array = tree.arrays(library='ak')
        target = ak.to_numpy(array['_label_']).astype(int)
        # get probabilities 
        logits = ak.to_numpy(array[f'score_label_sample{sample_number}']) # they go from 0.5 to 1

        pred_probs_sigmoid = sigmoid(logits)
        print(pred_probs_sigmoid[:5], pred_probs_sigmoid.shape)


    # Separate the scores based on label
    score_0 = score[label == 0] # predicted logits for class 1
    score_1 = score[label == 1] # predicted logits for class 0

    score_0 = sigmoid(score_0) # predicted probabilities for class 1 !!!!!!!!!!
    score_1 = sigmoid(score_1) # predicted probabilities for class 0 !!!!!!!!!!

    # reweighting of one class to the chosen target distribution!
    epsilon = 1e-10
    #weights_0 = score_0 / (1 - score_0 + epsilon)
    weights_0 = (score_0) / (1-score_0) # predstavlja reweight iz 1 v 0, ker je obrnjeno na glavo

    # Plotting distribution
    plt.figure(figsize=(8, 6))

    all_scores = np.concatenate((score_0, score_1))
    min_score = np.min(all_scores)
    max_score = np.max(all_scores)

    num_bins = 50
    bins = np.linspace(min_score, max_score, num_bins+1)

    # Original class 0
    plt.hist(score_0, bins=bins, density=True, alpha=0.6, label='Class 0', color='blue')

    # Target class 1
    plt.hist(score_1, bins=bins, density=True, alpha=0.6, label='Class 1', color='red')
    
    # Reweighted class 0
    plt.hist(score_0, bins=bins, weights=weights_0, density=True, label='Reweighted Class 1 to 0', histtype='step', linestyle='--', edgecolor='black', linewidth=1.5)
    #plt.hist(score_1, bins=bins, weights=weights_1, density=True, label='Reweighted Class 1 with weights_1, new reweighting', histtype='step', linestyle='--', edgecolor='black', linewidth=1.5)
    
    plt.xlabel(f"Classifier output for sample{sample_number}")
    plt.ylabel("Density")
    plt.title(f"Distribution of score_label_sample{sample_number}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"distribution_plots_correct/reweighted_plot_optuna_{optuna_version}_{num}_sample{sample_number}_1_to_0_sigmoid_sergio.png")
    #plt.savefig(f'distribution_plots/reweighted_plot_optuna_{optuna_version}_{num}_sample{sample_number}_1_to_0_using_weights_0.pdf')
    plt.close()