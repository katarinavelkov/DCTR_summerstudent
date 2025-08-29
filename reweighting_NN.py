import uproot
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


root_file_path = '/eos/user/k/kvelkov/DCTR_summerstudent/NN_outputs/nn_output_32_3.root'

# Open the ROOT file and access the Events tree
with uproot.open(root_file_path) as file:
    tree = file["Events"]
    print('Top keys in the tree:', tree.keys())

    prediction_probabilities = tree['nn_output'].array(library="np")
    target = tree['label_sample1'].array(library="np")
    prev_scores = tree['score_label_sample1'].array(library="np")
    print('Prediction:', prediction_probabilities[:5])
    print('Target:', target[:5])
    print('Previous scores:', prev_scores[:5])

    score_0 = prediction_probabilities[target == 0] # predicted probabilities for class 1
    score_1 = prediction_probabilities[target == 1] # predicted probabilities for class 0

    # reweighting of one class to the chosen target distribution!
    epsilon = 1e-10
    #weights_0 = score_0 / (1 - score_0 + epsilon)
    weights_0 = (1 - score_0) / (score_0 + epsilon)
    weights_1 = (1 - score_1) / (score_1 + epsilon)

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
    #plt.hist(score_0, bins=bins, weights=weights_0, density=True, label='Reweighted Class 0 to Class 1, new reweighting', histtype='step', linestyle='--', edgecolor='black', linewidth=1.5)
    plt.hist(score_1, bins=bins, weights=weights_1, density=True, label='Reweighted Class 1 to Class 0',
             histtype='step', linestyle='--', edgecolor='black', linewidth=1.5)
    
    plt.xlabel(f"Classifier output for sample 1")
    plt.ylabel("Density")
    plt.title(f"Distribution of score_label_sample1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"distribution_plots_NN/reweighted_plot_NN_32_3_1_to_0.png")
    #plt.savefig(f'distribution_plots/reweighted_plot_optuna_{optuna_version}_{num}_sample{sample_number}_1_to_0_using_weights_0.pdf')
    plt.close()