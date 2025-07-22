import uproot
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


optuna_version = 3  # Change this to the version of optuna run
number_of_models = 10  # Total number of models trained in that optuna run
sample_number = 1  # Change this to the sample number you want to plot

for num in range(number_of_models):
    print('-----------------------------------------------------------')
    print(f"Processing model {num} from optuna version {optuna_version}")
    root_file_path = f"optuna_outputs_{optuna_version}/optuna_model_{num}_predictions.root"

    # Open the ROOT file and access the Events tree
    with uproot.open(root_file_path) as file:
        tree = file["Events"]
        
        # Load the desired variable sample
        score = tree[f"score_label_sample{sample_number}"].array(library="np")
        label = tree[f"label_sample{sample_number}"].array(library="np")

        array = tree.arrays(library='ak')
        target = ak.to_numpy(array['_label_']).astype(int)
        # get probabilities 
        pred_probs = ak.to_numpy(array[f'score_label_sample{sample_number}']) # they go from 0.5 to 1


    # Separate the scores based on label
    score_0 = score[label == 0]
    score_1 = score[label == 1]

    # reweighting of one class to the chosen target distribution!
    epsilon = 1e-9
    weights_0 = score_0 / (1 - score_0 + epsilon)

    #reweighted_score_0 = score_0 * weights_0 

    # Plotting distribution
    plt.figure(figsize=(8, 6))
    plt.hist(score_0, bins=50, alpha=0.7, label='Original class 0', color='blue', density=True)
    plt.hist(score_1, bins=50, alpha=0.5, label='Target class 1', color='red', density=True)
    #plt.hist(reweighted_score_0, bins=50, alpha=0.6, label='Reweighted label_sample1 = 0', color='green', density=True)
    plt.hist(score_0, bins=50, label='Reweighted class 0', density=True, 
         weights=weights_0, 
         histtype='step',      
         edgecolor='black',    
         linewidth=1.5)  
    plt.xlabel(f"Classifier output sample{sample_number}")
    plt.ylabel("Density")
    plt.title(f"Distribution of score_label_sample{sample_number}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"distribution_plots/reweighted_plot_optuna_{optuna_version}_{num}_sample{sample_number}.png")
    plt.close()
    