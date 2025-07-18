import uproot
import numpy as np
import matplotlib.pyplot as plt

# Path to your ROOT file
root_file_path = "optuna_outputs/optuna_model_5_predictions.root"

# Open the ROOT file and access the Events tree
with uproot.open(root_file_path) as file:
    tree = file["Events"]
    
    # Load the variable sample2
    score = tree["score_label_sample1"].array(library="np")
    label = tree["label_sample1"].array(library="np")

# Separate the scores based on label
score_0 = score[label == 0]
score_1 = score[label == 1]

weights_class_0 = np.ones_like(score_0) / len(score_0)

# Plotting
plt.figure(figsize=(8, 6))
plt.hist(score_0, bins=50, alpha=0.6, label='label_sample1 = 0', color='blue', density=True)
plt.hist(score_1, bins=50, alpha=0.6, label='label_sample1 = 1', color='red', density=True)

plt.xlabel("score_label_sample1")
plt.ylabel("Density")
plt.title("Distribution of score_label_sample1")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("distribution_plots/plot_optuna_5_sample1_new2.png")