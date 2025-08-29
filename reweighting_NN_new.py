import uproot
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np

#Open ROOT file
file = uproot.open("NN_outputs/nn_output_32_3.root")
tree = file["Events"]

#Load branches as awkward arrays
nn_output = tree["nn_output"].array(library="ak")
label = tree["label_sample1"].array(library="ak")
score = tree["score_label_sample1"].array(library="ak")

#Define masks
mask_label1 = (label == 1)
mask_label0 = (label == 0)

#Reweighting: label == 0 events weighted like label == 1
weights_rew = ak.to_numpy(mask_label0 * (nn_output / (1 - nn_output)))

#Convert to NumPy
score_all = ak.to_numpy(score)
score_label1 = ak.to_numpy(score[mask_label1])
score_label0 = ak.to_numpy(score[mask_label0])

#Plot histograms
plt.figure(figsize=(8, 6))

bins = np.linspace(0, 1, 50)

plt.hist(score_label1, bins=bins, density=True, histtype="step",
         linewidth=2, color="red", label="label=1")
plt.hist(score_label0, bins=bins, density=True, histtype="step",
         linewidth=2, color="blue", label="label=0")
plt.hist(score_all, bins=bins, weights=weights_rew, density=True,
         histtype="step", linewidth=2, color="green",
         label="label=0 reweighted to label=1")

#Legend, labels
plt.xlabel("score_label_sample1")
plt.ylabel("Normalized events")
plt.legend()
plt.tight_layout()

#Save
plt.savefig("distribution_plots_NN/reweighting_32.png", dpi=300)