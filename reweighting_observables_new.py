import uproot
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import awkward as ak

sample_number = 1  # Change this to the sample number you want to plot, 1 or 2

root_file_path = f"optuna_outputs_32/optuna_model_0_predictions.root"
with uproot.open(root_file_path) as f:
    tree = f["Events"]  
    
    pt       = tree["GenObject_pt"].arrays(library="ak")["GenObject_pt"]
    isJet    = tree["GenObject_isJet"].arrays(library="ak")["GenObject_isJet"]
    isLepton = tree["GenObject_isLepton"].arrays(library="ak")["GenObject_isLepton"]
    nGenObject_pt = tree["nGenObject_pt"].array(library="np")

    # target values
    target = tree[f"_label_"].array(library="ak") 

print(target[:5])

#Open predictions from NN outputs
file = uproot.open("NN_outputs/nn_output_32_3.root")
tree = file["Events"]

#Load branches as awkward arrays
nn_output = tree["nn_output"].array(library="ak")
label = tree["label_sample1"].array(library="ak")
score = tree["score_label_sample1"].array(library="ak")

#Define masks
mask_label1 = (target == 1)
mask_label0 = (target == 0)

#Reweighting: label == 0 events weighted like label == 1
weights_rew = ak.to_numpy(mask_label1 * (nn_output / (1 - nn_output)))
weights_per_particle = ak.broadcast_arrays(pt, weights_rew)[1]

pt_0 = pt[target == 0]
pt_1 = pt[target == 1]

isJet_0 = isJet[target == 0]
isJet_1 = isJet[target == 1]

isLepton_0 = isLepton[target == 0]
isLepton_1 = isLepton[target == 1]

min_pt = 0
max_pt = 750
num_bins = 60
bins = np.linspace(min_pt, max_pt, num_bins)

plt.figure()
#plt.hist(ak.flatten(pt), bins=100, histtype="step", label="All particles")
plt.hist(ak.flatten(pt_1), bins=bins, label='Class 1', color='red', alpha=0.6)
plt.hist(ak.flatten(pt_0), bins=bins, label='Class 0', color='blue', alpha=0.6)
plt.hist(ak.flatten(pt), bins=bins, alpha=1, weights=ak.flatten(weights_per_particle),
        label='Reweighted Class 0 to Class 1', color='black', linewidth=1.0, linestyle='--', 
        histtype='step')
plt.xlabel("pT [GeV]")
plt.ylabel("Entries")
plt.xlim(0, 750)
plt.legend()
plt.tight_layout()
plt.savefig('observables_plots/pt_classes_reweighted.png')

plt.figure()
#plt.hist(ak.flatten(pt_jets), bins=100, histtype="step", label="Jets")
plt.hist(ak.flatten(pt_1[isJet_1]), bins=bins, label='Class 1', color='red', alpha=0.6)
plt.hist(ak.flatten(pt_0[isJet_0]), bins=bins, label='Class 0', color='blue', alpha=0.6)
plt.hist(ak.flatten(pt[isJet]), bins=bins, alpha=1, weights=ak.flatten(weights_per_particle[isJet]),
        label='Reweighted Class 0 to Class 1', color='black', linewidth=1.0, linestyle='--', histtype='step')
plt.xlabel("pT [GeV]")
plt.ylabel("Entries")
plt.xlim(0, 750)
plt.tight_layout()
plt.legend()
plt.savefig('observables_plots/pt_jets_classes_reweighted.png')

plt.figure()
#plt.hist(ak.flatten(pt_leptons), bins=100, histtype="step", label="Leptons")
plt.hist(ak.flatten(pt_1[isLepton_1]), bins=bins, label='Class 1', color='red', alpha=0.6)
plt.hist(ak.flatten(pt_0[isLepton_0]), bins=bins, label='Class 0', color='blue', alpha=0.6)
plt.hist(ak.flatten(pt[isLepton]), bins=bins, alpha=1, weights=ak.flatten(weights_per_particle[isLepton]),
        label='Reweighted Class 0 to Class 1', color='black', linewidth=1.0, linestyle='--', histtype='step')
plt.xlabel("pT [GeV]")
plt.ylabel("Entries")
plt.xlim(0, 750)
plt.tight_layout()
plt.legend()
plt.savefig('observables_plots/pt_leptons_classes_reweighted.png')