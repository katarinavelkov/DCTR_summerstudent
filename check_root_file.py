import uproot
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import awkward as ak

filename = 'predict_output/output.root'
#filename = r'/eos/cms/store/group/cmst3/user/sesanche/SummerStudent/train_0p01/00ae10f7-1953-4002-8bb7-42c294122af1.root.root'
with uproot.open(filename) as f:
    events = f['Events']
    print(events.keys())

    array = events.arrays(library='ak')

    for i in range(2):
        print(f'Event {i}:')
        for key in array.fields:
            print(f'  {key}: {array[key][i]}')
    
    # check distribution of 0 and 1 labels
    target = ak.to_numpy(array['_label_'])
    unique, counts = np.unique(target, return_counts=True)
    distribution = dict(zip(unique, counts))

    print("Class distribution:", distribution)