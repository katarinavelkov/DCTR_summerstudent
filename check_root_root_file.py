import uproot
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import awkward as ak

filename = r'/eos/cms/store/group/cmst3/user/sesanche/SummerStudent/train_0p01/e0d8f59d-12d6-418f-809f-9b8d5d4265a3.root.root'

with uproot.open(filename) as f:
    top_level_keys = f.keys()
    print("Top-level keys in the ROOT file:", top_level_keys)

    events = f['GenObjectTree']
    print("Branches in GenObjectTree:", events.keys())

    isSignal = events["isSignal"].array()
    print('uniques:', np.unique(isSignal))

    array = events.arrays(library='ak')

    for i in range(2):
        print(f'Event {i}:')
        for key in array.fields:
            print(f'  {key}: {array[key][i]}')
    
    # check distribution of 0 and 1 labels
    

    print("Class distribution:", distribution)