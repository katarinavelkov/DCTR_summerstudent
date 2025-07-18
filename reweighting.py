import uproot
import numpy as np
import awkward as ak
import os

number = 5
predictions_filename = f'optuna_outputs/optuna_model_{number}_predictions.root'

def open_root_file(filename):
    with uproot.open(filename) as file:
        events = file["Events"]
        array = events.arrays(library='ak')
    return array

def reweight():
    array = open_root_file(predictions_filename)
    predictions = ak.to_numpy(array['label_sample1'])
    scores = ak.to_numpy(array['score_label_sample1'])
    target = ak.to_numpy(array['_label_']).astype(int)
    predictions_2 = ak.to_numpy(array['label_sample2'])
    