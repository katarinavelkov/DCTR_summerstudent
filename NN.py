import uproot
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os

# -----------------------
# Dataset definition
# -----------------------
class ROOTDataset(Dataset):
    def __init__(self, rootfile, treename="Events"):
        with uproot.open(rootfile) as f:
            tree = f[treename]
            scores = tree["score_label_sample1"].array(library="np")
            labels = tree["label_sample1"].array(library="np")

        self.scores = scores
        self.X = torch.tensor(scores, dtype=torch.float32).unsqueeze(1)  # (N,1)
        self.y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # (N,1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -----------------------
# Simple NN model
# -----------------------
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, 1)  # logits (no sigmoid here!)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------
# Training loop with class weights
# -----------------------
def train_nn(dataset, epochs=40, batch_size=256, lr=1e-4):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = SimpleNN()

    # compute class imbalance
    num_pos = dataset.y.sum().item()
    num_neg = len(dataset) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)

    print(f"Training with pos_weight = {pos_weight.item():.2f}")

    # BCEWithLogitsLoss = stable + supports pos_weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataset):.4f}")

    return model

# -----------------------
# Write outputs to ROOT
# -----------------------
def write_outputs_to_root(dataset, model, outfilename="NN_outputs/nn_output_32_3.root"):
    model.eval()
    with torch.no_grad():
        logits = model(dataset.X)
        probs = torch.sigmoid(logits).numpy().flatten()  # <-- convert logits to probabilities
        labels = dataset.y.numpy().flatten()
        scores = dataset.scores.flatten()

    with uproot.recreate(outfilename) as f:
        f["Events"] = {
            "nn_output": probs,                 # probability in [0,1]
            "label_sample1": labels,
            "score_label_sample1": scores
        }
    print(f"Output written to {outfilename}")

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    #infile = "./predict_output/output_train.root"
    infile = 'optuna_outputs_32/optuna_model_0_predictions.root'
    dataset = ROOTDataset(infile)
    model = train_nn(dataset, epochs=40, batch_size=256, lr=1e-4)

    parameters_dict = {'epochs': 40, 'batch_size': 256, 'learning_rate': 1e-4}
    pd.DataFrame(parameters_dict, index=[0]).to_csv("NN_outputs/parameters_NN_32_3.txt", sep="\t", index=False)

    write_outputs_to_root(dataset, model, "NN_outputs/nn_output_32_3.root")