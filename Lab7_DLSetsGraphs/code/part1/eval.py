"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error
import torch

from utils import create_test_dataset
from models import DeepSets, LSTM

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
batch_size = 64
embedding_dim = 128
hidden_dim = 64

# Generates test data
X_test, y_test = create_test_dataset()
cards = [X_test[i].shape[1] for i in range(len(X_test))]
n_samples_per_card = X_test[0].shape[0]
n_digits = 11

# Retrieves DeepSets model
deepsets = DeepSets(n_digits, embedding_dim, hidden_dim).to(device)
print("Loading DeepSets checkpoint!")
checkpoint = torch.load("model_deepsets.pth.tar")
deepsets.load_state_dict(checkpoint["state_dict"])
deepsets.eval()

# Retrieves LSTM model
lstm = LSTM(n_digits, embedding_dim, hidden_dim).to(device)
print("Loading LSTM checkpoint!")
checkpoint = torch.load("model_lstm.pth.tar")
lstm.load_state_dict(checkpoint["state_dict"])
lstm.eval()

# Dict to store the results
results = {"deepsets": {"acc": [], "mae": []}, "lstm": {"acc": [], "mae": []}}

for i in range(len(cards)):
    y_pred_deepsets = list()
    y_pred_lstm = list()
    for j in range(0, n_samples_per_card, batch_size):

        ############## Task 6
        x_batch = torch.tensor(X_test[i][j : j + batch_size]).to(device)
        y_batch = torch.tensor(y_test[i][j : j + batch_size]).to(device)

        output_deepsets = deepsets(x_batch)
        output_lstm = lstm(x_batch)

        y_pred_deepsets.append(torch.round(output_deepsets))
        y_pred_lstm.append(torch.round(output_lstm))

    y_pred_deepsets = torch.cat(y_pred_deepsets)
    y_pred_deepsets = y_pred_deepsets.detach().cpu().numpy()

    acc_deepsets = accuracy_score(y_test[i], y_pred_deepsets)
    mae_deepsets = mean_absolute_error(y_test[i], y_pred_deepsets)
    results["deepsets"]["acc"].append(acc_deepsets)
    results["deepsets"]["mae"].append(mae_deepsets)

    y_pred_lstm = torch.cat(y_pred_lstm)
    y_pred_lstm = y_pred_lstm.detach().cpu().numpy()

    acc_lstm = accuracy_score(y_test[i], y_pred_lstm)
    mae_lstm = mean_absolute_error(y_test[i], y_pred_lstm)
    print(
        "Card: {:03d}".format(cards[i]),
        "DeepSets - Accuracy: {:.4f}".format(acc_deepsets),
        "DeepSets - MAE: {:.4f}".format(mae_deepsets),
        "LSTM - Accuracy: {:.4f}".format(acc_lstm),
        "LSTM - MAE: {:.4f}".format(mae_lstm),
    )
    results["lstm"]["acc"].append(acc_lstm)
    results["lstm"]["mae"].append(mae_lstm)


############## Task 7

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(cards, results["deepsets"]["acc"], label="DeepSets")
plt.plot(cards, results["lstm"]["acc"], label="LSTM")
plt.xlabel("Cardinality")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy comparison between DeepSets and LSTM")
plt.grid()
plt.show()
