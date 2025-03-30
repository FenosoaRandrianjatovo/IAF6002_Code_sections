from models._etsne import T_SNE,  ETSNE
from dataset import data_load, seed_
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
import pandas as pd


if __name__ == "__main__":

    name = "make_blobs"


    df, labels = make_blobs(
        n_samples=800, # Total number of samples
        n_features=30, #Number of features
        centers= 5   , #Number of classes
        cluster_std=1,  # Adjust for more or less separation
        random_state=42
    )
    X_samples = pd.DataFrame(df).values
    # y= labels
    model = T_SNE(n_components=2, perplexity=30, n_iter=400, random_state=128, learning_rate=0.6, verbose=True)
    y, KL_array = model.fit_trasform(X_samples)
    plt.figure(figsize=(10,7))
    plt.plot(KL_array)
    plt.title("KL-divergence", fontsize = 10)
    plt.xlabel("Iteration", fontsize = 10); plt.ylabel("KL-Divergence", fontsize = 10)
    plt.savefig("KL_divergence.png")
    plt.close()

    # print(y)
    plt.figure(figsize=(10,7))
    plt.scatter(y[:,0], y[:,1], c=labels.astype(int), cmap = 'tab10', s = 10)
    plt.title("tSNE from Scratch", fontsize = 10)
    plt.xlabel("tSNE1", fontsize = 10); plt.ylabel("tSNE2", fontsize = 10)
    plt.savefig("tSNE.png")
    plt.close()
    # plt.show()
