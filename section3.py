from models._etsne import ET_SNE,  ETSNE #Those Class are  found in section 1
from dataset import data_load, seed_  # Those Fuctions are found in section 2
import matplotlib.pyplot as plt



if __name__ == "__main__":

    name = "make_blobs"

    X_samples,  labels, dataloader, data_tensor, dataloader_full = data_load(data_name = name,  n_samples=8000, n_features=30, centers=5, cluster_std=1.0)
    # y= labels
    model = ET_SNE(n_components=2, perplexity=30, n_iter=400, random_state=128, learning_rate=0.6, verbose=True)
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
