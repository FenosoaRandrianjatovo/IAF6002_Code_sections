from models._etsne import ET_SNE,  ETSNE # Those Classes are  found in section 1
from dataset import data_load, seed_  # Those Fuctions are found in section 2
import matplotlib.pyplot as plt



if __name__ == "__main__":

    name = "make_blobs"
    # name = "retina"
    #Load the Data set based on the given name
    X,  labels, dataloader, data_tensor, dataloader_full = data_load(data_name = name,  n_samples=8000, n_features=30, centers=5, cluster_std=1.0)
    # y= labels
    # Convert the array from torch to numpy
    X_samples = X.numpy()
    # Initiate the Model using the ET_SNE Class
    model = ET_SNE(n_components=2, perplexity=30, n_iter=400, random_state=128, learning_rate=0.6, verbose=True)
    # To keep the prediction and save the KL for visualiation
    y, KL_array = model.fit_trasform(X_samples)

    # Visualisation of the KL Divergence in figure size format of width =10 and height =7, 
    # and font size is 10 for all the letter and the point size is s=10 , for all the plot
    plt.figure(figsize=(10,7))
    plt.plot(KL_array)
    plt.title("KL-divergence", fontsize = 10)
    plt.xlabel("Iteration", fontsize = 10); plt.ylabel("KL-Divergence", fontsize = 10)
        # To save the images in the current folder
    plt.savefig("KL_divergence.png")
    plt.close()

    # print(y)
    plt.figure(figsize=(10,7))
    plt.scatter(y[:,0], y[:,1], c=labels.astype(int), cmap = 'tab10', s = 10)
    plt.title("tSNE from Scratch", fontsize = 10)
    plt.xlabel("tSNE1", fontsize = 10); plt.ylabel("tSNE2", fontsize = 10)
    # To save the images in the current folder
    plt.savefig("tSNE.png")
    plt.close()
    # plt.show()
