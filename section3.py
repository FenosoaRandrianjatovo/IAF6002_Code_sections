from models._etsne import ET_SNE,  ETSNE # Those Classes are  found in section 1
from dataset import data_load, seed_  # Those Fuctions are found in section 2
import matplotlib.pyplot as plt



if __name__ == "__main__":
    """
    Main execution block for testing the ET_SNE (t-SNE) implementation.

    This script performs the following steps:
      1. Specifies the dataset name to load (e.g., "make_blobs" or "retina", "cortex", etc).
      2. Loads the dataset using the data_load function, generating synthetic data .
      3. Converts the data from a torch.Tensor to a NumPy array.
      4. Initializes the ET_SNE model with specified hyperparameters such as perplexity, learning rate,
         number of iterations, and random state.
      5. Fits the Et-SNE model to the data, obtaining a low-dimensional embedding and recording the KL 
         divergence over iterations.
      6. Visualizes the KL divergence over iterations, saving the plot as "KL_divergence.png".
      7. Visualizes the 2D t-SNE embedding, coloring points by their label, and saves the plot as "tSNE.png".

    This block serves as an example of how to run the ET_SNE model from scratch on a given  make blobs dataset.
    """
    
    # Set the dataset name; try "make_blobs" or test with "retina", "cortex"
    name = "make_blobs"
    
    # Load the dataset based on the specified name and parameters.
    X, labels, dataloader, data_tensor, dataloader_full = data_load(
        data_name=name,
        n_samples=8000,
        n_features=30,
        centers=5,
        cluster_std=1.0
    )
    
    # Convert the torch tensor to a NumPy array for processing
    X_samples = X.numpy()
    
    # Initialize the ET_SNE model with desired parameters
    model = ET_SNE(
        n_components=2,
        perplexity=30,
        n_iter=400,
        random_state=128,
        learning_rate=0.6,
        verbose=True
    )
    
    # Run Et-SNE to obtain the low-dimensional embedding and record KL divergence history
    y, KL_array = model.fit_trasform(X_samples)
    
    # Visualize and save  the KL divergence over iterations
    plt.figure(figsize=(10, 7)) #Size of the plot
    plt.plot(KL_array)
    plt.title("KL-divergence", fontsize=10) # size of the text inside  the images is 10
    plt.xlabel("Iteration", fontsize=10)
    plt.ylabel("KL-Divergence", fontsize=10) 
    plt.savefig("KL_divergence.png")
    plt.close()  #The image will be saved directly without appearing in the  IDE
    
    # Visualize and save the 2D t-SNE embedding with points colored by their labels
    plt.figure(figsize=(10, 7))
    plt.scatter(y[:, 0], y[:, 1], c=labels.astype(int), cmap='tab10', s=10),
    plt.title("tSNE from Scratch", fontsize=10) # size of the text inside  the images is 10
    plt.xlabel("tSNE1", fontsize=10)
    plt.ylabel("tSNE2", fontsize=10)
    plt.savefig("tSNE.png")
    plt.close() 

