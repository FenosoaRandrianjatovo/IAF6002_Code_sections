# Description: This file contains the functions for preprocessing the datasets and loading the data.
import scanpy as sc
import torch
import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn
from sklearn.preprocessing import StandardScaler, LabelEncoder
import random
from torch.utils.data import DataLoader, TensorDataset


def preprocessing(adata, batch_key=None):
    """
    Preprocess an AnnData object by filtering genes, normalizing counts, applying logarithmic transformation, 
    and selecting highly variable genes.

    This function executes the following steps on the provided dataset:
      1. Filters genes with fewer than 3 counts across all cells using `sc.pp.filter_genes`.
      2. Saves a copy of the original counts from `adata.X` into the "counts" layer to preserve raw data.
      3. Normalizes total counts per cell to a target sum of 10,000 to standardize sequencing depth.
      4. Applies a log(1+x) transformation to the normalized data to stabilize variance.
      5. Freezes the current state of the data by assigning it to `adata.raw` for downstream analysis.
      6. Identifies and subsets the top 1,200 highly variable genes using the "seurat_v3" method on the preserved counts layer.
         An optional `batch_key` can be provided to account for batch effects during variable gene selection.

    Parameters
    ----------
    adata : AnnData
        An annotated data matrix containing raw gene expression counts in `adata.X`.
    batch_key : str, optional
        The key in `adata.obs` that identifies batch labels for cells. This is used to adjust for batch effects 
        during the selection of highly variable genes.

    Returns
    -------
    AnnData
        The processed AnnData object, which includes normalized, log-transformed data in `adata.X`, a backup of raw 
        counts in `adata.layers["counts"]`, the original state saved in `adata.raw`, and a subset of highly variable genes.
    """
    sc.pp.filter_genes(adata, min_counts=3)
    adata.layers["counts"] = adata.X.copy()  # Preserve counts
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata  # Freeze the state in `.raw`
    
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=1200,
        subset=True,
        layer="counts",
        flavor="seurat_v3",
        batch_key=batch_key
    )
    return adata



def data_load(data_name, batch_size=16, cov=False, n_samples=40000, n_features=100, centers=5, cluster_std=1.0):
    """
    Load and prepare a dataset along with corresponding DataLoader objects for model training.

    This function supports loading both synthetic and real datasets based on the provided `data_name`. 
    Supported dataset identifiers include:
      - 'Two_moons': Generates a two-moons synthetic dataset.
      - 'iris': Loads the Iris dataset.
      - 'make_blobs': Generates a synthetic dataset with blob clusters.
      - 'MNIST': Loads the MNIST handwritten digits dataset.
      - 'cortex', 'pbmc', 'retina', 'heart_cell_atlas': Loads corresponding single-cell datasets using scvi and applies a preprocessing step.
      - Any other value of data_name is treated as a file path to a CSV file, where the last column is assumed to contain labels.

    For synthetic datasets ('Two_moons' and 'make_blobs'), parameters such as `n_samples`, `n_features`, 
    `centers`, and `cluster_std` control the characteristics of the generated data.

    Parameters
    ----------
    data_name : str
        Identifier for the dataset to load. Can be one of the predefined dataset names or a file path to a CSV file.
    batch_size : int, optional
        Batch size to be used in the DataLoader. Default is 16.
    cov : bool, optional
        Unused parameter reserved for potential covariate handling. Default is False.
    n_samples : int, optional
        Number of samples to generate for synthetic datasets. Default is 40000.
    n_features : int, optional
        Number of features for synthetic datasets generated with 'make_blobs'. Default is 100.
    centers : int, optional
        Number of centers (clusters) for the 'make_blobs' dataset. Default is 5.
    cluster_std : float, optional
        Standard deviation of clusters for the 'make_blobs' dataset. Default is 1.0.

    Returns
    -------
    X : torch.Tensor
        Tensor of features created from the dataset.
    labels : array-like
        Array or tensor of labels corresponding to the data.
    dataloader : torch.utils.data.DataLoader
        DataLoader for iterating over the dataset in batches (shuffled).
    data_tensor : torch.Tensor
        Torch tensor containing the dataset, explicitly cast to float32.
    dataloader_full : torch.utils.data.DataLoader
        DataLoader containing the entire dataset (batch size equals dataset size), not shuffled.

    Notes
    -----
    - For single-cell datasets (e.g., 'cortex', 'pbmc', 'retina', 'heart_cell_atlas'), a preprocessing 
      routine is applied prior to data extraction.
    - When `data_name` is not one of the predefined dataset names, the function attempts to load the data 
      from a CSV file. In this case, the last column is assumed to be the label, and the remaining columns 
      form the feature matrix.
    - Some branches (e.g., handling of covariates or batch effects) are reserved  for future extension.
    """
    
    covariate = torch.empty(0)
    batch = None

    if data_name == 'Two_moons':
        from sklearn.datasets import make_moons
        df, labels = make_moons(n_samples=n_samples, noise = 0.05, shuffle = False, random_state = 42)
    elif data_name =="iris":
        from sklearn.datasets import load_iris
        import pandas as pd

        # Load the Iris dataset
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names).values
        labels = data.target

    elif data_name =="make_blobs":

        from sklearn.datasets import make_blobs
        df, labels = make_blobs(
            n_samples=n_samples, # Total number of samples
            n_features=n_features, #Number of features
            centers= centers, #Number of classes
            cluster_std=cluster_std,  # Adjust for more or less separation
            random_state=42
        )


    elif data_name == 'MNIST':
        from torchvision import datasets, transforms
        MNIST = datasets.MNIST('./mnist', train=True, download=True, transform=transforms.ToTensor())
        df = MNIST.data.view(60000,28*28).to(torch.float) / 255
        df = df.apply_(lambda x: 1 if x>= 0.5 else 0)
        labels = MNIST.targets

    elif data_name == 'cortex':
        import scvi
        adata = scvi.data.cortex()
        preprocessing(adata)
        df = adata.layers['counts']
        labels = adata.obs['cell_type']

    elif data_name == 'pbmc':
        import scvi
        adata = scvi.data.pbmc_dataset()
        preprocessing(adata)
        df = adata.layers['counts'].toarray()
        labels = adata.obs['str_labels']

    elif data_name == 'retina':
        import scvi
        from torch.nn import functional as F
        adata = scvi.data.retina()
        preprocessing(adata)
        df = adata.layers['counts']
        labels = adata.obs['labels']
        batch = LabelEncoder().fit_transform(adata.obs['batch'])


    elif data_name == 'heart_cell_atlas':
        import scvi
        from torch.nn import functional as F
        adata = scvi.data.heart_cell_atlas_subsampled()
        preprocessing(adata)
        df = adata.layers['counts'].toarray()
        labels = adata.obs['cell_type']
 
    
    else:
      import pandas as pd
      df = pd.read_csv(data_name).values
      labels =df[:,-1] #

      df = df[:,:-1]

    X = torch.Tensor(df)
    # labels = torch.Tensor(labels).to(torch.int64)

    # Define the dataset and data loader    
    data_tensor = torch.tensor(X, dtype=torch.float32)
    batch_size = batch_size
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader_full = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    return X, labels, dataloader, data_tensor, dataloader_full

def seed_(num):
    """
    Set the random seed for reproducibility across PyTorch, NumPy, and Python's random module.

    This function configures multiple libraries to use a fixed seed, ensuring that experiments
    can be reproduced exactly. It performs the following actions:
      - Sets the seed for PyTorch's CPU operations.
      - Sets the seed for PyTorch's CUDA operations (for a single GPU and all GPUs).
      - Seeds NumPy's random number generator.
      - Configures PyTorch's cuDNN backend to disable benchmarking and enable deterministic behavior.
      - Seeds Python's built-in random module.

    Parameters
    ----------
    num : int
        The seed value to be used for all random number generators.
    """
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    torch.cuda.manual_seed_all(num)
    np.random.seed(num)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(num)
