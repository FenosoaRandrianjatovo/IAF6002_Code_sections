import numpy as np
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets import make_blobs


class T_SNE:
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000, random_state=42, verbose=False):
        """
        Parameters:
        n_components : int
            Number of dimensions to reduce to.
        perplexity : float
            Perplexity parameter for t-SNE.
        learning_rate : float
            Learning rate for optimization.
        n_iter : int
            Number of iterations for optimization.
        random_state : int or None
            Random seed for reproducibility.
        """
        # Initialize parameters
        self.n_components = n_components
        self.PERPLEXITY = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        self.embedding_ = None
        self.kl_divergence_ = None
        self.initialized = False
        self.data = None
        self.distances = None
        self.pairwise_distances = None
        self.pairwise_distances_squared = None
        self.verbose = verbose
        self.n = None
        self.n_samples = None
        self.n_features = None
    
    def prob_high_dim(self, dist, sigma, dist_row):
        """
        For each row of Euclidean distance matrix (dist_row) compute
        probability in high dimensions (1D array)
        """
        exp_distance = np.exp(-dist[dist_row] / (2*sigma**2))
        exp_distance[dist_row] = 0
        prob_not_symmetr = exp_distance / np.sum(exp_distance)
        #prob_symmetr = (prob_not_symmetr + prob_not_symmetr.T) / (2*n_samples)
        return prob_not_symmetr
    
  
    def _perplexity(self,prob):
        """
        Compute perplexity (scalar) for each 1D array of high-dimensional probability
        """
        return np.power(2, -np.sum([p*np.log2(p) for p in prob if p!=0]))
    
    
    def sigma_binary_search(self,perp_of_sigma, fixed_perplexity):
        """
        Solve equation perp_of_sigma(sigma) = fixed_perplexity 
        with respect to sigma by the binary search algorithm
        """
        sigma_lower_limit = 0
        sigma_upper_limit = 1000
        for i in range(20):
            approx_sigma = (sigma_lower_limit + sigma_upper_limit) / 2
            if perp_of_sigma(approx_sigma) < fixed_perplexity:
                sigma_lower_limit = approx_sigma
            else:
                sigma_upper_limit = approx_sigma
            if np.abs(fixed_perplexity - perp_of_sigma(approx_sigma)) <= 1e-5:
                break
        return approx_sigma
    

    def binary_search_sigma(self, dist):
        """
        Compute sigma for each row of distance matrix
        """
        n =self.n_samples
        prob = np.zeros((n,n))
        sigma_array = []
        for dist_row in range(n):
            func = lambda sigma: self._perplexity(self.prob_high_dim(dist, sigma, dist_row))
            binary_search_result = self.sigma_binary_search(func, self.PERPLEXITY)
            prob[dist_row] = self.prob_high_dim(dist,binary_search_result, dist_row)
            sigma_array.append(binary_search_result)
            if self.verbose and (dist_row + 1) % 100 == 0:
                print("Sigma binary search finished {0} of {1} cells".format(dist_row + 1, n))

        sigma_array = np.array(sigma_array)
        # Compute the final probability matrix
        P = prob + np.transpose(prob)/ (n*2)
        return P, sigma_array
    
   
    def prob_low_dim(self,Y):
        """
        Compute matrix of probabilities q_ij in low-dimensional space
        """
        inv_distances = np.power(1 + np.square(euclidean_distances(Y, Y)), -1)
        np.fill_diagonal(inv_distances, 0.)
        return inv_distances / np.sum(inv_distances, axis = 1, keepdims = True)
    
    def kl_divergence(self, P, Q):
        """
        Compute KL divergence between two probability matrices
        """
        Q = self.prob_low_dim(Q)
        return np.sum(P * np.log(P + 0.01) - P * np.log(Q + 0.01))
    
    def KL_gradient(self, P, y):
        """
        Compute the gradient of the KL divergence between the high-dimensional and low-dimensional 
        probability distributions.
    
        The gradient is computed according to the formula:
    
            ∂C/∂y_i = 4 ∑_j (P_ij - Q_ij) (y_i - y_j) (1 + ||y_i - y_j||^2)^{-1}
    
        where:
          - P is the high-dimensional similarity matrix (n x n),
          - Q is the low-dimensional similarity matrix computed from 'y', shape is still (n x n)
          - y_diff = y_i - y_j is the pairwise difference between the low-dimensional points, (n x d)
          - inv_dist = (1 + ||y_i - y_j||^2)^{-1} is the inverse distance weight. 
    
        Steps:
        1. Compute Q using the method 'self.prob_low_dim(y)' and ensure numerical stability by 
           setting a minimum value of 1e-5.
        2. Calculate the pairwise difference tensor 'y_diff' (shape: n x n x d) by expanding the 
           dimensions of 'y' to enable broadcasting.
        3. Compute the inverse distance matrix 'inv_dist' by applying the transformation 
           (1 + squared Euclidean distance)^{-1} on 'y'.
        4. Multiply the expanded (P - Q), y_diff, and expanded inv_dist element-wise, sum over 
           the second dimension (axis=1), and multiply by 4 to obtain the gradient.
    
        Parameters
        ----------
        P : numpy.ndarray, shape (n, n)
            The high-dimensional similarity matrix representing pairwise affinities between data points.
        y : numpy.ndarray, shape (n, d)
            The low-dimensional embeddings of the data points, where 'n' is the number of samples and 
            'd' is the dimensionality of the embedding space.
    
        Returns
        -------
        numpy.ndarray, shape (n, d)
            The gradient of the KL divergence with respect to 'y'. Each row corresponds to the 
            gradient vector for a point in the low-dimensional space.
            
        """
        Q = self.prob_low_dim(y)
        Q = np.maximum(Q, 1e-5)
        y_diff = np.expand_dims(y, 1) - np.expand_dims(y, 0) # (n x n x d)
        inv_dist = np.power(1 + np.square(euclidean_distances(y, y)), -1) # (n x n)
        return 4 * np.sum(np.expand_dims(P - Q, 2) * y_diff * np.expand_dims(inv_dist, 2), axis=1)

    
    def fit_trasform(self, X, y=None):
        """
        Fit the t-SNE model to the data.
        Parameters:
        X : array-like, shape (n_samples, n_features)
            Input data.
        """ 
        # Check if the input data is valid
        if X is None or len(X) == 0:
            raise ValueError("Input data cannot be None or empty.")
        if len(X.shape) != 2:
            raise ValueError("Input data must be a 2D array.")
        if X.shape[0] < 2:
            raise ValueError("Input data must have at least two samples.")
   
        # Initialize parameters
        dist = np.square(euclidean_distances(X, X))
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.n = self.n_samples

        P, _ = self.binary_search_sigma(dist)
        np.random.seed(self.random_state)
        # print((self.n_samples, self.n_components))

        # y =np.zeros((self.n_samples, self.n_components))
        rng = np.random.RandomState(self.random_state)
        Y = rng.normal(loc=0, scale=1, size=(self.n_samples, self.n_components))
        KL_array = []
        print("Running Gradient Descent: \n")

        for i in range(self.n_iter):
            Y = Y - self.learning_rate * self.KL_gradient(P, Y)

            KL_array.append(self.kl_divergence(P, Y))
            if self.verbose and i % 50 == 0:
                
                print("KL divergence = " + str(self.kl_divergence(P, Y)),f"for iteration {i + 1} of {self.n_iter}")

        return Y, np.array(KL_array)
