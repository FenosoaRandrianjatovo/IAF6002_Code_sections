import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class ET_SNE:
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=0.05, n_iter=1000, random_state=42, verbose=False):
        """
        Initialize the Et-SNE model with specified parameters.
    
        Parameters
        ----------
        n_components : int, default=2
            The number of dimensions for the low-dimensional embedding.
        perplexity : float, default=30.0
            The perplexity parameter, which balances attention between local and global aspects of the data.
        learning_rate : float, default=0.05
            The learning rate used during gradient descent optimization.
        n_iter : int, default=1000
            The total number of iterations for optimization.
        random_state : int or None, default=42
            The seed for random number generation to ensure reproducibility.
        verbose : bool, default=False
            If True, prints progress messages during computation.
    
        Attributes
        ----------
        embedding_ : None or np.ndarray
            The computed low-dimensional embedding after fitting the model. Initialized as None.
        kl_divergence_ : None or np.ndarray
            The history of Kullback-Leibler divergence values over iterations. Initialized as None.
        initialized : bool
            Flag indicating whether the model has been initialized. Defaults to False.
        data : None
            Placeholder for input data. Initialized as None.
        distances : None
            Placeholder for computed pairwise distances. Initialized as None.
        pairwise_distances : None
            Placeholder for the pairwise distance matrix. Initialized as None.
        pairwise_distances_squared : None
            Placeholder for the squared pairwise distance matrix. Initialized as None.
        n_samples : None or int
            The number of samples in the dataset. Initialized as None.
        n_features : None or int
            The number of features in the dataset. Initialized as None.
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
        Compute the high-dimensional conditional probability for a specific row of a distance matrix.
    
        This function calculates the probability distribution for one row of a Euclidean distance
        matrix based on a Gaussian (RBF) kernel. It computes the unnormalized probabilities for each
        element in the specified row (except for the diagonal element, which is set to zero), and then
        normalizes these probabilities so that they sum to 1.
    
        Parameters
        ----------
        dist : np.ndarray
            A 2D array representing the pairwise Euclidean distances between data points.
        sigma : float
            The standard deviation of the Gaussian kernel, used to scale the distances.
        dist_row : int
            The index of the row in the distance matrix for which the probability distribution is computed.
    
        Returns
        -------
        np.ndarray
            A 1D array containing the normalized probabilities for the specified row, where the probability
            corresponding to the self-distance is zero.
        """
        exp_distance = np.exp(-dist[dist_row] / (2 * sigma**2))
        exp_distance[dist_row] = 0
        prob_not_symmetr = exp_distance / np.sum(exp_distance)
        
        return prob_not_symmetr

    
      
    def _perplexity(self, prob):
        """
        Compute the perplexity of a high-dimensional probability distribution.
    
        The perplexity is a measure of the effective number of neighbors and is defined as 2 raised to the
        entropy of the probability distribution. This function calculates the entropy using base-2 logarithms,
        while ignoring any zero values to avoid undefined logarithms, and then returns the perplexity as:
        
            perplexity = 2^(entropy)
        
        Parameters
        ----------
        prob : array-like of float
            A 1D array representing the probability distribution for which to compute the perplexity.
    
        Returns
        -------
        float
            The perplexity computed from the probability distribution.
        """
        return np.power(2, -np.sum([p * np.log2(p) for p in prob if p != 0]))

    
    
    def sigma_binary_search(self, perp_of_sigma, fixed_perplexity):
        """
        Find the sigma value that yields the target perplexity using a binary search algorithm.
    
        This function solves the equation perp_of_sigma(sigma) = fixed_perplexity by iteratively 
        refining the search interval for sigma. The binary search is performed for up to 20 iterations, 
        and the process terminates early if the absolute difference between the computed and target 
        perplexity is within a tolerance of 1e-5.
    
        Parameters
        ----------
        perp_of_sigma : callable
            A function that takes a sigma value (float) as input and returns the corresponding perplexity.
        fixed_perplexity : float
            The target perplexity value that the sigma should approximate.
    
        Returns
        -------
        float
            The sigma value that approximately satisfies the equation perp_of_sigma(sigma) = fixed_perplexity.
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
        Compute per-sample sigma values and the corresponding high-dimensional probability matrix from a distance matrix.
    
        For each row in the provided distance matrix, this function calculates a sigma value such that the 
        computed perplexity (using a Gaussian kernel) approximates a target perplexity (self.PERPLEXITY). 
        It performs a binary search to solve for sigma by iteratively adjusting the search interval until the 
        perplexity computed from the row's probability distribution is within a small tolerance of the target value.
        
        The computed sigma is then used to generate a normalized probability distribution for that row. 
        After processing all rows, the final high-dimensional probability matrix is symmetrized by averaging 
        with its transpose.
    
        Parameters
        ----------
        dist : np.ndarray
            A 2D array of pairwise Euclidean distances between samples. Each row corresponds to distances 
            from a specific sample to all other samples.
    
        Returns
        -------
        P : np.ndarray
            The symmetrized high-dimensional probability matrix, where each entry represents the conditional 
            probability computed using a Gaussian kernel with the corresponding sigma.
        sigma_array : np.ndarray
            A 1D array of sigma values for each sample, derived from the binary search process to match the 
            target perplexity.
    
        Notes
        -----
        - The function assumes that `self.n_samples` defines the number of rows/samples in the distance matrix.
        - The target perplexity is defined as `self.PERPLEXITY`.
        - Progress is printed every 100 cells if `self.verbose` is True.
        """
        n = self.n_samples
        prob = np.zeros((n, n))
        sigma_array = []
        for dist_row in range(n):
            func = lambda sigma: self._perplexity(self.prob_high_dim(dist, sigma, dist_row))
            binary_search_result = self.sigma_binary_search(func, self.PERPLEXITY)
            prob[dist_row] = self.prob_high_dim(dist, binary_search_result, dist_row)
            sigma_array.append(binary_search_result)
            if self.verbose and (dist_row + 1) % 100 == 0:
                print("Sigma binary search finished {0} of {1} cells".format(dist_row + 1, n))
    
        sigma_array = np.array(sigma_array)
        # Compute the final symmetrized probability matrix
        P = (prob + np.transpose(prob)) / (n * 2)
        return P, sigma_array

    
       
    def prob_low_dim(self, Y):
        """
        Compute the low-dimensional conditional probability matrix for points in embedding space.
    
        This function calculates the pairwise similarities in the low-dimensional space using a Student's t-distribution
        (with one degree of freedom). It computes the inverse distances as:
        
            inv_distance = (1 + ||Y_i - Y_j||²)^(-1)
        
        The diagonal is set to zero to ignore self-similarity. The resulting matrix is normalized row-wise to form a 
        valid probability distribution for each point.
    
        Parameters
        ----------
        Y : np.ndarray
            A 2D array representing the low-dimensional embeddings of the data points, where each row corresponds 
            to a point in the embedding space.
    
        Returns
        -------
        np.ndarray
            A 2D array of the same shape as the input, where each element represents the normalized probability 
            (similarity) between points in the low-dimensional space.
        """
        inv_distances = np.power(1 + np.square(euclidean_distances(Y, Y)), -1)
        np.fill_diagonal(inv_distances, 0.)
        return inv_distances / np.sum(inv_distances, axis=1, keepdims=True)


    
    def kl_divergence(self, P, Q):
        """
        Compute the Kullback-Leibler (KL) divergence between two probability distributions.
    
        This function calculates the KL divergence between the high-dimensional probability matrix P 
        and the low-dimensional probability distribution Q, where Q is computed by applying the 
        low-dimensional probability function (using a Student's t-distribution) to the input Q. 
        A small constant (0.01) is added to both P and Q inside the logarithm to prevent numerical issues 
        with log(0).
    
        Parameters
        ----------
        P : np.ndarray
            A 2D array representing the high-dimensional probability distribution.
        Q : np.ndarray
            A 2D array representing the low-dimensional embedding of the data points. The function 
            internally transforms this into a probability distribution using `self.prob_low_dim`.
    
        Returns
        -------
        float
            The scalar KL divergence value between the two probability distributions.
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
        Fit the Et-SNE model to the data and transform it into a low-dimensional embedding.
    
        This method performs the following steps:
          1. Validates the input data ensuring it is a non-empty 2D array with at least two samples.
          2. Computes the squared pairwise Euclidean distance matrix from the input data.
          3. Initializes t-SNE parameters such as the number of samples, features, and components.
          4. Computes the high-dimensional probability matrix P using a binary search to determine optimal sigmas.
          5. Initializes the low-dimensional embedding Y with normally distributed random values.
          6. Performs gradient descent for a fixed number of iterations (self.n_iter), updating Y to minimize the 
             Kullback-Leibler divergence between the high-dimensional and low-dimensional distributions.
          7. Optionally prints the KL divergence every 50 iterations if verbose output is enabled.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data matrix where each row represents a sample and each column represents a feature.
        y : array-like, optional
            Target values (not used in this transformation but included for API consistency).
    
        Returns
        -------
        Y : np.ndarray of shape (n_samples, n_components)
            The low-dimensional embedding of the input data after t-SNE transformation.
        KL_array : np.ndarray
            An array containing the Kullback-Leibler divergence value at each iteration of the gradient descent,
            which can be used to monitor convergence.
    
        Raises
        ------
        ValueError
            If the input data X is None, empty, not a 2D array, or contains fewer than two samples.
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
