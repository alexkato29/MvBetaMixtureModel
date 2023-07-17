import h5py
import numpy as np
from scipy.stats import beta
from scipy.special import betaln
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from joblib import Parallel, delayed

class MVBetaMM:
    def __init__(self, n_mixtures=1, verbose=False, verbose_interval=10, random_state=256):
        """
        Initializes multivariate beta mixture model. It assumes multivariate via indepent beta distributions
        within mixtures, not via a Dirichlet distribution. This allows a sum > 1.

        Parameters:
        - n_mixtures (int): Number of MvBeta distributions in the model
        - verbose (boolean): If true, will print information during training
        - verbose_interval (int): Amount of iterations between verbose statements
        - random_state (int): Random state for all algorithms
        """
        self.n_mixtures = n_mixtures
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        self.random_state = random_state
        self.n_observations = None
        self.n_components = None
        self.converged_ = None
        self.weights_ = None
        self.params_ = None
        self.n_jobs = None
        self.method = None
    

    def _initialize(self, X, init_num):
        """
        Initializes the model parameters based on the given method

        Parameters:
        - X (matrix): Data to initialize with
        - init_num (int): Initialization number. Updates the random_state
        """
        # Initialize responsibilities based on method
        if self.method == "kmeans":
            resp = np.zeros(shape=(self.n_observations, self.n_mixtures))
            label = KMeans(n_clusters=self.n_mixtures, n_init=1, random_state=(self.random_state + init_num)).fit(X).labels_
            resp[np.arange(self.n_observations), label] = 1  # Set responsibility to 1 for the cluster it was assigned

        elif self.method == "random":
            np.random.seed(self.random_state + init_num)
            resp = np.random.uniform(size=(self.n_observations, self.n_mixtures))
            resp /= resp.sum(axis=1)[:, np.newaxis]
        
        # Add a small number for numerical stability (no log(0))
        resp += 10 * np.finfo(resp.dtype).eps

        # Compute the weights, alphas, and betas via M step
        self.params_ = np.zeros((self.n_mixtures, self.n_components*2))
        self._m_step(X, np.log(resp))
        self.verbose_initialization(init_num)

    
    def _estimate_log_weights(self):
        """
        Computes the log weights of the current model

        Returns:
        - log_weights (vector): Natural logarithm of the model weights
        """
        return np.log(self.weights_)
    
    
    def _compute_log_prob_for_mixture(self, X, mix):
        """
        Helper function to compute the log probability for a single mixture. Used for parallel computing

        Parameters:
        - X (matrix): Data
        - mix (int): Mixture number to assess the log probability of

        Returns:
        - log_prob (vector): Log probabilities of each observation associated with this mixture
        """
        alpha = self.params_[mix, :self.n_components]
        beta = self.params_[mix, self.n_components:]
        # Compute the log of the Beta function for each mixture
        log_beta_fn = betaln(alpha, beta)
        # Compute the log probability for each observation for current mixture
        log_prob = ((alpha - 1) * np.log(X) + (beta - 1) * np.log(1 - X) - log_beta_fn).sum(axis=1)
        return log_prob

    def _estimate_log_prob(self, X):
        """
        Estimates the log probability for all the mixtures

        Parameters:
        - X (matrix): Data

        Returns:
        - log_prob (matrix): Matrix of log probabilities. ij entry is the (unnormalized) log probability that 
                             observation i belongs to mixture j
        """
        if self.n_jobs == 1:
            log_prob = np.empty((self.n_observations, self.n_mixtures))
            for mix in range(self.n_mixtures):
                alpha = self.params_[mix, :self.n_components]
                beta = self.params_[mix, self.n_components:]

                # Compute the log of the Beta function for each mixture
                log_beta_fn = betaln(alpha, beta)

                # Compute the log probability for each observation for current mixture
                log_prob[:, mix] = ((alpha - 1) * np.log(X) + (beta - 1) * np.log(1 - X) - log_beta_fn).sum(axis=1)
            return log_prob

        else:
            log_prob = Parallel(n_jobs=self.n_jobs)(delayed(self._compute_log_prob_for_mixture)(X, mix) for mix in range(self.n_mixtures))
            return np.array(log_prob).T  # Transpose since the helper returns them as row vectors

    
    def _estimate_weighted_log_prob(self, X):
        """
        Estimates the weighted log probabilities for all the mixtures
        
        Parameters:
        - X (matrix): Data

        Returns:
        - weighted_log_prob (matrix): Matrix of weighted log probabilities. ij entry is the weighted (unnormalizd) 
                                      log probability that observation i belongs to mixture j
        """
        return self._estimate_log_prob(X) + self._estimate_log_weights()

    
    def _estimate_log_prob_resp(self, X):
        """
        Estimates the normalized log probabilites and the log responsiblities of each mixture

        Parameters:
        - X (matrix): Data

        Returns:
        - log_prob_norm (vector): Normalizing constant for each observation
        - log_resp (matrix): Matrix of log responsibilities. ij entry is the log prob that 
                             obs i belongs to mixture j (normalized and weighted)
        """
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)  # Normalizing constant

        # Ignore Underflow
        with np.errstate(under="ignore"):
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        
        return log_prob_norm, log_resp
    

    def _e_step(self, X):
        """
        Performs the expectation step of the EM algorithm

        Parameters:
        - X (matrix): Data
        
        Returns:
        - mean_log_prob_norm (matrix): Mean normalizing constant for all the observations
        - log_resp (matrix): Matrix of log responsibilities. ij entry is the log prob that 
                             obs i belongs to mixture j (normalized and weighted)
        """
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        return np.mean(log_prob_norm), log_resp
    

    def exp_responsibilities(self, log_resp):
        """
        Exponentiate the log responsibilities and compute the weighted importance of each mixture (unnormalized)

        Parameters:
        - log_resp (matrix): Matrix of log responsibilities. ij entry is the log prob that 
                             obs i belongs to mixture j (normalized and weighted)
        
        Returns:
        - resp (matrix): Exponentiated log responsibilites matrix
        - nk (vector): Weighted importance of each mixture
        """
        resp = np.exp(log_resp)
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps  # Number of elements in mixture k
        return resp, nk
    

    def update_weights(self, nk):
        """
        Updates the weights of the mixtures

        Parameters:
        - nk (vector): The sum of the probabilities of each mixture over every observation
        """
        self.weights_ = nk / np.sum(nk)
    

    def _m_step(self, X, log_resp):
        """
        Performs the M step of the EM algorithm via 1st and 2nd moment matching. Automatically
        updates the parameters of the model

        Parameters:
        - X (matrix): Data
        - log_resp (matrix): Matrix of log responsibilities. ij entry is the log prob that 
                             obs i belongs to mixture j (normalized and weighted)
        """

        # Update the weights
        resp, nk = self.exp_responsibilities(log_resp)
        self.update_weights(nk)
        
        # Calculate weighted sums and square sums for each mixture
        weighted_sums = resp.T @ X
        weighted_square_sums = resp.T @ (X ** 2)

        for i in range(self.n_mixtures):
            # Get weighted sum and square sum for this mixture
            weighted_sum = weighted_sums[i]
            weighted_square_sum = weighted_square_sums[i]
            
            # Calculate weighted mean and variance for each feature
            weighted_mean = weighted_sum / nk[i]
            weighted_variance = weighted_square_sum / nk[i] - weighted_mean ** 2

            # Compute the maximum possible weighted variance
            max_possible_weighted_variance = weighted_mean * (1 - weighted_mean) / 4
            weighted_variance = np.minimum(weighted_variance, max_possible_weighted_variance)
            weighted_variance += 10 * np.finfo(weighted_variance.dtype).eps

            # Calculate common factor once for each mixture
            common_factor = weighted_mean * (1 - weighted_mean) / (weighted_variance + 1e-10) - 1

            # Update parameters
            self.params_[i, :self.n_components] = common_factor * weighted_mean  # alphas
            self.params_[i, self.n_components:] = common_factor * (1 - weighted_mean)  # betas


    def verbose_initialization(self, n):
        if self.verbose:
            print(f"New {self.method} initialization. Init Number {n}")


    def verbose_iter(self, iter, lower_bound):
        if self.verbose and iter % self.verbose_interval == 0:
            print(f"Training Iteration {iter} complete. Best log probability lower bound: {lower_bound}")

    
    def verbose_converged(self, iter, lower_bound):
        if self.verbose:
            print(f"Converged on iteration {iter}. Log probability lower bound: {lower_bound}")


    def fit(self, X, n_init=10, method="kmeans", max_iter=100, tol=1e-5, n_jobs=1):
        """
        Fits the parameters and weights of the MVBeta model to maximize the loglikelihood of the model
        given the data X.

        Parameters:
        - X (matrix): Data to fit the model to
        - n_init (int): Number of initializations to try
        - max_iter (int): Maximum number of iterations allowed if convergence is not yet reached
        - tol (float): minimum allowed difference in log likelihoods before convergence is reached
        - n_jobs (int): Number of CPU cores to use on the E-Step (can significantly speed up compute)

        Returns:
        - self
        """
        self.n_observations, self.n_components = X.shape

        self.n_jobs = n_jobs
        self.converged_ = False
        max_lower_bound = -np.inf
        self.method = method if method.lower() in ["kmeans", "random"] else "kmeans"

        for init in range(n_init):
            self._initialize(X, init)
            lower_bound = -np.inf

            for iter in range(max_iter):
                prev_lower_bound = lower_bound
                log_prob_norm, log_resp = self._e_step(X)
                self._m_step(X, log_resp)

                lower_bound = log_prob_norm
                change = lower_bound - prev_lower_bound

                self.verbose_iter(iter, lower_bound)

                if abs(change) < tol:
                    self.verbose_converged(iter, lower_bound)
                    self.converged_ = True
                    break
            
            if lower_bound > max_lower_bound or max_lower_bound == -np.inf:
                max_lower_bound = lower_bound

                # Update the weights again to reflect the weights with the new parameters
                _, nk = self.exp_responsibilities(log_resp)
                self.update_weights(nk)

                best_params = [self.weights_, self.params_]
        
        self.weights_ = best_params[0]
        self.params_ = best_params[1]
        self.max_lower_bound = max_lower_bound
        
        return self
    

    def predict_proba(self, X):
        """
        Predcits the probability that X belongs to each of the distributions

        Parameters:
        - X (matrix): Data to probabilistically evaluate

        Returns:
        - Probs (matrix): NxK matrix where K is number of mixtures. ij is the probability obs i belongs to mixture k
        """
        _, log_resp = self._e_step(X)
        return np.exp(log_resp)
    

    def predict(self, X):
        """
        Predicts the most likely distribution for each observation in X

        Parameters:
        - X (matrix): Data to predict classes of

        Returns:
        - Classes (vector): The predicted classes [0, K-1] of the observations
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
