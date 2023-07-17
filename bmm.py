import time
import h5py
import numpy as np
from scipy.stats import beta
from scipy.special import betaln
from scipy.special import logsumexp
from sklearn.cluster import KMeans

class MVBetaMM:
    def __init__(self, n_mixtures=1, verbose=False, verbose_interval=10, random_state=256):
        """
        Initializes multivariate beta mixture model. It assumes multivariate via indepent beta distributions
        within mixtures, not via a Dirichlet distribution. This allows a sum > 1.

        Parameters:
        - n_mixtures (int): Number of MvBeta distributions in the mixture
        - verbose (boolean): If true, will print information during training
        - verbose_interval (int): Amount of iterations between verbose statements
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
        self.method = None
    

    def _initialize(self, X, init):
        # Initialize responsibilities based on method
        if self.method == "kmeans":
            resp = np.zeros(shape=(self.n_observations, self.n_mixtures))
            label = KMeans(n_clusters=self.n_mixtures, n_init=1, random_state=(self.random_state + init)).fit(X).labels_
            resp[np.arange(self.n_observations), label] = 1  # Set responsibility to 1 for the cluster it was assigned

        elif self.method == "random":
            np.random.seed(self.random_state + init)
            resp = np.random.uniform(size=(self.n_observations, self.n_mixtures))
            resp /= resp.sum(axis=1)[:, np.newaxis]
        
        resp += 10 * np.finfo(resp.dtype).eps
        self.params_ = np.zeros((self.n_mixtures, self.n_components*2))
        self._m_step(X, np.log(resp))
        self.verbose_initialization()

    
    def _estimate_log_weights(self):
        return np.log(self.weights_)

    
    def _estimate_log_prob(self, X):
        log_prob = np.empty((self.n_observations, self.n_mixtures))
        alpha = self.params_[:, :self.n_components]
        beta = self.params_[:, self.n_components:]

        # Compute the log of the Beta function for each mixture
        log_beta_fn = betaln(alpha, beta)

        # Compute the log probability for each observation and each mixture
        log_prob = (alpha - 1) * np.log(X[:, np.newaxis, :]) + (beta - 1) * np.log(1 - X[:, np.newaxis, :]) - log_beta_fn

        # Sum over features
        log_prob = log_prob.sum(axis=2)

        return log_prob

    
    def _estimate_weighted_log_prob(self, X):
        return self._estimate_log_prob(X) + self._estimate_log_weights()

    
    def _estimate_log_prob_resp(self, X):
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)  # Normalizing constant
        with np.errstate(under="ignore"):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, log_resp
    

    def _e_step(self, X):
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        return np.mean(log_prob_norm), log_resp
    

    def _m_step(self, X, log_resp):
        # Number of elements in mixture k
        resp = np.exp(log_resp)
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        self.weights_ = nk / np.sum(nk)
        
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


    def verbose_initialization(self):
        if self.verbose:
            print(f"New {self.method} initialization.")
            print(self.params_)
            print(self.weights_)


    def verbose_iter(self, iter, lower_bound):
        if self.verbose and iter % self.verbose_interval == 0:
            print(f"Training Iteration {iter} complete. Best log probability lower bound: {lower_bound}")
            print(self.weights_)
            print(self.params_)

    
    def verbose_converged(self, iter, lower_bound):
        if self.verbose:
            print(f"Converged on iteration {iter}. Log probability lower bound: {lower_bound}")


    def fit(self, X, n_init=10, method="kmeans", max_iter=100, tol=1e-5):
        """
        Fits the parameters and weights of the MVBeta model to maximize the loglikelihood of the model
        given the data X.

        Parameters:
        - X (matrix): Data to fit the model to
        - n_init (int): Number of initializations to try
        - max_iter (int): Maximum number of iterations allowed if convergence is not yet reached
        - tol (float): minimum allowed difference in log likelihoods before convergence is reached
        """
        self.n_observations, self.n_components = X.shape

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
        log_prob_norm, log_resp = self._e_step(X)
        return np.exp(log_resp)
    

    def predict_class(self, X):
        """
        Predicts the most likely distribution for each observation in X

        Parameters:
        - X (matrix): Data to predict classes of

        Returns:
        - Classes (vector): The predicted classes [0, K-1] of the observations
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
