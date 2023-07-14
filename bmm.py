import time
import h5py
import numpy as np
from scipy.stats import beta
from scipy.special import logsumexp
from sklearn.cluster import KMeans

class MVBetaMM:
    def __init__(self, n_mixtures=1):
        """
        Initializes multivariate beta mixture model. It assumes multivariate via indepent beta distributions
        within mixtures, not via a Dirichlet distribution. This allows a sum > 1.

        Parameters:
        - n_mixtures (int): Number of MvBeta distributions in the mixture
        """
        self.n_mixtures = n_mixtures
        self.n_components = None
        self.weights_ = None
        self.params_ = None


    def _loglikelihood(self, X):
        """
        Returns the log likelihood of each mixture for each observation in X

        Parameters:
        - X (matrix): Beta distributed data
        """
        # The ij element of loglik is the loglikelihood of obs i's belonging to mixture j
        loglik = np.zeros((X.shape[0], self.n_mixtures))

        for i in range(self.n_mixtures):
            # beta.logpdf will work in our multivariate sense. It assumes independent beta dists
            loglik[:, i] = np.sum(beta.logpdf(X, 
                                               self.params_[i, :self.n_components], 
                                               self.params_[i, self.n_components:]),
                                  axis=1) + np.log(self.weights_[i])
        return loglik


    def fit(self, X, max_iter=100, tol=1e-5):
        """
        Fits the parameters and weights of the MVBeta model to maximize the loglikelihood of the model
        given the data X.

        Parameters:
        - X (matrix): Data to fit the model to
        - max_iter (int): Maximum number of iterations allowed if convergence is not yet reached
        - tol (float): minimum allowed difference in log likelihoods before convergence is reached
        """
        n, d = X.shape

        # KMeans Initialization
        kmeans = KMeans(n_init="auto", n_clusters=self.n_mixtures).fit(X)
        labels = kmeans.labels_

        # Intialize Class Vars
        self.n_components = d
        self.params_ = np.ones((self.n_mixtures, self.n_components*2))  # params_ will have alphas first then betas
        self.weights_ = np.bincount(labels, minlength=self.n_mixtures) / n

        # Moment Matching Initialization of alpha and beta
        X_means = np.array([X[labels == i].mean(axis=0) for i in range(self.n_mixtures)])
        X_vars = np.array([X[labels == i].var(axis=0) for i in range(self.n_mixtures)])
        common_factor = X_means * (1 - X_means) / X_vars - 1
        self.params_[:, :self.n_components] = common_factor * X_means  # alphas
        self.params_[:, self.n_components:] = common_factor * (1 - X_means)  # betas

        loglik = self._loglikelihood(X)
        old_loglik = loglik.sum()

        for iter in range(max_iter):
            # E-step (slow operation)
            loglik -= logsumexp(loglik, axis=1)[:, np.newaxis]  # Normalizes each loglik value for each observation
            responsibilities = np.exp(loglik)  # exponentiaties to get probabilities
            self.weights_ = responsibilities.mean(axis=0)

            # M-step (quick operation)
            for i in range(self.n_mixtures):
                resp = responsibilities[:, i]
                weighted_sum = np.sum(resp[:, np.newaxis] * X, axis=0)
                weighted_square_sum = np.sum(resp[:, np.newaxis] * (X ** 2), axis=0)
                
                # Calculate weighted mean and variance for each feature
                weighted_mean = weighted_sum / np.sum(resp)
                weighted_variance = weighted_square_sum / np.sum(resp) - weighted_mean ** 2
                
                # Update parameters using method of moments
                common_factor = weighted_mean * (1 - weighted_mean) / weighted_variance - 1
                self.params_[i, :self.n_components] = common_factor * weighted_mean
                self.params_[i, self.n_components:] = common_factor * (1 - weighted_mean)

            # Update log likelihood (slow operation)
            loglik = self._loglikelihood(X)
            new_loglik = loglik.sum()
            lfin = time.time()

            # Check convergence
            if np.abs(new_loglik - old_loglik) < tol:
                print(f"Converged after {iter} iterations")
                break
            old_loglik = new_loglik

        return self


    def predict_proba(self, X):
        """
        Predcits the probability that X belongs to each of the distributions

        Parameters:
        - X (matrix): Data to probabilistically evaluate

        Returns:
        - Probs (matrix): NxK matrix where K is number of mixtures. ij is the probability obs i belongs to mixture k
        """
        loglik = self._loglikelihood(X)
        loglik -= logsumexp(loglik, axis=1)[:, np.newaxis]
        return np.exp(loglik)
    

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
    

    def save_model(self, file_path):
        """
        Save the model in h5 format for later use

        Parameters:
        - file_path (str): Path to the file you want to save the model to
        """
        with h5py.File(file_path, "w") as f:
            f.create_dataset("weights", data=self.weights_)
            f.create_dataset("params", data=self.params_)
            f.create_dataset("n_mixtures", data=self.n_mixtures)
            f.create_dataset("n_components", data=self.n_components)

    
    def load_model(self, file_path):
        """
        Load a model from an h5 file

        Parameters:
        - file_path (str): Path to the file you want to load
        """
        with h5py.File(file_path, "r") as f:
            self.weights_ = f["weights"][()]
            self.params_ = f["params"][()]
            self.n_mixtures = f["n_mixtures"][()]
            self.n_components = f["n_components"][()]
