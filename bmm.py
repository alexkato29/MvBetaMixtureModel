import numpy as np
from scipy.stats import beta
from scipy.special import logsumexp
from sklearn.cluster import KMeans

class MvBetaMM:
    def __init__(self, n_components=1):
        self.n_components = n_components
        self.weights_ = None
        self.params_ = None

    def _loglikelihood(self, X):
        loglik = np.zeros((X.shape[0], self.n_components))
        for i in range(self.n_components):
            loglik[:, i] = np.sum(beta.logpdf(X, 
                                               self.params_[i, :X.shape[1]], 
                                               self.params_[i, X.shape[1]:]),
                                  axis=1) + np.log(self.weights_[i])
        return loglik

    def fit(self, X, max_iter=100, tol=1e-5):
        # KMeans Initialization
        kmeans = KMeans(n_init="auto", n_clusters=self.n_components).fit(X)
        labels = kmeans.labels_
        self.params_ = np.ones((self.n_components, X.shape[1]*2))  # alphas and betas
        self.weights_ = np.bincount(labels, minlength=self.n_components) / X.shape[0]

        # Moment Matching Initialization of alpha and beta
        X_means = np.array([X[labels == i].mean(axis=0) for i in range(self.n_components)])
        X_vars = np.array([X[labels == i].var(axis=0) for i in range(self.n_components)])
        self.params_[:, :X.shape[1]] = X_means * ((X_means * (1 - X_means) / X_vars) - 1)  # alphas
        self.params_[:, X.shape[1]:] = (1 - X_means) * ((X_means * (1 - X_means) / X_vars) - 1)  # betas

        loglik = self._loglikelihood(X)
        old_loglik = loglik.sum()

        for iter in range(max_iter):
            # E-step
            loglik -= logsumexp(loglik, axis=1)[:, np.newaxis]
            responsibilities = np.exp(loglik)

            # M-step
            self.weights_ = responsibilities.mean(axis=0)
            # Update parameters
            for i in range(self.n_components):
                resp = responsibilities[:, i]
                weighted_sum = np.sum(resp[:, np.newaxis] * X, axis=0)
                weighted_square_sum = np.sum(resp[:, np.newaxis] * (X ** 2), axis=0)
                
                # Calculate weighted mean and variance for each feature
                weighted_mean = weighted_sum / np.sum(resp)
                weighted_variance = weighted_square_sum / np.sum(resp) - weighted_mean ** 2
                
                # Update parameters using method of moments
                common_factor = weighted_mean * (1 - weighted_mean) / weighted_variance - 1
                self.params_[i, :X.shape[1]] = common_factor * weighted_mean
                self.params_[i, X.shape[1]:] = common_factor * (1 - weighted_mean)

            loglik = self._loglikelihood(X)
            new_loglik = loglik.sum()
            if np.abs(new_loglik - old_loglik) < tol:
                print(f"Converged after {iter} iterations")
                break
            old_loglik = new_loglik

        return self


    def predict_proba(self, X):
        loglik = self._loglikelihood(X)
        loglik -= logsumexp(loglik, axis=1)[:, np.newaxis]
        return np.exp(loglik)
    

    def predict_class(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
