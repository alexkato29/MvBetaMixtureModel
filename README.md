## MvBeta Mixture Model
Multivariate Beta Mixture Model provides a simple implementation of a beta mixture model using the EM algorithm. This is not a Dirichlet mixture model. MvBeta assumes that each mixture is comprised of multiple independent beta distributions, and their results are not constrained to sum to one.

### Features
The MvBetaMM object allows for simple training of a mixture model. For example usage, see the demonstration.ipynb notebook. 

### Installation
This code doesn't require a special installation process if Python and necessary libraries are already installed. Download the BetaMixture.py file and import the MvBetaMM class in your Python script:

```python
from BetaMixture.py import MvBetaMM
```

Required dependencies: NumPy, SciPy, sklearn, Joblib, and h5py. 

### Disclaimer
This package is a demonstration and may not be suited for production-level tasks without additional modifications and error handling. Use it at your own risk.

