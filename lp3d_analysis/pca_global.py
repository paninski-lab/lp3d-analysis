
import os

import numpy as np
import pandas as pd
import pickle 

from eks.core import backward_pass, compute_initial_guesses, compute_nll, ensemble, forward_pass
from eks.ibl_paw_multiview_smoother import pca, remove_camera_means
from eks.stats import compute_mahalanobis
from eks.multicam_smoother import multicam_optimize_smooth, inflate_variance, make_dlc_pandas_index 
from lightning_pose.utils.pca import  NaNPCA 

from sklearn.decomposition import FactorAnalysis


class EnhancedFactorAnalysis(FactorAnalysis):
    """
    Enhanced Factor Analysis that extends scikit-learn's FactorAnalysis
    to handle ensemble variances when transforming data.
    
    This class inherits directly from sklearn.decomposition.FactorAnalysis
    and only modifies the transform method to handle ensemble variances.
    
    Parameters
    ----------
    n_components : int, default=3
        Number of latent components. Usually set to 3 for 3D pose estimation.
    
    Other parameters are inherited from sklearn.decomposition.FactorAnalysis.
    """

    def transform(self, X, v=None):
        """
        Transform data X into latent space.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to transform, where n_samples is the number of samples and n_features is the number
            of features.
        v : array-like, shape (n_samples, n_features), optional
            Ensemble variances for each observation. If None, standard FactorAnalysis transform
            is used.
            
        Returns
        -------
        z_hat : array-like, shape (n_samples, n_components)
            The latent variables.
        B : array-like, shape (n_samples, n_components, n_components)
            The posterior covariance matrices for the latent variables.
        """
        if v is None:
            # use standard FactorAnalysis transform
            return super().transform(X), None
        
        # Get the components and mean from the fit
        W = self.components_.T  # Shape: (n_features, n_components)
        mu_x = self.mean_       # Shape: Mean of the observations (n_features array)

        n_samples, n_features = X.shape
        epsilon = 1e-6

        B = np.zeros((n_samples, W.shape[1], W.shape[1]))
        z_hat = np.zeros((n_samples, W.shape[1]))  # (posterior mean)
        
        for i in range(n_samples):
            D_inv = np.diag(1.0 / (v[i] + epsilon))  # Compute D⁻¹ adding epsilon
            B[i] = np.linalg.inv(W.T @ D_inv @ W)    # B = (Wᵀ D⁻¹ W)⁻¹
            z_hat[i] = B[i] @ (W.T @ D_inv @ (X[i] - mu_x))
            
        return z_hat, B 

    def reconstruct(self, Z, latent_cov=None):
        """
        Reconstruct from latent variables back to the original feature space.
        
        Parameters
        ----------
        Z : array-like, shape (n_samples, n_components)
            Latent variables.
        latent_cov : array-like, shape (n_samples, n_components, n_components), optional
            Covariance matrices for the latent variables.
            
        Returns
        -------
        x_hat : array-like, shape (n_samples, n_features)
            Reconstructed data.
        cov : array-like, shape (n_samples, n_features)
            Diagonal elements of the covariance matrices for reconstructed data.
        """
        W = self.components_.T  # (n_features, n_components)
        mu_x = self.mean_       # (n_features,)
        n_samples, n_components = Z.shape
        n_features = W.shape[0]
        
        # Reconstruct mean: x_hat = W z + mu_x
        x_hat = np.dot(Z, W.T) + mu_x  # shape (n_samples, n_features)
        
        # If no latent covariances, return zeros for covariance
        if latent_cov is None:
            cov = np.zeros((n_samples, n_features))
            return x_hat, cov
        
        # Calculate reconstruction covariance from latent covariance
        full_cov = np.zeros((n_samples, n_features, n_features))
        for i in range(n_samples):
            full_cov[i] = W @ latent_cov[i] @ W.T
        
        # Extract diagonal for simplified representation
        # cov = np.diagonal(full_cov, axis1=1, axis2=2)  # Shape (n_samples, n_features)
        cov = full_cov
        
        return x_hat, cov
    
    def transform_reconstruct(self, X, v=None):
        """
        Transform data to latent space and then reconstruct it.
        This matches the behavior of the original reconstruct method.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to transform and reconstruct.
        v : array-like, shape (n_samples, n_features), optional
            Ensemble variances for each observation.
            
        Returns
        -------
        x_hat : array-like, shape (n_samples, n_features)
            Reconstructed data.
        cov : array-like, shape (n_samples, n_features)
            Diagonal elements of the covariance matrices.
        """
        z_hat, B = self.transform(X, v=v)
        return self.reconstruct(z_hat, latent_cov=B)
    
    