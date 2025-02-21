
import os

import numpy as np
import pandas as pd
import pickle 

from eks.core import backward_pass, compute_initial_guesses, compute_nll, ensemble, forward_pass
from eks.ibl_paw_multiview_smoother import pca, remove_camera_means
from eks.stats import compute_mahalanobis
from eks.multicam_smoother import multicam_optimize_smooth, inflate_variance, make_dlc_pandas_index 
from lightning_pose.utils.pca import  NaNPCA 



def ensemble_kalman_smoother_multicam2(
    markers_list: list,
    keypoint_names: list,
    smooth_param: float | list | None = None,
    quantile_keep_pca: float = 95.0,
    camera_names: list | None = None,
    s_frames: list | None = None,
    avg_mode: str = 'median',
    var_mode: str = 'confidence_weighted_var',
    inflate_vars: bool = False,
    inflate_vars_kwargs: dict = {},
    verbose: bool = False,
    pca_object = None,
) -> tuple:
    """
    Use multi-view constraints to fit a 3D latent subspace for each body part.

    Args:
        markers_list: List of lists of pd.DataFrames, where each inner list contains
            DataFrame predictions for a single camera.
        keypoint_names: List of body parts to run smoothing on
        smooth_param: Value in (0, Inf); smaller values lead to more smoothing (default: None).
        quantile_keep_pca: Percentage of points kept for PCA (default: 95).
        camera_names: List of camera names corresponding to the input data (default: None).
        s_frames: Frames for auto-optimization if smooth_param is not provided (default: None).
        avg_mode: mode for averaging across ensemble
            'median' | 'mean'
        var_mode: mode for computing ensemble variance
            'var' | 'confidence_weighted_var'
        inflate_vars: True to use Mahalanobis distance thresholding to inflate ensemble variance
        inflate_vars_kwargs: kwargs for compute_mahalanobis function when running variance inflation

        verbose: True to print out details

    Returns:
        tuple: Dataframes with smoothed predictions, final smoothing parameters.
    """

    # Collection array for marker output by camera view
    camera_arrs = [[] for _ in camera_names]

    # Loop over keypoints; apply EKS to each individually
    for k, keypoint in enumerate(keypoint_names):
        # Setup: Interpolate right cam markers to left cam timestamps
        markers_list_cameras = markers_list[k]
        num_cameras = len(camera_names)

        markers_list_stacked_interp = []
        markers_list_interp = [[] for _ in range(num_cameras)]
        camera_likelihoods_stacked = []

        for model_id in range(len(markers_list_cameras[0])):
            bl_markers_curr = []
            camera_markers_curr = [[] for _ in range(num_cameras)]
            camera_likelihoods = [[] for _ in range(num_cameras)]

            for i in range(markers_list_cameras[0][0].shape[0]):
                curr_markers = []

                for camera in range(num_cameras):
                    markers = np.array(
                        markers_list_cameras[camera][model_id].to_numpy()[i, [0, 1]]
                    )
                    likelihood = np.array(
                        markers_list_cameras[camera][model_id].to_numpy()[i, [2]]
                    )[0]

                    camera_markers_curr[camera].append(markers)
                    curr_markers.append(markers)
                    camera_likelihoods[camera].append(likelihood)

                # Combine predictions for all cameras
                bl_markers_curr.append(np.concatenate(curr_markers))

            markers_list_stacked_interp.append(bl_markers_curr)
            camera_likelihoods_stacked.append(camera_likelihoods)

            camera_likelihoods = np.asarray(camera_likelihoods)
            for camera in range(num_cameras):
                markers_list_interp[camera].append(camera_markers_curr[camera])
                camera_likelihoods[camera] = np.asarray(camera_likelihoods[camera])

        markers_list_stacked_interp = np.asarray(markers_list_stacked_interp)
        markers_list_interp = np.asarray(markers_list_interp)
        camera_likelihoods_stacked = np.asarray(camera_likelihoods_stacked)

        keys = [f"{keypoint}_x", f"{keypoint}_y"]
        markers_list_cams = [[] for _ in range(num_cameras)]

        for k in range(len(markers_list_interp[0])):
            for camera in range(num_cameras):
                markers_cam = pd.DataFrame(markers_list_interp[camera][k], columns=keys)
                markers_cam[f'{keypoint}_likelihood'] = camera_likelihoods_stacked[k][camera]
                markers_list_cams[camera].append(markers_cam)

        # Compute ensemble median for each camera
        cam_ensemble_preds = []
        cam_ensemble_vars = []
        cam_ensemble_likes = []
        cam_ensemble_stacks = []

        for camera in range(num_cameras):
            (
                cam_ensemble_preds_curr,
                cam_ensemble_vars_curr,
                cam_ensemble_likes_curr,
                cam_ensemble_stacks_curr,
            ) = ensemble(markers_list_cams[camera], keys, avg_mode=avg_mode, var_mode=var_mode)

            cam_ensemble_preds.append(cam_ensemble_preds_curr)
            cam_ensemble_vars.append(cam_ensemble_vars_curr)
            cam_ensemble_likes.append(cam_ensemble_likes_curr)
            cam_ensemble_stacks.append(cam_ensemble_stacks_curr)

        # Filter by low ensemble variances
        hstacked_vars = np.hstack(cam_ensemble_vars)
        max_vars = np.max(hstacked_vars, 1)
        good_frames = np.where(max_vars <= np.percentile(max_vars, quantile_keep_pca))[0]

        good_cam_ensemble_preds = [
            cam_ensemble_preds[camera][good_frames] for camera in range(num_cameras)
        ]

        good_ensemble_preds = np.hstack(good_cam_ensemble_preds)
        means_camera = [good_ensemble_preds[:, i].mean() for i in
                        range(good_ensemble_preds.shape[1])]

        ensemble_preds = np.hstack(cam_ensemble_preds)
        ensemble_vars = np.hstack(cam_ensemble_vars)
        ensemble_likes = np.hstack(cam_ensemble_likes)
        ensemble_stacks = np.concatenate(cam_ensemble_stacks, 2)
        remove_camera_means(ensemble_stacks, means_camera)
        good_scaled_ensemble_preds = remove_camera_means(
            good_ensemble_preds[None, :, :], means_camera
        )[0]

        if pca_object is not None:
            ensemble_pca = pca_object
            inflate_vars_kwargs["loading_matrix"] = ensemble_pca.components_.T
            inflate_vars_kwargs["mean"] = ensemble_pca.mean_
        else:
            ensemble_pca, ensemble_ex_var = pca(good_scaled_ensemble_preds, 3)

        scaled_ensemble_preds = remove_camera_means(
            ensemble_preds[None, :, :], means_camera
        )[0]

        ensemble_pcs = ensemble_pca.transform(scaled_ensemble_preds)
        good_ensemble_pcs = ensemble_pcs[good_frames]
        y_obs = scaled_ensemble_preds

        if inflate_vars:
            # set some maha defaults
            if 'likelihood_threshold' not in inflate_vars_kwargs:
                inflate_vars_kwargs['likelihood_threshold'] = 0.9
            if 'v_quantile_threshold' not in inflate_vars_kwargs:
                inflate_vars_kwargs['v_quantile_threshold'] = 50.0
            inflated = True
            tmp_vars = ensemble_vars
            while inflated:
                # Compute Mahalanobis distances
                maha_results = compute_mahalanobis(
                    y_obs, tmp_vars,
                    likelihoods=ensemble_likes,
                    **inflate_vars_kwargs,
                )
                # Inflate variances based on Mahalanobis distances
                inflated_ens_vars, inflated = inflate_variance(
                    tmp_vars, maha_results['mahalanobis'], threshold=5, scalar=2,
                )
                tmp_vars = inflated_ens_vars
        else:
            inflated_ens_vars = ensemble_vars

        # Kalman Filter
        m0 = np.asarray([0.0, 0.0, 0.0])
        S0 = np.asarray([[np.var(good_ensemble_pcs[:, 0]), 0.0, 0.0],
                         [0.0, np.var(good_ensemble_pcs[:, 1]), 0.0],
                         [0.0, 0.0, np.var(good_ensemble_pcs[:, 2])]])  # diagonal: var
        A = np.eye(3)
        d_t = good_ensemble_pcs[1:] - good_ensemble_pcs[:-1]
        C = ensemble_pca.components_.T
        R = np.eye(ensemble_pca.components_.shape[1])
        cov_matrix = np.cov(d_t.T)
        smooth_param_final, ms, Vs, _, _ = multicam_optimize_smooth(
            cov_matrix, y_obs, m0, S0, C, A, R, inflated_ens_vars, s_frames, smooth_param
        )
        if verbose:
            print(f"Smoothed {keypoint} at smooth_param={smooth_param_final}")

        y_m_smooth = np.dot(C, ms.T).T
        y_v_smooth = np.swapaxes(np.dot(C, np.dot(Vs, C.T)), 0, 1)

        # Final cleanup
        c_i = [[camera * 2, camera * 2 + 1] for camera in range(num_cameras)]
        for c, camera in enumerate(camera_names):
            data_arr = camera_arrs[c]
            x_i, y_i = c_i[c]

            data_arr.extend([
                y_m_smooth.T[x_i] + means_camera[x_i],
                y_m_smooth.T[y_i] + means_camera[y_i],
                ensemble_likes[:, x_i],
                ensemble_preds[:, x_i],
                ensemble_preds[:, y_i],
                inflated_ens_vars[:, x_i],
                inflated_ens_vars[:, y_i],
                y_v_smooth[:, x_i, x_i],
                y_v_smooth[:, y_i, y_i],
            ])

    labels = [
        'x', 'y', 'likelihood', 'x_ens_median', 'y_ens_median',
        'x_ens_var', 'y_ens_var', 'x_posterior_var', 'y_posterior_var'
    ]

    pdindex = make_dlc_pandas_index(keypoint_names, labels=labels)
    camera_dfs = []

    for c, camera in enumerate(camera_names):
        camera_arr = np.asarray(camera_arrs[c])
        camera_df = pd.DataFrame(camera_arr.T, columns=pdindex)
        camera_dfs.append(camera_df)

    return camera_dfs, smooth_param_final


class NaNPCA2(NaNPCA):

    def transform(self,X):
        # check_is_fitted(self)
        is_valid = ~np.isnan(X)

        # X = self._validate_data(X, dtype=[np.float64, np.float32], reset=False)

        if self.mean_ is not None:
            X = X - self.mean_

        X_ = X.copy()
        X_[~is_valid] = 0

        X_transformed = np.zeros((X_.shape[0], self.n_components_))
        W = self.components_.T
        for i in range(X_.shape[0]):
            if is_valid[i].sum() == 0:
                X_transformed[i] = 0
            else:
                try:
                    cov_mat = np.diag(1.0 * is_valid[i])
                    B = np.linalg.inv(W.T @ cov_mat @ W)
                    z_hat = B @ W.T @ cov_mat @ X_[i]  # compute posterior mean
                    X_transformed[i] = z_hat
                except:
                    X_transformed[i] = 0

        if self.whiten:
            X_transformed /= np.sqrt(self.explained_variance_)

        return X_transformed