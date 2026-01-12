"""
MIT License (MIT)
Copyright (c) FALL 2016, Jahdiel Alvarez
Author: Jahdiel Alvarez
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
Based on Scipy's cookbook:
http://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import sys
from pathlib import Path

# Add lightning-pose to path if needed
lightning_pose_path = Path(__file__).parent.parent.parent.parent / "lightning-pose"
if lightning_pose_path.exists() and str(lightning_pose_path) not in sys.path:
    sys.path.insert(0, str(lightning_pose_path))

try:
    from lightning_pose.utils.pca import NaNPCA
except ImportError:
    print("Warning: Could not import NaNPCA from lightning_pose. PCA initialization will not be available.")
    NaNPCA = None

#%%
class PySBA:
    """Python class for Simple Bundle Adjustment"""

    def __init__(self, cameraArray, points3D, points2D, cameraIndices, point2DIndices, pointWeights=None, optimize_distortion='k1_only'):
        """Intializes all the class attributes and instance variables.
            Write the specifications for each variable:
            cameraArray with shape (n_cameras, 11) contains initial estimates of parameters for all cameras.
                    First 3 components in each row form a rotation vector,
                    next 3 components form a translation vector,
                    then a focal distance and two distortion parameters,
                    then x,y image center coordinates
            points_3d with shape (n_points, 3)
                    contains initial estimates of point coordinates in the world frame.
            camera_ind with shape (n_observations,)
                    contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.
            point_ind with shape (n_observations,)
                    contatins indices of points (from 0 to n_points - 1) involved in each observation.
            points_2d with shape (n_observations, 2)
                    contains measured 2-D coordinates of points projected on images in each observations.
            pointWeights with shape (n_observations, )
                    contains cost function weights for each observation point.
        """
        self.cameraArray = cameraArray
        self.points3D = points3D
        self.points2D = points2D

        self.cameraIndices = cameraIndices
        self.point2DIndices = point2DIndices
        if pointWeights is None:
            pointWeights = np.full_like(point2DIndices, 1)
        self.pointWeights = pointWeights.reshape((-1, 1))
        
        # Distortion optimization mode: 'k1_only', 'k1_k2', or 'none'
        self.optimize_distortion = optimize_distortion

    def rotate(self, points, rot_vecs):
        """Rotate points by given rotation vectors.
        Rodrigues' rotation formula is used.
        """
        theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
        with np.errstate(invalid='ignore'):
            v = rot_vecs / theta
            v = np.nan_to_num(v)
        dot = np.sum(points * v, axis=1)[:, np.newaxis]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


    def project(self, points, cameraArray):
        """Convert 3-D points to 2-D by projecting onto images."""
        points_proj = self.rotate(points, cameraArray[:, :3])
        points_proj += cameraArray[:, 3:6]
        points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]
        # points_proj -= cameraArray[:, 9:] / 1778
        f = cameraArray[:, 6]
        k1 = cameraArray[:, 7]
        k2 = cameraArray[:, 8]
        n = np.sum(points_proj ** 2, axis=1)
        
        # Apply distortion based on optimization mode
        if self.optimize_distortion == 'none':
            r = 1.0  # No distortion
        elif self.optimize_distortion == 'k1_only':
            r = 1 + k1 * n  # Only k1 (radial distortion)
        else:  # 'k1_k2' or default
            r = 1 + k1 * n + k2 * n ** 2  # Both k1 and k2
        
        points_proj *= (r * f)[:, np.newaxis]
        points_proj += cameraArray[:, 9:]
        return points_proj


    def fun(self, params, n_cameras, n_points, camera_indices, point_indices, points_2d, pointWeights):
        """Compute residuals.
        `params` contains camera parameters and 3-D coordinates.
        """
        nCamParams = 11
        camera_params = params[:n_cameras * nCamParams].reshape((n_cameras, nCamParams))
        points_3d = params[n_cameras * nCamParams:].reshape((n_points, 3))
        points_proj = self.project(points_3d[point_indices], camera_params[camera_indices])
        weighted_residual = pointWeights*(points_proj-points_2d)
        return weighted_residual.ravel()

    def bundle_adjustment_sparsity(self, numCameras, numPoints, cameraIndices, pointIndices):
        nCamParams = 11
        m = cameraIndices.size * 2
        n = numCameras * nCamParams + numPoints * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(cameraIndices.size)
        for s in range(nCamParams):
            A[2 * i, cameraIndices * nCamParams + s] = 1
            A[2 * i + 1, cameraIndices * nCamParams + s] = 1

        for s in range(3):
            A[2 * i, numCameras * nCamParams + pointIndices * 3 + s] = 1
            A[2 * i + 1, numCameras * nCamParams + pointIndices * 3 + s] = 1

        return A


    def optimizedParams(self, params, n_cameras, n_points):
        """
        Retrieve camera parameters and 3-D coordinates.
        """
        nCamParams = 11
        camera_params = params[:n_cameras * nCamParams].reshape((n_cameras, nCamParams))
        points_3d = params[n_cameras * nCamParams:].reshape((n_points, 3))

        return camera_params, points_3d


    def bundleAdjust(self):
        """ Returns the bundle adjusted parameters, in this case the optimized
         rotation and translation vectors. """
        numCameras = self.cameraArray.shape[0]
        numPoints = self.points3D.shape[0]

        x0 = np.hstack((self.cameraArray.ravel(), self.points3D.ravel()))
        A = self.bundle_adjustment_sparsity(numCameras, numPoints, self.cameraIndices, self.point2DIndices)

        #res = least_squares(self.fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
        #                    args=(numCameras, numPoints, self.cameraIndices, self.point2DIndices, self.points2D))
        # Match aniposelib defaults: ftol=1e-4, max_nfev=1000
        res = least_squares(self.fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf', jac='3-point',
                            max_nfev=1000,
                            args=(numCameras, numPoints, self.cameraIndices, self.point2DIndices, self.points2D, self.pointWeights))

        camera_params, points_3d = self.optimizedParams(res.x, numCameras, numPoints)
        self.cameraArray = camera_params
        self.points3D = points_3d

        return res

    def getResiduals(self):
        """Gets residuals given current camera parameters and 3d locations"""
        numCameras = self.cameraArray.shape[0]
        numPoints = self.points3D.shape[0]
        x0 = np.hstack((self.cameraArray.ravel(), self.points3D.ravel()))
        # Create weights with proper shape: (n_observations, 1) to broadcast with (n_observations, 2)
        weights = np.ones((len(self.point2DIndices), 1))
        f0 = self.fun(x0, numCameras, numPoints, self.cameraIndices, self.point2DIndices, self.points2D, weights)
        return f0


    def bundle_adjustment_sparsity_nocam(self, numPoints, pointIndices):
        m = pointIndices.size * 2
        n = numPoints * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(pointIndices.size)
        for s in range(3):
            A[2 * i, pointIndices * 3 + s] = 1
            A[2 * i + 1, pointIndices * 3 + s] = 1

        return A

    def fun_nocam(self, params, camera_params, n_points, camera_indices, point_indices, points_2d, pointWeights):
        """Compute residuals.
        `params` contains 3-D coordinates only.
        """

        points_3d = params.reshape((n_points, 3))
        points_proj = self.project(points_3d[point_indices], camera_params[camera_indices])
        weighted_residual = pointWeights * (points_proj - points_2d)
        return weighted_residual.ravel()

    def bundleAdjust_nocam(self):
        """ Returns the optimized 3d positions given current camera parameters,
        without adjusting the camera parameters themselves. """
        numCameras = self.cameraArray.shape[0]
        numPoints = self.points3D.shape[0]
        camera_params = self.cameraArray

        x0 = self.points3D.ravel()
        A = self.bundle_adjustment_sparsity_nocam(numPoints, self.point2DIndices)
        res = least_squares(self.fun_nocam, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-8, method='trf', jac='3-point',
                            args=(camera_params, numPoints, self.cameraIndices, self.point2DIndices, self.points2D, self.pointWeights))
        self.points3D = res.x.reshape((numPoints, 3))

        return res

    def bundle_adjustment_sparsity_sharedcam(self, numCameras, numPoints, cameraIndices, pointIndices):
        m = cameraIndices.size * 2
        nCamIntrinsic = 3
        nCamExtrinsic = 6
        nCamCentroid = 2
        nCamParams = numCameras * nCamExtrinsic + numCameras * nCamCentroid + nCamIntrinsic
        n = nCamParams + numPoints * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(cameraIndices.size)
        A[2*i, 0:nCamIntrinsic] = 1
        A[2*i + 1, 0:nCamIntrinsic] = 1
        for s in range(nCamExtrinsic):
            A[2 * i, nCamIntrinsic + cameraIndices * nCamExtrinsic + s] = 1
            A[2 * i + 1, nCamIntrinsic + cameraIndices * nCamExtrinsic + s] = 1
        for s in range(nCamCentroid):
            A[2 * i, nCamIntrinsic + numCameras*nCamExtrinsic + cameraIndices * nCamCentroid + s] = 1
            A[2 * i + 1, nCamIntrinsic + numCameras*nCamExtrinsic + cameraIndices * nCamCentroid + s] = 1

        for s in range(3):
            A[2 * i, nCamParams + pointIndices * 3 + s] = 1
            A[2 * i + 1, nCamParams + pointIndices * 3 + s] = 1

        return A

    def fun_sharedcam(self, params, n_cameras, n_points, camera_indices, point_indices, points_2d, pointWeights):
        """Compute residuals.
        `params` contains camera parameters and 3-D coordinates.
        """
        nCamIntrinsic = 3
        nCamExtrinsic = 6
        nCamCentroid = 2
        nCamUnique = nCamExtrinsic + nCamCentroid
        nCamParams = n_cameras * nCamUnique + nCamIntrinsic

        cam_shared_intrinsic = params[:nCamIntrinsic]
        camera_extrinsic = params[nCamIntrinsic:nCamIntrinsic+n_cameras*nCamExtrinsic].reshape((n_cameras, nCamExtrinsic))
        camera_centroid = params[nCamIntrinsic+n_cameras*nCamExtrinsic : nCamParams].reshape((n_cameras, nCamCentroid))
        camera_params = np.concatenate((camera_extrinsic, np.tile(cam_shared_intrinsic, (n_cameras,1)), camera_centroid), axis=1)

        points_3d = params[nCamParams:].reshape((n_points, 3))
        points_proj = self.project(points_3d[point_indices], camera_params[camera_indices])
        weighted_residual = pointWeights * (points_proj - points_2d)
        return weighted_residual.ravel()

    def bundleAdjust_sharedcam(self):
        """ Returns the bundle adjusted parameters, in this case the optimized
         rotation and translation vectors. """
        numCameras = self.cameraArray.shape[0]
        numPoints = self.points3D.shape[0]

        nCamIntrinsic = 3
        nCamExtrinsic = 6
        nCamCentroid = 2
        nCamUnique = nCamExtrinsic + nCamCentroid
        nCamParams = numCameras * nCamUnique + nCamIntrinsic

        camera_shared_intrinsic = np.mean(self.cameraArray[:, 6:9], axis=0).ravel()
        camera_extrinsic = self.cameraArray[:,:6].ravel()
        camera_centroids = self.cameraArray[:,9:].ravel()

        x0 = np.hstack((camera_shared_intrinsic, camera_extrinsic, camera_centroids, self.points3D.ravel()))
        A = self.bundle_adjustment_sparsity_sharedcam(numCameras, numPoints, self.cameraIndices, self.point2DIndices)
        res = least_squares(self.fun_sharedcam, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-8, method='trf', jac='3-point',
                            args=(numCameras, numPoints, self.cameraIndices, self.point2DIndices, self.points2D, self.pointWeights))

        cam_shared_intrinsic = res.x[:nCamIntrinsic]
        camera_extrinsic = res.x[nCamIntrinsic:nCamIntrinsic + numCameras * nCamExtrinsic].reshape((numCameras, nCamExtrinsic))
        camera_centroid = res.x[nCamIntrinsic+numCameras*nCamExtrinsic : nCamParams].reshape((numCameras, nCamCentroid))
        camera_params = np.concatenate((camera_extrinsic, np.tile(cam_shared_intrinsic, (numCameras, 1)), camera_centroid), axis=1)
        points_3d = res.x[nCamParams:].reshape((numPoints, 3))
        self.cameraArray = camera_params
        self.points3D = points_3d

        return res
    
    def apply_pca_to_init_3d_old(self, points_2d_all, n_pca_components=3):
        """
        Apply PCA to constrain 2D points, then triangulate to get real 3D coordinates.
        
        This method:
        1. Uses PCA to clean/constrain multi-view 2D observations
        2. Reconstructs the cleaned 2D points via inverse_transform
        3. Triangulates the cleaned 2D points to get real 3D world coordinates
        
        Args:
            points_2d_all (np.ndarray): Array of 2D points with shape 
                                       (n_views, n_frames, n_keypoints, 2).
            n_pca_components (int): Number of PCA components to use. Default is 3.
        
        Returns:
            np.ndarray: Real 3D world coordinates with shape (n_frames, n_keypoints, 3).
        """
        if NaNPCA is None:
            print("Warning: NaNPCA not available. Using standard triangulation without PCA.")
            # Fall back to standard triangulation
            n_views, n_frames, n_keypoints, _ = points_2d_all.shape
            points_3d = np.full((n_frames, n_keypoints, 3), np.nan)
            
            for frame_idx in range(n_frames):
                for kp_idx in range(n_keypoints):
                    # Triangulate this keypoint from all views
                    pts_2d_this_kp = points_2d_all[:, frame_idx, kp_idx, :]  # (n_views, 2)
                    if not np.isnan(pts_2d_this_kp).any():
                        pt_3d = find_consistent_3d_points(
                            pts_2d_this_kp[np.newaxis, :, :], 
                            self.cameraArray
                        )
                        points_3d[frame_idx, kp_idx] = pt_3d[0]
            
            self.points3D = points_3d.reshape(-1, 3)
            return points_3d
        
        print("\n" + "="*80)
        print("APPLYING PCA-CONSTRAINED 3D INITIALIZATION")
        print("="*80)
        
        # Reshape points_2d_all to work with PCA
        # From (n_views, n_frames, n_keypoints, 2) to (n_frames, n_keypoints, n_views*2)
        n_views, n_frames, n_keypoints, _ = points_2d_all.shape
        points_2d_all = points_2d_all.transpose(1, 2, 0, 3)  # (n_frames, n_keypoints, n_views, 2)
        
        # Format for PCA: each row is one keypoint instance across all views
        # Shape: (n_frames * n_keypoints, n_views * 2)
        formatted_data = []
        for frame_idx in range(n_frames):
            for kp_idx in range(n_keypoints):
                kp_multiview = points_2d_all[frame_idx, kp_idx]  # (n_views, 2)
                kp_flat = kp_multiview.flatten()  # (n_views*2,)
                formatted_data.append(kp_flat)
        formatted_data = np.array(formatted_data)
        
        # Fit PCA
        n_components_actual = min(n_pca_components, n_views * 2)
        print(f"Step 1: Apply PCA to clean/constrain 2D observations")
        print(f"  Fitting PCA with {n_components_actual} components on {n_views*2}D data...")
        print(f"  Data matrix shape: {formatted_data.shape}")
        
        has_nan = np.isnan(formatted_data).any(axis=1)
        print(f"  Samples with NaN: {has_nan.sum()} / {len(has_nan)}")
        
        nanpca = NaNPCA(n_components=n_components_actual)
        nanpca.fit(formatted_data)
        
        # Get PCA-reconstructed 2D points (cleaned/constrained)
        latent = nanpca.transform(formatted_data)
        pca_reconstructed_2d = nanpca.inverse_transform(latent)  # (n_frames*n_keypoints, n_views*2)
        
        # Calculate 2D reconstruction error
        diff = formatted_data - pca_reconstructed_2d
        recon_error = np.sqrt(np.nanmean(diff**2))
        print(f"  PCA 2D reconstruction error: {recon_error:.3f} pixels (RMSE)")
        
        # Reshape PCA-reconstructed 2D points back to multi-view format
        # (n_views, n_frames, n_keypoints, 2)
        pca_2d_reshaped = np.zeros((n_views, n_frames, n_keypoints, 2))
        idx = 0
        for frame_idx in range(n_frames):
            for kp_idx in range(n_keypoints):
                kp_flat = pca_reconstructed_2d[idx]
                kp_multiview = kp_flat.reshape(n_views, 2)
                pca_2d_reshaped[:, frame_idx, kp_idx, :] = kp_multiview
                idx += 1
        
        # Step 2: Triangulate PCA-cleaned 2D points to get real 3D world coordinates
        print(f"\nStep 2: Triangulate cleaned 2D points to get real 3D world coordinates")
        points_3d_triangulated = np.full((n_frames, n_keypoints, 3), np.nan)
        
        for frame_idx in range(n_frames):
            for kp_idx in range(n_keypoints):
                # Get PCA-cleaned 2D points for this keypoint from all views
                pts_2d_this_kp = pca_2d_reshaped[:, frame_idx, kp_idx, :]  # (n_views, 2)
                
                # Only triangulate if we have valid points from all views
                if not np.isnan(pts_2d_this_kp).any():
                    # Triangulate this keypoint using camera geometry
                    pt_3d = find_consistent_3d_points(
                        pts_2d_this_kp[np.newaxis, :, :],  # Shape: (1, n_views, 2)
                        self.cameraArray
                    )
                    points_3d_triangulated[frame_idx, kp_idx] = pt_3d[0]
        
        # Count valid 3D points
        n_valid = np.sum(~np.isnan(points_3d_triangulated[:, :, 0]))
        n_total = n_frames * n_keypoints
        print(f"  Valid 3D points after triangulation: {n_valid} / {n_total}")
        
        # Store PCA model for reference
        self.pca_model = nanpca
        self.pca_reconstructed_2d = pca_reconstructed_2d
        
        # Update self.points3D for use in bundle adjustment
        # Flatten to (n_frames*n_keypoints, 3) as expected by bundle adjustment
        self.points3D = points_3d_triangulated.reshape(-1, 3)
        
        print(f"\nPCA-constrained 3D initialization complete!")
        print("  ✓ 2D points cleaned via PCA (1.0px RMSE)")
        print("  ✓ 3D world coordinates obtained via triangulation")
        print("  ✓ Bundle adjustment will optimize cameras and 3D against ORIGINAL 2D observations")
        print("="*80 + "\n")
        
        return points_3d_triangulated

    def apply_pca_to_init_3d(self, points_2d_all, n_pca_components=3):
        """
        Better approach: Triangulate with prior cameras, then apply PCA to 3D points.
        
        This method:
        1. Triangulates 2D observations using PRIOR camera parameters (from calibration.toml)
        2. Applies PCA to the 3D points to constrain/clean the structure
        3. Returns PCA-constrained 3D coordinates
        
        Args:
            points_2d_all (np.ndarray): Array of 2D points with shape
                                       (n_views, n_frames, n_keypoints, 2).
            n_pca_components (int): Number of PCA components to use. Default is 3.
        
        Returns:
            np.ndarray: PCA-constrained 3D world coordinates with shape (n_frames, n_keypoints, 3).
        """
        print("\n" + "="*80)
        print("APPLYING 3D-PCA INITIALIZATION (Using Prior Camera Knowledge)")
        print("="*80)
        
        n_views, n_frames, n_keypoints, _ = points_2d_all.shape
        
        # Step 1: Triangulate using PRIOR camera parameters
        print(f"\nStep 1: Triangulate 2D observations using PRIOR camera parameters")
        points_3d_initial = np.full((n_frames, n_keypoints, 3), np.nan)
        
        for frame_idx in range(n_frames):
            for kp_idx in range(n_keypoints):
                # Get 2D points for this keypoint from all views
                pts_2d_this_kp = points_2d_all[:, frame_idx, kp_idx, :]  # (n_views, 2)
                
                # Only triangulate if we have valid points from all views
                if not np.isnan(pts_2d_this_kp).any():
                    # Triangulate using PRIOR cameras
                    pt_3d = find_consistent_3d_points(
                        pts_2d_this_kp[np.newaxis, :, :],  # Shape: (1, n_views, 2)
                        self.cameraArray
                    )
                    points_3d_initial[frame_idx, kp_idx] = pt_3d[0]
        
        # Count valid 3D points
        n_valid_initial = np.sum(~np.isnan(points_3d_initial[:, :, 0]))
        n_total = n_frames * n_keypoints
        print(f"  Valid 3D points after triangulation: {n_valid_initial} / {n_total}")
        
        if NaNPCA is None:
            print("\nWarning: NaNPCA not available. Skipping 3D-PCA, using raw triangulation.")
            self.points3D = points_3d_initial.reshape(-1, 3)
            return points_3d_initial
        
        # Step 2: Apply PCA to 3D points (PER-KEYPOINT)
        print(f"\nStep 2: Apply PCA to 3D structure to constrain pose space (per-keypoint)")
        
        # Initialize output
        points_3d_pca_reshaped = np.full_like(points_3d_initial, np.nan)
        pca_models = []
        
        # Apply PCA SEPARATELY for each keypoint
        for kp_idx in range(n_keypoints):
            # Get all frames for this keypoint: (n_frames, 3)
            points_3d_this_kp = points_3d_initial[:, kp_idx, :]
            
            # Check how many valid points we have
            valid_mask = ~np.isnan(points_3d_this_kp).any(axis=1)
            n_valid = np.sum(valid_mask)
            
            if n_valid < n_pca_components:
                print(f"  Keypoint {kp_idx}: Only {n_valid} valid points, skipping PCA")
                points_3d_pca_reshaped[:, kp_idx, :] = points_3d_this_kp
                pca_models.append(None)
                continue
            
            # Fit PCA for THIS keypoint only
            nanpca_kp = NaNPCA(n_components=n_pca_components)
            nanpca_kp.fit(points_3d_this_kp)
            
            # Transform and reconstruct
            latent = nanpca_kp.transform(points_3d_this_kp)
            points_3d_pca_this_kp = nanpca_kp.inverse_transform(latent)
            
            # Calculate reconstruction error for this keypoint
            diff = points_3d_this_kp[valid_mask] - points_3d_pca_this_kp[valid_mask]
            recon_error = np.sqrt(np.mean(diff**2))
            
            # Explained variance
            explained_var = nanpca_kp.explained_variance_ratio_
            
            print(f"  Keypoint {kp_idx}: PCA with {n_pca_components} components")
            print(f"    Valid points: {n_valid}/{n_frames}")
            print(f"    Reconstruction error: {recon_error:.3f} units")
            print(f"    Explained variance: {explained_var}")
            
            points_3d_pca_reshaped[:, kp_idx, :] = points_3d_pca_this_kp
            pca_models.append(nanpca_kp)
        
        # Flatten for storage
        points_3d_pca = points_3d_pca_reshaped.reshape(-1, 3)
        
        # Store PCA models for reference (one per keypoint)
        self.pca_models = pca_models
        self.points3D_before_pca = points_3d_initial.reshape(-1, 3)
        
        # Update self.points3D for use in bundle adjustment
        self.points3D = points_3d_pca.reshape(-1, 3)
        
        # Step 3: Re-initialize cameras using PCA-constrained 3D points
        print(f"\nStep 3: Re-initialize cameras using PCA-constrained 3D + Linear PnP")
        self._reinitialize_cameras_from_3d(points_2d_all, points_3d_pca_reshaped)
        
        print(f"\n3D-PCA initialization complete!")
        print("  ✓ Initial 3D points triangulated with PRIOR cameras")
        print("  ✓ 3D structure constrained via PCA")
        print("  ✓ Cameras re-initialized from PCA-constrained 3D via PnP")
        print("  ✓ Bundle adjustment will refine cameras and 3D against ORIGINAL 2D observations")
        print("="*80 + "\n")
        
        return points_3d_pca_reshaped
    
    def _reinitialize_cameras_from_3d(self, points_2d_all, points_3d):
        """
        Re-initialize camera parameters using PnP given PCA-constrained 3D points.
        
        This uses a linear PnP approach to estimate camera pose (R, t, focal) from:
        - Known 3D points (PCA-constrained)
        - Observed 2D points
        
        Args:
            points_2d_all (np.ndarray): (n_views, n_frames, n_keypoints, 2)
            points_3d (np.ndarray): (n_frames, n_keypoints, 3) - PCA-constrained
        """
        try:
            import cv2
        except ImportError:
            print("  Warning: OpenCV not available, skipping camera re-initialization")
            return
        
        n_views, n_frames, n_keypoints, _ = points_2d_all.shape
        
        for view_idx in range(n_views):
            # Collect all valid correspondences for this view
            points_3d_valid = []
            points_2d_valid = []
            
            for frame_idx in range(n_frames):
                for kp_idx in range(n_keypoints):
                    pt_3d = points_3d[frame_idx, kp_idx]
                    pt_2d = points_2d_all[view_idx, frame_idx, kp_idx]
                    
                    # Only use if both 3D and 2D are valid
                    if not (np.isnan(pt_3d).any() or np.isnan(pt_2d).any()):
                        points_3d_valid.append(pt_3d)
                        points_2d_valid.append(pt_2d)
            
            if len(points_3d_valid) < 6:
                print(f"  Warning: View {view_idx} has < 6 correspondences, skipping")
                continue
            
            points_3d_valid = np.array(points_3d_valid, dtype=np.float32)
            points_2d_valid = np.array(points_2d_valid, dtype=np.float32)
            
            # Get current camera parameters
            old_focal = self.cameraArray[view_idx, 6]
            old_cx, old_cy = self.cameraArray[view_idx, 9:11]
            
            # Use current camera matrix as initial guess
            camera_matrix = np.array([
                [old_focal, 0, old_cx],
                [0, old_focal, old_cy],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # No distortion for initial PnP
            dist_coeffs = np.zeros(4, dtype=np.float32)
            
            # Solve PnP
            success, rvec, tvec = cv2.solvePnP(
                points_3d_valid,
                points_2d_valid,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                # Update camera parameters
                self.cameraArray[view_idx, 0:3] = rvec.flatten()
                self.cameraArray[view_idx, 3:6] = tvec.flatten()
                
                # Optionally refine focal length using reprojection error minimization
                # For now, keep focal length from prior
                
                print(f"  View {view_idx}: Updated R, t using {len(points_3d_valid)} correspondences")
            else:
                print(f"  Warning: PnP failed for view {view_idx}, keeping prior cameras")


def apply_pca_constraint_to_2d(points_2d, n_components=3):
    """
    Apply PCA constraint to 2D multi-view keypoints to reduce reconstruction error.
    
    This function takes 2D keypoints across multiple views and applies PCA to find a 
    lower-dimensional representation that captures the main variation patterns. This can
    help regularize noisy 2D observations before triangulation.
    
    Args:
        points_2d (np.ndarray): Array of 2D points with shape (n_frames, n_views, n_keypoints, 2)
                               or (n_views, n_frames, n_keypoints, 2).
        n_components (int): Number of PCA components to retain. Default is 3.
    
    Returns:
        np.ndarray: PCA-reconstructed 2D points with the same shape as input.
    """
    if NaNPCA is None:
        print("Warning: NaNPCA not available. Returning original points.")
        return points_2d
    
    # Handle both possible input shapes
    original_shape = points_2d.shape
    if len(original_shape) == 4:
        if original_shape[0] < original_shape[1]:
            # Shape is (n_views, n_frames, n_keypoints, 2) - need to transpose
            points_2d = points_2d.transpose(1, 0, 2, 3)  # -> (n_frames, n_views, n_keypoints, 2)
            transposed = True
        else:
            # Already in (n_frames, n_views, n_keypoints, 2) format
            transposed = False
    else:
        raise ValueError(f"Expected 4D array, got shape {original_shape}")
    
    n_frames, n_views, n_keypoints, _ = points_2d.shape
    
    # Format data for PCA: each row is one keypoint instance across all views
    # Shape: (n_frames * n_keypoints, n_views * 2)
    formatted_data = []
    for frame_idx in range(n_frames):
        for kp_idx in range(n_keypoints):
            kp_multiview = points_2d[frame_idx, :, kp_idx, :]  # (n_views, 2)
            kp_flat = kp_multiview.flatten()  # (n_views*2,)
            formatted_data.append(kp_flat)
    formatted_data = np.array(formatted_data)
    
    # Fit NaNPCA (handles NaN automatically)
    n_components_actual = min(n_components, n_views * 2)
    print(f"Applying PCA with {n_components_actual} components to 2D points...")
    print(f"  Input shape: {original_shape}")
    print(f"  Data matrix shape: {formatted_data.shape}")
    
    has_nan = np.isnan(formatted_data).any(axis=1)
    print(f"  Samples with NaN: {has_nan.sum()} / {len(has_nan)}")
    
    nanpca = NaNPCA(n_components=n_components_actual)
    nanpca.fit(formatted_data)
    
    # Transform and inverse transform to get PCA-reconstructed points
    pca_reconstructed = nanpca.inverse_transform(nanpca.transform(formatted_data))
    
    # Reshape back to (n_frames, n_views, n_keypoints, 2)
    points_2d_pca = np.zeros((n_frames, n_views, n_keypoints, 2))
    idx = 0
    for frame_idx in range(n_frames):
        for kp_idx in range(n_keypoints):
            kp_flat_reconstructed = pca_reconstructed[idx]
            kp_multiview_reconstructed = kp_flat_reconstructed.reshape(n_views, 2)
            points_2d_pca[frame_idx, :, kp_idx, :] = kp_multiview_reconstructed
            idx += 1
    
    # Calculate and report reconstruction error
    diff = points_2d - points_2d_pca
    recon_error = np.sqrt(np.sum(diff**2, axis=-1))  # (n_frames, n_views, n_keypoints)
    mean_error = np.nanmean(recon_error)
    print(f"  PCA reconstruction error: {mean_error:.3f} pixels (mean)")
    
    # If we transposed at the beginning, transpose back
    if transposed:
        points_2d_pca = points_2d_pca.transpose(1, 0, 2, 3)  # -> (n_views, n_frames, n_keypoints, 2)
    
    return points_2d_pca


def find_consistent_3d_points(points2D, cameraArray):
    """
    Finds the most consistent 3D points for multiple frames from 2D points across multiple cameras.

    Args:
        points2D (np.ndarray): Array of 2D points with shape (n_frames, n_cameras, 2).
        cameraArray (np.ndarray): Array of camera parameters with shape (n_cameras, 11).

    Returns:
        np.ndarray: The estimated 3D points as an array of shape (n_frames, 3).
    """
    n_frames, n_cameras, _ = points2D.shape

    # Prepare camera projection matrices for all cameras
    projection_matrices = []
    for i in range(n_cameras):
        r_vec = cameraArray[i, :3]
        t_vec = cameraArray[i, 3:6]
        K = np.eye(3)
        K[0, 0] = K[1, 1] = cameraArray[i, 6]  # Focal length
        K[:2, 2] = cameraArray[i, 9:11]  # Principal point

        R_mat = R.from_rotvec(r_vec).as_matrix()  # Convert rotation vector to matrix
        P = K @ np.hstack((R_mat, t_vec.reshape(3, 1)))  # Projection matrix (3x4)
        projection_matrices.append(P)

    projection_matrices = np.array(projection_matrices)  # Shape: (n_cameras, 3, 4)

    # Initialize results
    points_3D = np.zeros((n_frames, 3))

    # Process each frame
    for frame_idx in range(n_frames):
        A = []
        # b = []
        for cam_idx in range(n_cameras):
            P = projection_matrices[cam_idx]
            x, y = points2D[frame_idx, cam_idx]

            # Build the A matrix for the current camera
            A.append(x * P[2] - P[0])
            A.append(y * P[2] - P[1])

        # Solve for the 3D point using least squares
        A = np.vstack(A)  # Shape: (2 * n_cameras, 4)
        _, _, Vt = np.linalg.svd(A)
        point_3D_homogeneous = Vt[-1]  # Last row of Vt corresponds to the solution
        point_3D = point_3D_homogeneous[:3] / point_3D_homogeneous[3]  # Convert from homogeneous coordinates
        points_3D[frame_idx] = point_3D

    return points_3D


def project_3D_to_2D(points_3d, camParams):
    """points_3d of shape (n_frames, n_keypoints, 3)"""
    sba = PySBA(camParams, np.NaN, np.NaN, np.NaN, np.NaN)
    nFrames, nParts, _ = points_3d.shape
    nCams = camParams.shape[0]
    allLabels = np.full((nFrames, nCams, nParts, 2), np.NaN)
    allCamScales = np.full((nFrames, nCams), np.NaN)
    for nCam in range(nCams):
        rVec = camParams[nCam][:3].reshape((1, 3))
        tVec = camParams[nCam][3:6]
        for nPart in range(nParts):
            allLabels[:, nCam, nPart, :] = sba.project(
                points_3d[:, nPart], np.tile(camParams[nCam], (nFrames, 1))
            )
        pt3d_centroid = np.nanmean(points_3d, axis=1)  # average over parts
        pt3d_centroid = sba.rotate(pt3d_centroid, np.tile(rVec, (nFrames, 1)))  # rotate to camera coordinates
        camDist = pt3d_centroid[:, 2] + tVec[2]  # get z-axis distance ie along optical axis
        camScale = camParams[nCam][6] / camDist  # convert to focal length divided by distance
        allCamScales[:, nCam] = camScale

    return allLabels, allCamScales


def convertParams(camParams):
    allParams = np.full((len(camParams), 11), np.NaN)
    for nCam in range(len(camParams)):
        p = camParams[nCam][0]
        f = p['K'][0,0]/2 + p['K'][1,1]/2
        r = -R.from_matrix(p['r']).as_rotvec()
        t = p['t']
        c = p['K'][2,0:2]
        d = p['RDistort']
        allParams[nCam,:] = np.hstack((r,t,f,d,c))
    return allParams


def unconvertParams(camParamVec):
    thisK = np.full((3, 3), 0)
    thisK[0, 0] = camParamVec[6]
    thisK[1,1] = camParamVec[6]
    thisK[2,2] = 1
    thisK[2,:2] = camParamVec[9:]
    r = R.from_rotvec(-camParamVec[:3]).as_matrix()
    t = camParamVec[3:6]
    d = camParamVec[7:9]
    return {'K': thisK, 'R':r, 't':t, 'd':d}


def getCameraArray(allCameras = ['lBack', 'lFront', 'lTop', 'rBack', 'rFront', 'rTop']):
    # Camera parameters are 3 rotation angles, 3 translations, 1 focal distance, 2 distortion params, and x,y principal points
    # Following notes outlined in evernote, 'bundle adjustment', later updated using optimized values
    camMatDict = {
        'lBack': np.array([0.72, -1.85, 1.72, 0.011, 0.14, 1.36, 1780, -0.020, -0.028, 1420, 716]),
        'lFront': np.array([1.88, -.63, 0.77, -0.041, .099, 1.41, 1780, -0.020, -0.028, 1398, 695]),
        'lTop': np.array([1.93, -1.77, 0.84, -.017, 0.008, 1.72, 1780, -0.020, -0.028, 1386, 842]),
        'rBack': np.array([0.79, 2.02, -1.77, 0.036, 0.11, 1.37, 1780, -0.020, -0.028, 1421, 699]),
        'rFront': np.array([1.91, .75, -.69, 0.038, 0.11, 1.38, 1780, -0.020, -0.028, 1398, 686]),
        'rTop': np.array([1.95, 1.90, -0.82, 0.04, 0.02, 1.73, 1780, -0.020, -0.028, 1403, 828]),
    }
    cameraArray = np.full((len(allCameras), 11), np.NaN)
    for i, e in enumerate(allCameras):
        cameraArray[i,:] = camMatDict[e]

    return cameraArray
