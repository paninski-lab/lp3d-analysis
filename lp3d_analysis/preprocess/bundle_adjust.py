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

#%%
class PySBA:
    """Python class for Simple Bundle Adjustment"""

    def __init__(self, cameraArray, points3D, points2D, cameraIndices, point2DIndices, pointWeights=None):
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
        r = 1 + k1 * n + k2 * n ** 2
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
        res = least_squares(self.fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-6, method='trf', jac='3-point',
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
        f0 = self.fun(x0, numCameras, numPoints, self.cameraIndices, self.point2DIndices, self.points2D, np.full_like(self.point2DIndices, 1))
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
        pt3d_centroid = np.mean(points_3d, axis=1)  # average over parts
        pt3d_centroid = sba.rotate(pt3d_centroid, np.tile(rVec, (nFrames, 1)))  # rotate to camera coordinates
        camDist = pt3d_centroid[:, 2] + tVec[2]  # get z-axis distance ie along optical axis
        camScale = camParams[nCam][6] / camDist  # convert to focal length divided by distance
        allCamScales[:, nCam] = camScale

    return allLabels, allCamScales

#%%
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
