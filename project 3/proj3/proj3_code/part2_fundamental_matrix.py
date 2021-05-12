"""Fundamental matrix utilities."""

import numpy as np


def normalize_points(points: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Perform coordinate normalization through linear transformations.
    Args:
        points: A numpy array of shape (N, 2) representing the 2D points in
            the image

    Returns:
        points_normalized: A numpy array of shape (N, 2) representing the
            normalized 2D points in the image
        T: transformation matrix representing the product of the scale and
            offset matrices
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    cu, cv = np.mean(points, axis = 0)
    su = 1.0 / np.std((points - np.array([cu, cv]))[:,0])
    sv = 1.0 / np.std((points - np.array([cu, cv]))[:,1])

    scale_matrix = np.array([[su, 0 ,0],[0,sv, 0],[0,0,1]])
    offset_matrix = np.array([[1, 0, -cu],[0, 1, -cv],[0, 0, 1]])
    T = scale_matrix.dot(offset_matrix)

    points_1s = np.ones([len(points),3])
    points_1s[:,0:2] = points

    points_normalized_1s = points_1s.dot(T.T) #(N,3).dot(3,3) = (N,3)
    points_normalized = points_normalized_1s[:,0:2]


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return points_normalized, T


def unnormalize_F(
    F_norm: np.ndarray, T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """
    Adjusts F to account for normalized coordinates by using the transformation
    matrices.

    Args:
        F_norm: A numpy array of shape (3, 3) representing the normalized
            fundamental matrix
        T_a: Transformation matrix for image A
        T_B: Transformation matrix for image B

    Returns:
        F_orig: A numpy array of shape (3, 3) representing the original
            fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    F_orig = T_b.T.dot(F_norm).dot(T_a)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F_orig


def estimate_fundamental_matrix(
    points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    """
    Calculates the fundamental matrix. You may use the normalize_points() and
    unnormalize_F() functions here.

    Args:
        points_a: A numpy array of shape (N, 2) representing the 2D points in
            image A
        points_b: A numpy array of shape (N, 2) representing the 2D points in
            image B

    Returns:
        F: A numpy array of shape (3, 3) representing the fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    A = np.zeros([len(points_a), 8]) 
    norm_a, T_a = normalize_points(points_a)
    norm_b, T_b = normalize_points(points_b)
    
    for i in range(norm_a.shape[0]):
        ua, va = norm_a[i]
        ub, vb = norm_b[i]
        A[i] = [ua * ub, va * ub, ub, ua * vb, va * vb, vb, ua, va]

    temp = -1 * np.ones(A.shape[0])
    F, residuals, rank, s = np.linalg.lstsq(A, temp, rcond = None)
    F = np.append(F,1).reshape(3,3)
    
    U, S, Vh = np.linalg.svd(F)
    S = np.diag(S)
    S[2,2] = 0
    
    F_norm = U.dot(S).dot(Vh)
    
    F = unnormalize_F(F_norm, T_a, T_b)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F
