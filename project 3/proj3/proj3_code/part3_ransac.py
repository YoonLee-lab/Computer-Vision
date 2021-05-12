import numpy as np
import math

from proj3_code.part2_fundamental_matrix import estimate_fundamental_matrix


def calculate_num_ransac_iterations(
    prob_success: float, sample_size: int, ind_prob_correct: float) -> int:
    """
    Calculates the number of RANSAC iterations needed for a given guarantee of
    success.

    Args:
        prob_success: float representing the desired guarantee of success
        sample_size: int the number of samples included in each RANSAC
            iteration
        ind_prob_success: float representing the probability that each element
            in a sample is correct

    Returns:
        num_samples: int the number of RANSAC iterations needed

    """
    num_samples = None
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    base = 1 - ind_prob_correct ** sample_size
    num_samples = math.ceil(math.log(1 - prob_success, base))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return int(num_samples)


def ransac_fundamental_matrix(
    matches_a: np.ndarray, matches_b: np.ndarray) -> np.ndarray:
    """
    For this section, use RANSAC to find the best fundamental matrix by
    randomly sampling interest points. You would reuse
    estimate_fundamental_matrix() from part 2 of this assignment and
    calculate_num_ransac_iterations().

    If you are trying to produce an uncluttered visualization of epipolar
    lines, you may want to return no more than 30 points for either left or
    right images.

    Tips:
        0. You will need to determine your prob_success, sample_size, and
            ind_prob_success values. What is an acceptable rate of success? How
            many points do you want to sample? What is your estimate of the
            correspondence accuracy in your dataset?
        1. A potentially useful function is numpy.random.choice for creating
            your random samples.
        2. You will also need to choose an error threshold to separate your
            inliers from your outliers. We suggest a threshold of 0.1.

    Args:
        matches_a: A numpy array of shape (N, 2) representing the coordinates
            of possibly matching points from image A
        matches_b: A numpy array of shape (N, 2) representing the coordinates
            of possibly matching points from image B
    Each row is a correspondence (e.g. row 42 of matches_a is a point that
    corresponds to row 42 of matches_b)

    Returns:
        best_F: A numpy array of shape (3, 3) representing the best fundamental
            matrix estimation
        inliers_a: A numpy array of shape (M, 2) representing the subset of
            corresponding points from image A that are inliers with respect to
            best_F
        inliers_b: A numpy array of shape (M, 2) representing the subset of
            corresponding points from image B that are inliers with respect to
            best_F
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    np.random.seed(2)
    
    N = len(matches_a)
    matches_a_1s = np.hstack([matches_a, np.ones([N, 1])])
    matches_b_1s = np.hstack([matches_b, np.ones([N, 1])])
    
    best_count = 0
    max_iter = calculate_num_ransac_iterations(0.99, 9, 0.5)
    print(max_iter)
    threshold = 0.1 #0.05
    
    i = 0
    while i < min(200 * N, max_iter): 
        i += 1
        random_sample = np.random.choice(N, 8)
        points_a = matches_a[random_sample]
        points_b = matches_b[random_sample]
        
        F = estimate_fundamental_matrix(points_a, points_b)
        
        inliners = []
        for j in range(len(matches_a)):
            a,b,c = matches_b_1s[j].dot(F)
            x,y,z = matches_a_1s[j]
            score = (a * x + b * y + c) / math.sqrt(a**2 + b**2)

            if np.abs(score)<threshold: 
                inliners.append(j)
        inliners = np.array(inliners)
        
        if len(inliners) > best_count:
            best_count = len(inliners)
            best_F, inliers_a, inliers_b = F, matches_a[inliners], matches_b[inliners]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return best_F, inliers_a, inliers_b
