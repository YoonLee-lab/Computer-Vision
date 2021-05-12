#!/usr/bin/python3

import numpy as np
import torch

from torch import nn
from typing import Tuple


SOBEL_X_KERNEL = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]).astype(np.float32)
SOBEL_Y_KERNEL = np.array(
    [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ]).astype(np.float32)


def compute_image_gradients(image_bw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Use convolution with Sobel filters to compute the image gradient at each
    pixel.

    Args:
        image_bw: A numpy array of shape (M,N) containing the grayscale image

    Returns:
        Ix: Array of shape (M,N) representing partial derivatives of image
            w.r.t. x-direction
        Iy: Array of shape (M,N) representing partial derivative of image
            w.r.t. y-direction
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    image = image_bw

    # kernel = SOBEL_X_KERNEL
    # padded = kernel.shape[1] // 2
    # groups = image.shape[1] // kernel.shape[1]

    # conv2d = nn.Conv2d(1, 1 ,3 , padding = 1, stride = 1)
    # conv2d.weight = torch.nn.Parameter(torch.Tensor(kernel.reshape(1,3,3)))
    # Ix = conv2d(torch.Tensor(image))
    filter = SOBEL_X_KERNEL
    height = filter.shape[0] // 2
    width = filter.shape[1] // 2
    padded = np.pad(image, ((height,),(width,)), 'constant', constant_values = 0)

    Ix = np.zeros((image.shape[0], image.shape[1]))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            kernel = padded[i:i + filter.shape[0], j:j + filter.shape[1]]
            Ix[i, j] = np.sum((np.multiply(kernel, filter)), axis = (0,1))


    filter = SOBEL_Y_KERNEL
    height = filter.shape[0] // 2
    width = filter.shape[1] // 2
    padded = np.pad(image, ((height,),(width,)), 'constant', constant_values = 0)

    Iy = np.zeros((image.shape[0], image.shape[1]))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            kernel = padded[i:i + filter.shape[0], j:j + filter.shape[1]]
            Iy[i, j] = np.sum((np.multiply(kernel, filter)), axis = (0,1))   

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return Ix, Iy


def get_gaussian_kernel_2D_pytorch(ksize: int, sigma: float) -> torch.Tensor:
    """Create a Pytorch Tensor representing a 2d Gaussian kernel

    Args:
        ksize: dimension of square kernel
        sigma: standard deviation of Gaussian

    Returns:
        kernel: Tensor of shape (ksize,ksize) representing 2d Gaussian kernel

    You should be able to reuse your project 1 Code here.
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    r = np.arange(0, ksize)
    kernel = 1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-(r-ksize//2)**2/(2*sigma**2)) 
    kernel = kernel * (1/np.sum(kernel))

    kernel = np.reshape(np.array(kernel),(ksize, 1))
    kernel = torch.Tensor(np.outer(kernel, kernel))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return kernel


def second_moments(
    image_bw: np.ndarray,
    ksize: int = 7,
    sigma: float = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Compute second moments from image.

    Compute image gradients Ix and Iy at each pixel, the mixed derivatives,
    then the second moments (sx2, sxsy, sy2) at each pixel, using convolution
    with a Gaussian filter.

    Args:
        image_bw: array of shape (M,N) containing the grayscale image
        ksize: size of 2d Gaussian filter
        sigma: standard deviation of Gaussian filter

    Returns:
        sx2: array of shape (M,N) containing the second moment in x direction
        sy2: array of shape (M,N) containing the second moment in y direction
        sxsy: array of dim (M,N) containing the second moment in the x then the
            y direction
    """

    sx2, sy2, sxsy = None, None, None
    ###########################################################################
    # TODO: YOUR SECOND MOMENTS CODE HERE                                     #
    ###########################################################################

    Ix, Iy = compute_image_gradients(image_bw)
    Ix = torch.Tensor(Ix)
    Iy = torch.Tensor(Iy)
    Ixx = torch.mul(Ix, Ix)
    Iyy = torch.mul(Iy, Iy)
    Ixy = torch.mul(Ix, Iy)
    image = torch.unsqueeze(torch.stack((Ixx,Iyy,Ixy), 0),1)
    # # print(image.shape)

    # kernel = get_gaussian_kernel_2D_pytorch(ksize, sigma)
    # kernel = torch.unsqueeze(torch.stack((kernel,kernel,kernel)),1)
    # # print(kernel.shape)
    # kernel = nn.Parameter(kernel)

    # conv2d = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=ksize, bias=False, 
    # 	padding=(ksize//2,ksize//2), groups = 1)
    # conv2d.weight = kernel
    # sx2, sy2, sxsy = conv2d(image)
    # print(sx2, sy2, sxsy)

    kernel = get_gaussian_kernel_2D_pytorch(ksize, sigma)
    kernel = torch.unsqueeze(torch.unsqueeze(kernel, 0),0)
    Ixx = torch.unsqueeze(torch.unsqueeze(Ixx, 0),0)
    Iyy = torch.unsqueeze(torch.unsqueeze(Iyy, 0),0)
    Ixy = torch.unsqueeze(torch.unsqueeze(Ixy, 0),0)

    sx2 = torch.squeeze(nn.functional.conv2d(Ixx, kernel, padding=ksize // 2)).detach().numpy()
    sy2 = torch.squeeze(nn.functional.conv2d(Iyy, kernel, padding=ksize // 2)).detach().numpy()
    sxsy = torch.squeeze(nn.functional.conv2d(Ixy, kernel, padding=ksize // 2)).detach().numpy()



    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return sx2, sy2, sxsy


def compute_harris_response_map(
    image_bw: np.ndarray,
    ksize: int = 7,
    sigma: float = 5,
    alpha: float = 0.05
) -> np.ndarray:
    """Compute the Harris cornerness score at each pixel (See Szeliski 7.1.1)

    Recall that R = det(M) - alpha * (trace(M))^2
    where M = [S_xx S_xy;
               S_xy  S_yy],
          S_xx = Gk * I_xx
          S_yy = Gk * I_yy
          S_xy  = Gk * I_xy,
    and * is a convolutional operation over a Gaussian kernel of size (k, k).
    (You can verify that this is equivalent to taking a (Gaussian) weighted sum
    over the window of size (k, k), see how convolutional operation works here:
        http://cs231n.github.io/convolutional-networks/)

    Ix, Iy are simply image derivatives in x and y directions, respectively.
    You may find the Pytorch function nn.Conv2d() helpful here.

    Args:
        image_bw: array of shape (M,N) containing the grayscale image
            ksize: size of 2d Gaussian filter
        sigma: standard deviation of gaussian filter
        alpha: scalar term in Harris response score

    Returns:
        R: array of shape (M,N), indicating the corner score of each pixel.
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    Sxx, Syy, Sxy = second_moments(image_bw, ksize, sigma)
    det = Sxx * Syy -  Sxy * Sxy
    trace = Sxx + Syy
    R = det - alpha * trace * trace

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return R


def maxpool_numpy(R: np.ndarray, ksize: int) -> np.ndarray:
    """ Implement the 2d maxpool operator with (ksize,ksize) kernel size.

    Note: the implementation is identical to my_conv2d_numpy(), except we
    replace the dot product with a max() operator.

    Args:
        R: array of shape (M,N) representing a 2d score/response map

    Returns:
        maxpooled_R: array of shape (M,N) representing the maxpooled 2d
            score/response map
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    height = ksize// 2
    width = ksize // 2
    padded = np.pad(R, ((height,),(width,)), 'constant', constant_values = 0)

    maxpooled_R = np.zeros((R.shape[0], R.shape[1]))

    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            kernel = padded[i:i + ksize, j:j + ksize]
            maxpooled_R[i, j] = np.max(kernel)

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return maxpooled_R


def nms_maxpool_pytorch(
    R: np.ndarray,
    k: int,
    ksize: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Get top k interest points that are local maxima over (ksize,ksize)
    neighborhood.

    HINT: One simple way to do non-maximum suppression is to simply pick a
    local maximum over some window size (u, v). This can be achieved using
    nn.MaxPool2d. Note that this would give us all local maxima even when they
    have a really low score compare to other local maxima. It might be useful
    to threshold out low value score before doing the pooling (torch.median
    might be useful here).

    You will definitely need to understand how nn.MaxPool2d works in order to
    utilize it, see https://pytorch.org/docs/stable/nn.html#maxpool2d

    Threshold globally everything below the median to zero, and then
    MaxPool over a 7x7 kernel. This will fill every entry in the subgrids
    with the maximum nearby value. Binarize the image according to
    locations that are equal to their maximum. Multiply this binary
    image, multiplied with the cornerness response values. We'll be testing
    only 1 image at a time.

    Args:
        R: score response map of shape (M,N)
        k: number of interest points (take top k by confidence)
        ksize: kernel size of max-pooling operator

    Returns:
        x: array of shape (k,) containing x-coordinates of interest points
        y: array of shape (k,) containing y-coordinates of interest points
        c: array of shape (k,) containing confidences of interest points
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    zeros = torch.zeros(R.shape)
    ones = torch.ones(R.shape)

    R = torch.unsqueeze(torch.Tensor(R), 0)
    max_pool = nn.MaxPool2d((ksize,ksize), padding = ksize//2, stride = 1)
    median = torch.median(R)
    
    thresh = torch.where(R < median, zeros, R)
    pooled = max_pool(thresh)

    binarize = torch.where(R == pooled, ones, zeros)
    keep = torch.squeeze(torch.mul(binarize, R))
    confidences, i = torch.topk(keep.flatten(), k)
    x, y = np.array(np.unravel_index(i.numpy(), keep.shape))
    confidences = confidences.detach().numpy()


    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return x, y, confidences


def remove_border_vals(
    img: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray
) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Remove interest points that are too close to a border to allow SIFT feature
    extraction. Make sure you remove all points where a 16x16 window around
    that point cannot be formed.

    Args:
        img: array of shape (M,N) containing the grayscale image
        x: array of shape (k,) representing x coord of interest points
        y: array of shape (k,) representing y coord of interest points
        c: array of shape (k,) representing confidences of interest points

    Returns:
        x: array of shape (p,), where p <= k (less than or equal after pruning)
        y: array of shape (p,)
        c: array of shape (p,)
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    w = img.shape[0]
    h = img.shape[1]

    x,y,c = x[x>=7], y[x>=7], c[x>=7]
    x,y,c = x[x<w-8], y[x<w-8], c[x<w-8]
    x,y,c = x[y>=7], y[y>=7], c[y>=7]
    x,y,c = x[y<h-8], y[y<h-8], c[y<h-8]

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return x, y, c


def get_harris_interest_points(
    image_bw: np.ndarray,
    k: int = 2500
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implement the Harris Corner detector. You will find
    compute_harris_response_map(), nms_maxpool_pytorch(), and
    remove_border_vals() useful. Make sure to sort the interest points in
    order of confidence!

    Args:
        image_bw: array of shape (M,N) containing the grayscale image
        k: maximum number of interest points to retrieve

    Returns:
        x: array of shape (p,) containing x-coordinates of interest points
        y: array of shape (p,) containing y-coordinates of interest points
        c: array of dim (p,) containing the strength(confidence) of each
            interest point where p <= k.
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    R = compute_harris_response_map(image_bw)

    x, y, c = nms_maxpool_pytorch(R, k, 7)
    temp = x
    x = y
    y = temp
    c = c / np.max(c)
    y, x, c = remove_border_vals(image_bw, y, x, c)

    i = np.argsort(c)
    x = x[i]
    y = y[i]
    c = c[i]

    if len(c) > k: 
    	x = x[:k]
    	y = y[:k]
    	c = c[:k]

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return x, y, c
