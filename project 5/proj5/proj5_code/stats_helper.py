import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image
# from sklearn.preprocessing import StandardScaler


def compute_mean_and_std(dir_name: str) -> Tuple[float, float]:
    """
    Computes the mean and the standard deviation of all images present within
    the directory.

    Note: convert the image in grayscale and then in [0,1] before computing the
    mean and standard deviation

    Mean = (1/n) * \sum_{i=1}^n x_i
    Variance = (1 / (n-1)) * \sum_{i=1}^n (x_i - mean)^2
    Standard Deviation = 1 / Variance

    Args:
        dir_name: the path of the root dir

    Returns:
        mean: mean value of the gray color channel for the dataset
        std: standard deviation of the gray color channel for the dataset
    """
    mean = None
    std = None
    ############################################################################
    # Student code begin
    ############################################################################

    files = glob.glob(dir_name + '/*/*/*.jpg')
    print(dir_name)

    sum = []
    for image_path in files: 
        img = Image.open(image_path).convert('L')
        arr = np.array(img) / 255
        arr = arr.reshape(-1, 1)#.astype(np.float64)
        sum.extend(list(arr))

    mean = np.mean(sum)
    std = np.std(sum)
    print(mean)
    print(std)

    # sse = None
    # for image_path in files: 
    #     img = Image.open(image_path).convert('L')
    #     arr = np.array(img) / 255
    #     arr = arr.reshape(-1, 1).astype(np.float64)

    #     if sse is None: 
    #         sse = np.zeros(arr.shape).astype(np.float64)
    #     sse += (arr - mean) * (arr - mean)
    # std = sse / (len(files) - 1)

    ############################################################################
    # Student code end
    ############################################################################
    return mean, std
