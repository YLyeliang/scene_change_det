# -*- coding: utf-8 -*- 
# @Time : 2020/12/1 5:16 下午 
# @Author : yl
import cv2
import math
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np


def feature_match(img1,
                  img2,
                  type='sift',
                  mask1=None,
                  mask2=None):
    """
    The function perform below process.
    1. Detect specified features from image1 and image2 by given type
    2. Using BruteForce match method to finds the best match for each descriptor between 2 images.

    Args:
        img1 (ndarray): image type, ndarray of cv2 or numpy.
        img2 (ndarray): image type, ndarray of cv2 or numpy.
        type (str): Feature type used to detect from image.
        mask1 (ndarray): The mask array has same shape with img. 1 means the region need to detect features and 0 not.
        mask2 (ndarray): The mask array has same shape with img. 1 means the region need to detect features and 0 not.

    Returns:
        matches : Match relations in each descriptor between 2 images.
        keypoints1: Detected keypoints in image
        keypoints2: Detected keypoints in image

    """
    assert len(img1.shape) == 2
    img1h, img1w = img1.shape[:2]
    if type == 'sfit':
        feature = cv2.SIFT_create()
    elif type == 'orb':
        feature = cv2.ORB_create()
    else:
        raise NotImplementedError

    # detect the features
    keypoints1, descriptors1 = feature.detectAndCompute(img1, mask1)
    keypoints2, descriptors2 = feature.detectAndCompute(img2, mask2)

    # built the Match method
    bf = cv2.BFMatcher(crossCheck=True)

    matches = bf.match(descriptors1, descriptors2)

    return matches, keypoints1, keypoints2


def feature_filter(img1, img2, matches, keypoints1, keypoints2):
    """
    Filter some key-points that have obvious mismatching.
    This should decided according to specified task and images.

    Args:
        img1 (ndarray): image type, ndarray of cv2 or numpy.
        img2 (ndarray): image type, ndarray of cv2 or numpy.
        matches (list): Match relations in each descriptor between 2 images. Each one contains
        trainidx, queryIdx, distance, et al.
        keypoints1 (list): List of keypoints. Each keypoint contains coordinate, id, et al.
        keypoints2 (list): List of keypoints. Each keypoint contains coordinate, id, et al.

    Returns:

    """
    assert len(img1.shape) == 2
    img1h, img1w = img1.shape
    matches_filter = []
    for match_info in matches:
        pt1 = keypoints1[match_info.queryIdx].pt
        pt2 = keypoints2[match_info.trainIdx].pt

        # concat two images in horizontal way, calculate the angle of two key-points.
        dy = pt2[1] - pt1[1]
        dx = pt2[0] + img1w - pt1[0]
        angle = math.atan(float(dy) / dx)

        # In the uav scene monitoring task, the covering range of two images differ only in small region.
        # Thus, filter those key-points changes too big.
        if math.fabs(pt1[1] - pt2[1]) < img1h / 10 and math.fabs(pt1[0] - pt2[0]) < img1w / 10:
            matches_filter.append(match_info)

    return matches_filter, keypoints1, keypoints2


def feature_sample(img1, img2, matches, keypoints1, keypoints2, sample_num=3):
    """
    Sample the best matched key-points for following registration.
    This sample method process as below:
    1. split the image into blocks bygiven number from top-left to bottom-right.
    2. find the keypoint in each block with minimum matched distance.
    3. preserve the result into the dict and return.

    Args:
        img1 (ndarray): image type, ndarray of cv2 or numpy.
        img2 (ndarray): image type, ndarray of cv2 or numpy.
        matches (list): Match relations in each descriptor between 2 images. Each one contains
        trainidx, queryIdx, distance, et al.
        keypoints1 (list): List of keypoints. Each keypoint contains coordinate, id, et al.
        keypoints2 (list): List of keypoints. Each keypoint contains coordinate, id, et al.

    Returns:

    """
    assert len(img1.shape) == 2
    img1h, img1w = img1.shape

    best_dict = {f'{i}': None for i in range(sample_num)}
    width_range = np.linspace(0, img1w, sample_num + 1)
    height_range = np.linspace(0, img1h, sample_num + 1)
    for m in matches:
        pt = keypoints1[m.queryIdx].pt
        for i in range(sample_num):
            if width_range[i] < pt[0] < width_range[i + 1] and height_range[i] < pt[1] < height_range[i + 1]:
                if best_dict[f'{i}'] is None or best_dict[f'{i}'][1] > m.distance:
                    best_dict[f'{i}'] = [m, m.distance]
                # terminate if the key-point fall in one block, and calculate the next match.
                break
    best_match = [value[0] for key, value in best_dict.items()]

    return best_match, keypoints1, keypoints2
