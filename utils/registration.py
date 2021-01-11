# -*- coding: utf-8 -*- 
# @Time : 2020/12/7 4:16 下午 
# @Author : yl

import os.path as osp
import cv2
import numpy as np


def distances_of_sides(points, shape):
    """
    Calculate the distance of point from four sides of image.
    Args:
        points (ndarrays): key-points, with shape (num_points, 2).
        shape (list): image shape (h, w)

    Returns:

    """
    h, w = shape
    left = points[:, 0]
    top = points[:, 1]
    right = w - left
    bottom = h - top
    return left, top, right, bottom


def calculate_crop_coordinates(pt_sides, ratio, img1_side, img2_side):
    """
    Calculate the crop coordinates of two images, by given a pair of keypoints on two images, and the ratio
    of img1 over img2.
    Then, calculate the distance between keypoint and side on one image, and related distance on another image.
    Compare the distance, if one's related distance over-passed the distance of side, then this image cover more
    region, we need to crop region.

    Args:
        pt_sides:
        ratio:
        img1_side:
        img2_side:

    Returns:

    """
    img1_min, img2_min = 0, 0
    img1_max, img2_max = img1_side, img2_side
    pt_img1, pt_img2 = pt_sides
    if pt_img1 < pt_img2:
        img2_min = pt_img2 - pt_img1 / ratio
        if img2_min < 0:
            img1_min = pt_img1 - pt_img2 * ratio
        img1_max = (img2_side - pt_img2) * ratio + pt_img1
        if img1_max > img1_side:
            img2_max = (img1_side - pt_img1) / ratio + pt_img2
    else:
        img1_min = pt_img1 - pt_img2 * ratio
        if img1_min < 0:
            img2_min = pt_img2 - pt_img1 / ratio
        img2_max = (img1_side - pt_img1) / ratio + pt_img2
        if img2_max > img2_side:
            img1_max = (img2_side - pt_img2) * ratio + pt_img1

    img1_min, img1_max = max(0, int(img1_min)), min(img1_side, int(img1_max))
    img2_min, img2_max = max(0, int(img2_min)), min(img2_side, int(img2_max))
    return img1_min, img1_max, img2_min, img2_max


def registration(img1,
                 img2,
                 matches,
                 keypoints1,
                 keypoints2):
    """
    Perform registration process.
    Given image1 and image2, crop the images to make both covering the same region as much as possible.
    1.

    Args:
        img1 (ndarray): image type, ndarray of cv2 or numpy.
        img2 (ndarray): image type, ndarray of cv2 or numpy.
        matches (list): Match relations in each descriptor between 2 images. Each one contains
        trainidx, queryIdx, distance, et al.
        keypoints1 (list): List of keypoints. Each keypoint contains coordinate, id, et al.
        keypoints2 (list): List of keypoints. Each keypoint contains coordinate, id, et al.
        keep_same (bool): Whether resize two images into the same size.
        out_root (None| str): The output path of images.
    Returns:

    """
    img1h, img1w = img1.shape
    img2h, img2w = img2.shape

    # get coordinates of matched points in two images.
    img1_points = np.float32([keypoints1[matches[i].queryIdx].pt for i in range(len(matches))])
    img2_points = np.float32([keypoints2[matches[i].trainIdx].pt for i in range(len(matches))])

    # calculate the distance to the four sides of matched point in each image.
    img1_left, img1_top, img1_right, img1_bottom = distances_of_sides(img1_points, img1.shape)
    img2_left, img2_top, img2_right, img2_bottom = distances_of_sides(img2_points, img2.shape)

    # shift left the array. [1,2,3] -> [2,3,1] and calculate the distance [1-2,2-3,1-3]
    img1_points_leftshift = np.stack(img1_points[1:, :], img1_points[0, :])
    img2_points_leftshift = np.stack(img2_points[1:, :], img1_points[0, :])
    img1_move_x = np.abs(img1_points[:, 0] - img1_points_leftshift[:, 0])
    img1_move_y = np.abs(img1_points[:, 1] - img1_points_leftshift[:, 1])
    img2_move_x = np.abs(img2_points[:, 0] - img2_points_leftshift[:, 0])
    img2_move_y = np.abs(img2_points[:, 1] - img2_points_leftshift[:, 1])

    # calculate the move ratio between two matched images. According to the similar triangle theorem, we can calculate
    # the actual distance a pixel move at image 2 from image 1.
    dx_img1_2 = img1_move_x / img2_move_x
    dy_img1_2 = img1_move_y / img2_move_y

    pt_left = (img1_left[0], img2_left[0])
    pt_top = (img1_top[0], img2_top[0])

    img1_xmin, img1_xmax, img2_xmin, img2_xmax = calculate_crop_coordinates(pt_left, dx_img1_2, img1w, img2w)
    img1_ymin, img1_ymax, img2_ymin, img2_ymax = calculate_crop_coordinates(pt_top, dy_img1_2, img1h, img2h)

    return (img1_xmin, img1_ymin, img1_xmax, img1_ymax), (img2_xmin, img2_ymin, img2_xmax, img2_ymax)
