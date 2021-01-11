# -*- coding: utf-8 -*- 
# @Time : 2020/12/8 2:13 下午 
# @Author : yl


import cv2
import os.path as osp
import numpy as np
from utils.features import feature_match, feature_filter, feature_sample
from utils.registration import registration
import torch.nn as nn
import torch


class BaseDetector(object):
    def __init__(self,
                 feature_type='sift',
                 det_type='pool',
                 kernel_size=20):
        self.feature_type = feature_type
        self.det_type = det_type
        if det_type == 'pool':
            self.avg_pool = nn.AvgPool2d(kernel_size, stride=0, padding=0)

    def get_keypoints(self, img1, img2, mask1=None, mask2=None):
        matches, keypoints1, keypoints2 = feature_match(img1, img2, self.feature_type)

        matches, keypoints1, keypoints2 = feature_filter(img1, img2, matches, keypoints1, keypoints2)

        matches, keypoints1, keypoints2 = feature_sample(img1, img2, matches, keypoints1, keypoints2, sample_num=3)

        return matches, keypoints1, keypoints2

    def registration(self,
                     img1,
                     img2,
                     keep_origin=True,
                     keep_same=True,
                     out_root=None,
                     postfix=0):
        assert len(img1.shape) == len(img2.shape)
        if len(img1.shape) == 3:
            img1_crop, img2_crop = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            img1_crop, img2_crop = img1.copy(), img2.copy()

        matches, keypoints1, keypoints2 = self.get_keypoints(img1_crop, img2_crop)

        img1h, img1w = img1.shape
        crop1_coord, crop2_coord = registration(img1_crop, img2_crop, matches, keypoints1, keypoints2)
        img1_xmin, img1_ymin, img1_xmax, img1_ymax = crop1_coord
        img2_xmin, img2_ymin, img2_xmax, img2_ymax = crop2_coord

        if keep_origin:
            img1_crop = img1[img1_ymin:img1_ymax, img1_xmin:img1_xmax]
            img2_crop = img2[img2_ymin:img2_ymax, img2_xmin:img2_xmax]
        else:
            img1_crop = img1_crop[img1_ymin:img1_ymax, img1_xmin:img1_xmax]
            img2_crop = img2_crop[img2_ymin:img2_ymax, img2_xmin:img2_xmax]

        if keep_same:
            crop_w = img2_xmax - img2_xmin
            crop_h = img2_ymax - img2_ymin
            img1_crop = cv2.resize(img1_crop, (crop_w, crop_h))

        if out_root:
            cv2.imwrite(osp.join(out_root, f"img1_crop_{postfix}.png"), img1_crop)
            cv2.imwrite(osp.join(out_root, f"img2_crop_{postfix}.png"), img2_crop)
            img_out = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=2)
            cv2.imwrite(osp.join(out_root, f"registration_match_{postfix}.png"), img_out)

        return img1_crop, img2_crop, crop1_coord, crop2_coord

    def change_det(self, img1, img2, debug_root=None, thresh=15, postfix=0):
        # concat together to perform average pool and diff.
        img1h, img1w = img1.shape[:2]

        crop = np.stack([img1, img2])
        tensor = torch.from_numpy(crop).unsqueeze(0).float()
        avg = self.avg_pool(tensor).squeeze().numpy()
        avg_diff = np.abs(np.diff(avg, axis=0)).squeeze().astype(np.uint8)

        # threshold, the value could be debugged.
        _, avg_thresh = cv2.threshold(avg_diff, thresh, 255, cv2.THRESH_BINARY)
        ratio = 255 / np.max(avg_diff)
        avg_diff_denorm = (avg_diff * ratio).astype(np.uint8)

        avg_diff_denorm_resize = cv2.resize(avg_diff_denorm, (img1w, img1h))
        avg_thresh = cv2.resize(avg_thresh, (img1w, img1h))

        # heatmap = cv2.applyColorMap(avg_diff_denorm_resize, cv2.COLORMAP_JET)

        # debug
        if debug_root:
            cv2.imwrite(osp.join(debug_root, f'avg_diff_{postfix}.png'), avg_diff_denorm_resize)
            cv2.imwrite(osp.join(debug_root, f'avg_thresh_{postfix}.png'), avg_thresh)
            # cv2.imwrite(osp.join(debug_root, f'heatmap_{postfix}.png'), heatmap)
        return avg_diff_denorm

    def draw_result(self, image, result, crop_coord, thresh=0, color=(0, 0, 128), type='mask', out=None):
        if type == 'mask':
            result_ids = np.where(result > thresh)
            id_y, id_x = result_ids
            xmin, ymin, xmax, ymax = crop_coord
            id_y += ymin
            id_x += xmin
            shift_ids = [id_y, id_x]
            image[shift_ids] = image[shift_ids] * 0.5 + color * 0.5  # draw mask
            image = image.astype(np.uint8)
        elif type == 'bbox':
            cnt, _ = cv2.findContours(image, mode=cv2.CHAIN_APPROX_SIMPLE, method=cv2.CONTOURS_MATCH_I1)
            xx = cv2.minAreaRect(cnt)
            cv2.drawContours(image, xx, -1, color, thickness=2)

        if out:
            cv2.imwrite("result.png", image)
        else:
            return image

    def forward(self, img1, img2, count=0, thresh=15):
        img1_crop, img2_crop, crop1_coord, crop2_coord = self.registration(img1, img2, keep_origin=True, keep_same=True,
                                                                           out_root='',
                                                                           postfix=count)
        result = self.change_det(img1_crop, img2_crop)

        self.draw_result(img2_crop, result, crop2_coord, thresh=thresh, out='None')

    def __call__(self, img1, img2, count=0, thresh=15):
        return self.forward(img1, img2, count=0, thresh=thresh)
