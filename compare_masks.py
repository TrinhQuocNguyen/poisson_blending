#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os

def normalize_white_black(img):
    image_mask = np.copy(img)
    # image_mask /= image_mask.max()/255.0

    image_mask[image_mask <= 128] = 0
    image_mask[image_mask > 128] = 255
    return image_mask

def get_standardized(mask_standard, bb_mask):
    # image_mask = np.copy(mask_standard)
    mask_standard[bb_mask > 128] = 0
    return mask_standard

def compare(mask_standard, bb_mask):
    print("come here")
    mask_standard = normalize_white_black(mask_standard)
    cv2.imshow("mask_standard", mask_standard)

    bb_mask = normalize_white_black(bb_mask)
    cv2.imshow("bb_mask", bb_mask)

    fixed = mask_standard.copy()
    fixed[bb_mask > 128] = 0

    print("is equal: ", np.array_equal(fixed, mask_standard))
    cv2.imshow("fixed", fixed)
    cv2.imwrite("./data/compare_masks/fixed.png", fixed)

    cv2.waitKey(0)


def draw_masks(mask_bb, p1 = (100, 100), p2 = (200, 200)):
    cv2.rectangle(mask_bb, p1, p2, (255, 255, 255), -1) # draw white region
    cv2.imshow("img", mask_bb)
    cv2.waitKey()


if __name__ == '__main__':
    print("compare masks")
    mask_standard = cv2.imread("./data/compare_masks/mask_detect_04.png", 0)
    mask_standard = cv2.resize(mask_standard, (512,512))
    bb_mask = cv2.imread("./data/compare_masks/Picture2.png", 0)
    bb_mask = cv2.resize(bb_mask, (512,512))

    compare(mask_standard, bb_mask)
    # mask_bb = np.zeros((512, 512), np.uint8)
    # draw_masks(mask_bb)