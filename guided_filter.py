# Author: Yahui Liu <yahui.liu@unitn.it>

import os
import glob
import cv2
from cv2.ximgproc import guidedFilter
import numpy as np

import super_hero

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='deepcrack')
parser.add_argument('--results_dir', type=str, default='D:/tmp/GOPR1031_new_512_blended_02/170/results/deepcrack/test_latest/img')
parser.add_argument('--thresh', type=float, default=0.31, help='using the best threshold')
parser.add_argument('--epsilon', type=float, default=0.065, help='eps = 1e-6*255*255')

parser.add_argument('--radius', type=int, default=5)
parser.add_argument('--suffix_image', type=str, default='image.png', help='Suffix of predicted file name')
parser.add_argument('--suffix_gf', type=str, default='gf.png', help='Suffix of side output file name')
parser.add_argument('--suffix_output', type=str, default='out', help='Suffix of refined results')
args = parser.parse_args()


if __name__ == '__main__':
    results_dir = args.results_dir
    image_list = glob.glob(os.path.join(results_dir, '*{}.png'.format("image")))
    image_list.sort()
    image_list = [ll.replace("\\", "/") for ll in image_list]

    gf_list = [ll.replace(args.suffix_image, args.suffix_gf) for ll in image_list]
    gf_list = [ll.replace("\\", "/") for ll in gf_list]
    print("image_list: ", image_list)
    print("gf_list: ", gf_list)

    for ii, gf in zip(image_list, gf_list):
        overlay = ii.replace("image.png", "overlay.png")
        print(overlay)
        ori_img = cv2.imread(ii)
        cv2.imshow("ori_img", ori_img)

        mask = cv2.imread(gf)
        red_img = super_hero.overlay_red_color_using_mask(ori_img, mask, 0.2, 1., 0)
        cv2.imshow("overlay", ori_img)
        cv2.imshow("mask", mask)
        cv2.imshow("red_img", red_img)
        cv2.waitKey(0)
        # cv2.imwrite(overlay, ori_img)
