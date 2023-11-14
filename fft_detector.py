#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse
import PIL.Image
import pyamg
from copy import deepcopy
import cv2
import unsharp_mask
from matplotlib import pyplot as plt
from numpy import ma
import time

import super_hero


def detect(image):
    # pre-processing
    img = np.copy(image)

    # check optimal size
    rows, cols = img.shape
    crow, ccol = int(rows / 4), int(cols / 4)
    # print(rows, cols)

    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow:crow + int(rows / 2), ccol:ccol + int(cols / 2)] = 1

    # operate the dft
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    croped = np.copy(magnitude_spectrum)
    croped[mask == 0.] = 0

    cv2.imshow("img", img)
    cv2.imshow("croped", croped)
    applied1 = super_hero.overlay_black(img, mask*255.)
    cv2.imshow("applied1", applied1)

    cv2.waitKey(0)


def get_sum_high_frequency(img):
    # check optimal size
    rows, cols = img.shape
    crow, ccol = int(rows / 4), int(cols / 4)

    # print(rows, cols)

    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow:crow + int(rows / 2), ccol:ccol + int(cols / 2)] = 1

    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    # dft = abs(dft)
    # dft_shift = np.fft.fftshift(dft)
    # dft *= mask
    # print(cv2.magnitude(dft[:, :, 0], dft[:, :, 1]))
    magnitude_spectrum = 20 * ma.log(cv2.magnitude(dft[:, :, 0], dft[:, :, 1]))
    # magnitude_spectrum = abs(magnitude_spectrum)

    # print(magnitude_spectrum)
    # croped = magnitude_spectrum * mask
    cropped = np.copy(magnitude_spectrum)
    cv2.imshow("mask-small", mask*255)

    cropped[mask == 0.] = 0.

    sum = np.sum(cropped)

    # img_back = cv2.idft(dft)
    # img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    # plt.subplot(221),plt.imshow(img, cmap = 'gray')
    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(222),plt.imshow(magnitude_spectrum, cmap = 'gray')
    # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    # plt.subplot(223),plt.imshow(cropped, cmap = 'gray')
    # plt.title('Cropped'), plt.xticks([]), plt.yticks([])
    # plt.subplot(224),plt.imshow(img_back, cmap = 'gray')
    # plt.title('Inverse DFT'), plt.xticks([]), plt.yticks([])
    # plt.show()

    # print(standard)
    # print("std:", np.std(croped))
    # print("median:", np.median(croped))
    # print("mean:", np.mean(croped))

    # print(sum)
    if np.isneginf(sum):
        img_back = cv2.idft(dft)
        img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
        plt.subplot(141),plt.imshow(img, cmap = 'gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(142),plt.imshow(magnitude_spectrum, cmap = 'gray')
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        plt.subplot(143),plt.imshow(cropped, cmap = 'gray')
        plt.title('Cropped'), plt.xticks([]), plt.yticks([])
        plt.subplot(144),plt.imshow(img_back, cmap = 'gray')
        plt.title('img_back'), plt.xticks([]), plt.yticks([])
        plt.show()
        # print(sum)

    return sum


def separate(image):
    # pre-processing
    img = np.copy(image)
    # check optimal size
    rows, cols = img.shape
    # print(rows, cols)

    # hyper-parameters
    sliding_window_size = 16
    step = 4

    w = rows - sliding_window_size
    h = cols - sliding_window_size

    mask = np.zeros((rows, cols), np.uint8)

    # print("std:", np.std(img))
    # print("median:", np.median(img))
    # print("mean:", np.mean(img))

    MAGIC_NUMBER_OF_STD_DEV = 0.930481433
    standard = np.mean(img) - np.std(img) * MAGIC_NUMBER_OF_STD_DEV

    X = []
    frequency = []
    for i in range(0, int(w / step)):
        for j in range(0, int(h / step)):
            # This will keep all only the crops in the memory.
            temp = img[i * step:i * step + sliding_window_size,
                   j * step:j * step + sliding_window_size].copy()

            X.append(temp)
            frequency.append(get_sum_high_frequency(temp))

            # if get_sum_high_frequency(temp) < 8494.898876953126:
            #     mask[i * step:i * step + sliding_window_size,
            #     j * step:j * step + sliding_window_size] = 1

    # print(len(X))
    print("std:", np.std(frequency))
    print("median:", np.median(frequency))
    print("mean:", np.mean(frequency))
    MAGIC_NUMBER_OF_STD_DEV = 1.9
    standard = np.mean(frequency) - np.std(frequency) * MAGIC_NUMBER_OF_STD_DEV
    print("standard: ", standard)




    # plt.hist(frequency, bins='auto')  # arguments are passed to np.histogram
    # plt.title("Histogram with 'auto' bins")
    # plt.show()

    for f in range(len(frequency)):
        # print(frequency[f])
        if frequency[f] < standard:
            i = int(f/(w / step))
            j = f % int(h / step)

            mask[i * step:i * step + sliding_window_size,
            j * step:j * step + sliding_window_size] = 1


    # for i in range(len(X)):
    #     print(get_sum_high_frequency(X[i]))
    #     cv2.imshow("X", X[i])
    #     cv2.waitKey(1)


    # plt.subplot(121),plt.imshow(img)
    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(mask, cmap = 'gray')
    # plt.title('Mask'), plt.xticks([]), plt.yticks([])
    # plt.show()

    # crop_img = img[0:0+sliding_window_size, 0:0+sliding_window_size]
    # cv2.imshow("cropped", crop_img)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    return mask


if __name__ == '__main__':
    image = cv2.imread("D:/102GOPR_noise/GOPR0537/raindrop0066.jpg", 0)
    image = cv2.resize(image, (512, 512))
    separated = separate(image)
    cv2.imshow("separated", separated*255)
    detect(image)
    print(get_sum_high_frequency(image))
    # cv2.waitKey(0)

    # cap = cv2.VideoCapture("D:/100GOPRO_noise/GOPR0076_testing.MP4")
    cap = cv2.VideoCapture("D:/102GOPR_noise/GOPR0537.MP4")
    # cap = cv2.VideoCapture("D:/100GOPRO_noise/GOPR0080_testing.MP4")
    while cap.isOpened():

        ret, frame = cap.read()
        if ret:
            # count time
            start_time = time.time()
            frame = cv2.imread("D:/102GOPR_noise/GOPR0537/raindrop0066.jpg")
            frame = cv2.resize(frame, (512, 512))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = separate(gray)
            cv2.imshow("frame", frame)

            cv2.imshow("mask", mask*255)
            applied = super_hero.overlay_white(frame, mask*255)

            # super_hero.print_fps(applied, 1/(time.time() - start_time))

            cv2.imshow("applied", applied)

            cv2.waitKey(0)
            print("--- Executed time: %s seconds ---" % (time.time() - start_time))
            print("--- Fps:  ---", 1/(time.time() - start_time))


