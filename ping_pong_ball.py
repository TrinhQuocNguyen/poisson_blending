import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import copy
import time

cap = cv2.VideoCapture(0)
x_before, y_before, r_before = 0, 0, 0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
black_board_raw = np.zeros((height, width), dtype='uint8')

while (1):

    # Take each frame
    _, frame = cap.read()
    start_time = time.time()
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    # Convert BGR to GRAYSCALE
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # black_board rewrite
    black_board = black_board_raw.copy()

    # define range of orange color in HLS
    lower_orange = np.array([15, 120, 90])
    upper_orange = np.array([30, 255, 255])

    # Threshold the HSV image to get only orange colors
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 3.0, 10000,
                               param1=10, param2=50)
    # Draw circles
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        a = 0.2
        b = 0.1

        x_now, y_now, r_now = circles[0]
        x = int(x_now * a + x_before * (1 - a))
        y = int(y_now * a + y_before * (1 - a))
        r = int(r_now * b + r_before * (1 - b))
        x_before, y_before, r_before = x, y, r
        cv2.circle(black_board, (x, y), int(r * 1.2), (255,255,255), -1)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=black_board)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('black_bord', black_board)
    cv2.imshow('res', res)
    print(time.time() - start_time)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()