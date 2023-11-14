import cv2
import numpy as np
cap = cv2.VideoCapture("//NAS5/Level4/4100_顧客_準最高機密/太陽誘電 - Taiyo Yuden/Traffic 2/2020_10_07_Data_of_1_angle_long/2020_09_30_dark_trim/2020_09_30_Trim01.mp4")

_, first_frame = cap.read()
first_frame = cv2.resize(first_frame, (500, 256))
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_gray = cv2.GaussianBlur(first_gray, (5, 5), 0)

while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, (500, 256))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    first_frame = cv2.resize(first_frame, (500, 256))

    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    difference = cv2.absdiff(first_gray, gray_frame)
    _, difference = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)
    cv2.imshow("First frame", first_frame)
    cv2.imshow("Frame", frame)
    cv2.imshow("difference", difference)

    key = cv2.waitKey(30)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()