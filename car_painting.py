import numpy as np
import cv2
import matplotlib.pyplot as plt

import os


path = "car_painting_data/36_000.png"

src = cv2.imread(path)
# img = cv2.resize(src, (900, 900))
img = src


b = 30.  # brightness
c = 0.2  # contrast
img = cv2.addWeighted(img, 1. + c / 127., img, 0, b - c)

height, width = img.shape[:2]
print(height / 3)

one_third_height = int(height / 3)
one_third_width = int(width / 3)

cv2.line(img, (one_third_height, 0), (one_third_height, height), (255, 0, 0), 2)
cv2.line(img, (one_third_height * 2, 0), (one_third_height * 2, height), (255, 0, 0), 2)

cv2.line(img, (0, one_third_width), (width, one_third_width), (255, 0, 0), 2)
cv2.line(img, (0, one_third_width * 2), (width, one_third_width * 2), (255, 0, 0), 2)

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1.5
fontColor = (0, 255, 0)
normalColor = (255, 255, 255)
lineType = 2

# Positions

space = 40

top_left = (0, space)
top_center = (one_third_width, space)
top_right = (one_third_width * 2, space)

center_left = (0, one_third_height + space)
center = (one_third_width, one_third_height + space)
center_right = (one_third_width * 2, one_third_height + space)

down_left = (0, one_third_height * 2 + space)
down_center = (one_third_width, one_third_height * 2 + space)
down_right = (one_third_width * 2, one_third_height * 2 + space)

cv2.putText(img, 'normal', top_left, font, fontScale, normalColor, lineType)
cv2.putText(img, 'normal', top_center, font, fontScale, normalColor, lineType)
cv2.putText(img, 'normal', top_right, font, fontScale, normalColor, lineType)

cv2.putText(img, 'normal', center_left, font, fontScale, normalColor, lineType)
cv2.putText(img, 'normal', center, font, fontScale, normalColor, lineType)
cv2.putText(img, 'normal', center_right, font, fontScale, normalColor, lineType)


cv2.putText(img, 'normal', down_left, font, fontScale, normalColor, lineType)
cv2.putText(img, 'dirt', down_center, font, fontScale, fontColor, lineType)
cv2.putText(img, 'normal', down_right, font, fontScale, normalColor, lineType)



# cv2.line(img,(0,0),(511,511),(255,0,0),5)
# cv2.line(img,(0,0),(511,511),(255,0,0),5)

cv2.imshow("input", img)

cv2.imwrite(os.path.splitext(path)[0] + "_processed.png", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
