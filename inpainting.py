import numpy as np
import cv2 as cv

# img = cv.imread('data/messi.png')
# mask = cv.imread('data/mask.png', 0)


img = cv.imread('rain/i_raindrop0351.jpg')
mask = cv.imread('rain/m_raindrop0351.jpg', 0)

height, width = 640,640
img = cv.resize(img,(width, height))
mask = cv.resize(mask,(width, height))

dst_telea = cv.inpaint(img,mask,3,cv.INPAINT_TELEA)
dst_ns = cv.inpaint(img,mask,3,cv.INPAINT_NS)


# cv.imwrite("rain/inpainted_telea_opencv.jpg",dst)
cv.imwrite("rain/inpainted_telea_opencv.jpg",dst_telea)
cv.imwrite("rain/inpainted_ns_opencv.jpg",dst_ns)

cv.imshow('dst_telea', dst_telea)
cv.imshow('dst_ns', dst_ns)

cv.waitKey(0)

cv.destroyAllWindows()

