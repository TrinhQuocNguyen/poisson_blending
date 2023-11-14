import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('D:/[PROJECTS]/BlurDetection2/raindrop0351.jpg',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
# fshift = np.copy(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

#
# plt.subplot(121),plt.imshow(img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
#
# plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()

# magnitude_spectrum = magnitude_spectrum.astype(np.uint8)
# cv2.imshow("img", img)
# cv2.imshow("magnitude_spectrum", magnitude_spectrum)
# cv2.waitKey(0)

rows, cols = img.shape
crow,ccol = int(rows/2) , int(cols/2)

# fshift[crow-30:crow+30, ccol-30:ccol+30] = 0

magnitude_spectrum_2 = 20*np.log(np.abs(fshift))

f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# plt.subplot(221),plt.imshow(img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
#
# plt.subplot(222),plt.imshow(img_back, cmap = 'gray')
# plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
#
# plt.subplot(223),plt.imshow(img_back)
# plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
#
# plt.subplot(224),plt.imshow(magnitude_spectrum_2, cmap = 'gray')
# plt.title('magnitude_spectrum_2'), plt.xticks([]), plt.yticks([])
#
# plt.show()



img = cv2.imread('D:/[PROJECTS]/BlurDetection2/raindrop0351.jpg',0)

dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
# dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft[:,:,0], dft[:,:,1]))

# plt.subplot(121),plt.imshow(img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()

rows, cols = img.shape
crow,ccol = int(rows/2) , int(cols/2)

# create a mask first, center square is 1, remaining all zeros
mask = np.ones((rows,cols,2),np.uint8)
# mask[crow-30:crow+30, ccol-30:ccol+30] = 1
# print(type(mask))
# apply mask and inverse DFT
f_ishift = dft*mask
# f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
print(type(img_back))

plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(img_back, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(mask[:,:,1], cmap = 'gray')
plt.title('mask'), plt.xticks([]), plt.yticks([])
plt.show()
