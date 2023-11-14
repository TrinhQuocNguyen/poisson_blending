import cv2
import numpy as np

import poissonblending


def unsharp_mask(image, kernel_size=(5, 5), sigma= 5.0, amount=1.0, threshold=5):
    """Return a sharpened version of the image, using an unsharp mask."""
    # For details on unsharp masking, see:
    # https://en.wikipedia.org/wiki/Unsharp_masking
    # https://homepages.inf.ed.ac.uk/rbf/HIPR2/unsharp.htm
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)

    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)

    return sharpened


def unsharp_mask_common(image, kernel_size=(5, 5), sigma= 5.0, amount=1.0, threshold=5):
    """Return a sharpened version of the image, using an unsharp mask."""
    gaussian_3 = cv2.GaussianBlur(image, kernel_size, sigma)
    unsharp_image = cv2.addWeighted(image, amount, gaussian_3, -0.5, 0, image)

    return unsharp_image

if __name__ == '__main__':

    #image = cv2.imread("testimages/test1_ret.png")

    # image = cv2.imread('D:/tmp/test_seamless_cloning/output_pconv/raindrop0351.jpg')
    image = cv2.imread("D:/blender_output/nosie_GOPR0268_line/_raindrop_6288.jpg")
    image = cv2.resize(image, (512, 512))

    print(image)


    cv2.imshow("image", image)
    #gaussian_3 = cv2.GaussianBlur(image, (9, 9), 10.0)

    #unsharp_image = cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0)
    sharpened = unsharp_mask(image)

    #cv2.imshow("unsharp mask", unsharp_image)
    cv2.imshow("sharpened", sharpened)
    cv2.waitKey(0)
    # cv2.imwrite("lenna_unsharp.jpg", unsharp_image)