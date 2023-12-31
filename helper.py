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