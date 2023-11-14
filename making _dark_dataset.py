import random
import numpy as np
import cv2
import os
import numpy as np


def change_contrast_brightness(img, c_from=1.0, c_to=1.0, b_from=-200, b_to=-100):
    """
    https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
    :param img: input image
    :param c_from: from contrast value
    :param c_to: to contrast value
    :param b_from: from brightness value
    :param b_to: to brightness value
    :return: output image
    """
    alpha = random.uniform(c_from, c_to)
    beta = random.uniform(b_from, b_to)
    print("alpha (contrast):", alpha)
    print("beta (brightness): ", beta)
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return out


def gamma_correction(img, g_from=50.0, g_to=100.5):
    """
    https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
    :param img: input image
    :param g_from: from gamma value
    :param g_to: to gamma value
    :return: output image
    """
    gamma = random.uniform(g_from, g_to)
    print("gamma: ", gamma)
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    out = cv2.LUT(img, lookUpTable)
    return out


def noisy(noise_typ, image, n_from = 0, n_to = 0, random = False):
    """
    Link: https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
    Parameters
    ----------
    image : ndarray
        Input image data. Will be converted to float.
    mode : str
        One of the following strings, selecting the type of noise to add:

        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        's&p'       Replaces random pixels with 0 or 1.
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n is uniform noise with specified mean & variance.

    n_from: from noise value
    n_to: to noise value
    """

    if random:
        choices = ["gauss", "poisson", "s&p", "speckle"]
        noise_typ = random.choice(choices)

    print(noise_typ)
    image = image / 255.
    noisy = image

    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        noisy = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        noisy[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        noisy[coords] = 0
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)

    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss

    return noisy


def generate(input_path, output_path):
    """
    :param input_path: input images path
    :param output_path: output images path
    """
    dir_files = os.listdir(input_path)
    dir_files.sort()
    for file_inter in dir_files:
        base_file_name = os.path.basename(file_inter)
        img = cv2.imread(input_path + "/" + base_file_name)
        out_img = change_contrast_brightness(img)
        out_img = gamma_correction(out_img)
        out_img = noisy(out_img)

        cv2.imwrite(output_path + "/" + base_file_name, out_img)
        cv2.imshow("out_img", out_img)
        cv2.waitKey(1)


if __name__ == '__main__':
    print("START")
    img = cv2.imread("code_seamless/newbg.jpg")
    # out = change_contrast_brightness(img)
    # out = gamma_correction(out)

    out = noisy("poisson", img)
    cv2.imshow("Input", img)
    cv2.imshow("Output", out)
    cv2.waitKey(0)
