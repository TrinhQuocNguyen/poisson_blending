import cv2
import numpy as np
from copy import deepcopy
import time


def blending(masked, mask, pred, show=False):
    """Blending to keep the information form the masked image

    Arguments:
        masked
        mask
        pred
        show
    """
    blended = deepcopy(masked)
    blended[mask == 0] = pred[mask ==0]

    if show:
        cv2.imshow("blended", blended)
        cv2.waitKey(0)

    return blended


def unsharp_mask(image, kernel_size=(5, 5), sigma= 5.0, amount=1.0, threshold=5):
    """Return a sharpened version of the image, using an unsharp mask."""
    # For details on unsharp masking, see:
    # https://en.wikipedia.org/wiki/Unsharp_masking
    # https://homepages.inf.ed.ac.uk/rbf/HIPR2/unsharp.htm
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)

    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.float32)

    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)

    return sharpened


def unsharp_mask_common(image, kernel_size=(9, 9), sigma=10.0, amount=1.5):
    """Return a sharpened version of the image, using an unsharp mask."""

    gaussian_3 = cv2.GaussianBlur(image, kernel_size, sigma)
    unsharp_image = cv2.addWeighted(image, amount, gaussian_3, 1-amount, 0)

    return unsharp_image

def cloning_images(ori_image_path, output_images_path, max):

    ori_image = cv2.imread(ori_image_path)
    ori_image = cv2.resize(ori_image, (1920, 1080))
    for n in range(0, max):
        print("PROCESSING: " + output_images_path + "/" + "eye_" + '{0:06}'.format(n) + '_mask.png')
        cv2.imwrite(output_images_path + "/" + "eye_" + '{0:06}'.format(n) + '_mask.png', ori_image)
        # cv2.imshow("img", ori_image)
        # cv2.waitKey(1)


def seamless_cloning(im, mask, obj, show=False,  mode = 'mixed_clone'):
    """Seamless cloning to make obj looked real in im backgound

    Arguments:
        obj {image} -- object - prossed image from model
        im {image} -- backgound image - noise image
        mask {image} -- mask of the obj

    Keyword Arguments:
        mode {str} -- mode of seamless cloning (default: {'mixed_clone'})
        - mixed_clone
        - normal_clone
        - monochrome_transfer
    Returns:
        darray -- processed image after cloning
    """
    # pre-processing the mask
    # convert to grayscale image
    gray_image = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # thresh = 127
    # im_bw = cv2.threshold(gray_image, thresh, 255, cv2.THRESH_BINARY)[1]
    # https://stackoverflow.com/questions/7624765/converting-an-opencv-image-to-black-and-white
    # convert to only 0 and 255
    (thresh, im_bw) = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # The location of the center of the src in the dst
    width, height, channels = im.shape
    # caculate the center point of the obj image
    minx = 1e5
    maxx = 1
    miny = 1e5
    maxy = 1

    for y in range(1, height):
        for x in range(1, width):
            # check wh
            # if ((im_bw[x][y] != 0) and (im_bw[x][y] != 255)):
            #     print(x, y , " : ",im_bw[x][y])
            if im_bw[x][y] > 0:
                minx = min(minx, x)
                maxx = max(maxx, x)
                miny = min(miny, y)
                maxy = max(maxy, y)

    center = (int(miny+(maxy-miny)/2), int(minx+(maxx-minx)/2))

    # center = (int(height / 2), int(width / 2))
    print("center", center)

    dest_cloned = obj

    # Seamlessly clone src into dst and put the results in output
    if mode == 'normal_clone':
        dest_cloned = cv2.seamlessClone(obj, im, im_bw, center, cv2.NORMAL_CLONE)
    elif mode == 'mixed_clone':
        dest_cloned = cv2.seamlessClone(obj, im, im_bw, center, cv2.MIXED_CLONE)
    else: # monochrome_transfer mode
        dest_cloned= cv2.seamlessClone(obj, im, im_bw, center, cv2.MONOCHROME_TRANSFER)

    # cv2.imshow("obj", obj)
    # cv2.imshow("im", im)
    # cv2.imshow("mask", mask)
    # cv2.imshow("dest_cloned", dest_cloned)

    # cv2.imshow("dest_cloned", dest_cloned)
    # cv2.waitKey(0)
    return dest_cloned


def overlay_white(im, mask):
    """
    Overlaying image with a mask:
    - information: white
    - background: black
    :param im: image needed overlaid
    :param mask:
    - information: white
    - background: black
    :return: overlaid image
    """

    dst = np.copy(im)
    dst[mask == 255.] = 255.

    return dst


def normalize_mask(image):
    """
    Convert white <-> black in a mask
    :param image: original mask
    :return: converted image
    """

    image_mask = np.copy(image)

    image_mask[image_mask < 128] = 0
    image_mask[image_mask >= 128] = 255

    return image_mask

def overlay_white_from_path(im_path, mask_path):

    """
    Overlaying image with a mask:
    - information: black
    - background: white
    :param im: image needed overlaid
    :param mask:
    - information: black
    - background: white
    :return: overlaid image
    """
    im = cv2.imread(im_path)
    mask = cv2.imread(mask_path)
    dst = np.copy(im)
    dst[mask == 255.] = 255.

    return dst

def overlay_black(im, mask):

    """
    Overlaying image with a mask:
    - information: black
    - background: white
    :param im: image needed overlaid
    :param mask:
    - information: black
    - background: white
    :return: overlaid image
    """
    dst = np.copy(im)
    dst[mask == 0.] = 255.

    return dst

def count_nonblack_np(img):
    """Return the number of pixels in img that are not black.
    img must be a Numpy array with colour values along the last axis.

    """
    nonblack = img.any(axis=-1).sum()

    percent = nonblack / (img.shape[0]*img.shape[1]) * 100
    return percent


def convert_white_black(image):
    """
    Convert white <-> black in a mask
    :param image: original mask
    :return: converted image
    """

    image_mask = np.copy(image)

    image_mask[image_mask <= 128] = 128
    image_mask[image_mask > 128] = 0
    image_mask[image_mask > 0] = 255

    return image_mask


def print_fps(img, fps=24.0):
    """
    print fps text on img
    :param img: original img
    :param fps: fps
    :return: printed image
    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    fps = round(fps, 2)
    cv2.putText(img, 'Fps: ' + str(fps), (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return img


def overlay_transparent(background, overlay, x, y):

    """
    Example:
    background = cv2.imread("./data/bg.jpg")
    # cv2.IMREAD_UNCHANGED parameter is important
    overlay = cv2.imread("./data/warning_02.png", cv2.IMREAD_UNCHANGED)
    img = overlay_transparent(background, overlay, 350, 350)
    cv2.imshow('dst', img)
    cv2.waitKey(0)

    :param background: background = cv2.imread("./data/bg.jpg")
    :param overlay: overlay = cv2.imread("./data/warning_02.png", cv2.IMREAD_UNCHANGED)
    :param x:
    :param y:
    :return: overlaid
    """
    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background


def overlay_red_color_using_mask(ori_img, mask, alpha=0.2, beta=0.8, gamma=0):

    red_img = np.zeros(ori_img.shape, ori_img.dtype)
    red_img[:, :] = (0, 0, 255)
    cv2.imshow("red_img", red_img)
    # cv2.waitKey(0)
    # red_mask = cv2.addWeighted(red_img, red_img, mask=mask)
    # red_mask = np.copy(im)

    red_img[mask==0]= 0
    # dst = src1 * alpha + src2 * beta + gamma;
    print(alpha)
    cv2.addWeighted(red_img, alpha, ori_img, beta, gamma, ori_img)
    return red_img



if __name__ == '__main__':
    # ori_img = cv2.imread("D:/tmp/GOPR1031_new_512_blended_02/170/results/deepcrack/test_latest/images/11142-1_image.png")
    # mask = cv2.imread("D:/tmp/GOPR1031_new_512_blended_02/170/results/deepcrack/test_latest/images/11142-1_gf.png")
    # cv2.imshow("1", ori_img)

    # red_img = overlay_red_color_using_mask(ori_img, mask)
    # cv2.imshow("ori_img", ori_img)
    # cv2.imshow("mask", mask)
    # cv2.imshow("red_img", red_img)
    # cv2.waitKey(0)
    mask = cv2.imread("D:/[DATA]/raindrop_koiwa/mask_collection/mask_2.png")

    abc = normalize_mask(mask)
    cv2.imwrite("D:/[DATA]/raindrop_koiwa/mask_collection/mask_02.png",abc)
    cv2.imshow("dst", abc)
    cv2.waitKey(0)


    # dst = overlay_white_from_path("C:/Users/Trinh/Downloads/koiwa/eye_000900.png", "C:/Users/Trinh/Downloads/koiwa/eye_000900_mask.png")
    # cv2.imshow("dst", dst)
    # cv2.imwrite("C:/Users/Trinh/Downloads/koiwa/eye_000900_masked.png",dst)
    #
    # cv2.waitKey(0)




