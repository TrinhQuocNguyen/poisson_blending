import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import os, random

def overlay():
    im = cv2.imread("data/test2.jpg")
    mask = cv2.imread("data/test2_predict.jpg")

    # masked = np.ma.masked_where(mask == 0, mask)
    masked = np.ma.masked_where(mask == 0, mask)
    cv2.imshow("input", im)

    alpha = 0.9
    dst = cv2.addWeighted(im, alpha, masked, 1 - alpha, 0)

    cv2.imshow('dst', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_last_file_name(folder_path):
    """
    Get the last file name in a folder base on the first number in file name
    Only pay attention to files (not folders)

    :param folder_path: folder path to get the last file name
    :return: string of the last file name (or empty string)
    :example: ['101_abc.h5', '92_abc.h5'] -> '101_abc.h5'
    """

    from os import listdir
    from os.path import isfile, join
    only_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

    only_files = sorted(only_files, key=lambda x: int(x.split("_")[0]))

    if only_files:
        return only_files[-1]
    else:
        return ""


# https://stackoverflow.com/questions/24341114/simple-illumination-correction-in-images-opencv-c
def illumination_correction():
    # -----Reading the image-----------------------------------------------------
    img = cv2.imread('D:/tmp/test_seamless_cloning/output_pconv/raindrop0351.jpg')

    cv2.imshow("img", img)

    # -----Converting image to LAB Color model-----------------------------------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    cv2.imshow("lab", lab)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    cv2.imshow('l_channel', l)
    cv2.imshow('a_channel', a)
    cv2.imshow('b_channel', b)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    cl = clahe.apply(l)
    cv2.imshow('CLAHE output', cl)

    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))
    cv2.imshow('limg', limg)

    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    cv2.imshow('final', final)
    cv2.waitKey(0)


from random import randint
import itertools
import numpy as np
import cv2


def count_nonblack_np(img):
    """Return the number of pixels in img that are not black.
    img must be a Numpy array with colour values along the last axis.

    """
    nonblack = img.any(axis=-1).sum()

    percent = nonblack/ (img.shape[0]*img.shape[1]) * 100
    return percent

def random_mask_rectangles_center(height, width, channels=3, percent_from=10., percent_to=20., only_rec = True, short_rec= False):
    """Generate the random mask based on percentage of the entire image

    Arguments:
        height {int} -- height of the image
        width {int} -- width of the image

    Keyword Arguments:
        channels {int} -- chanel of the image (default: {3})
        percent_from {float} -- how many percent <from> (default: {10.})
        percent_to {float} -- how many percent <to> (default: {20.})
        only_rec {bool} -- only draw rectangles (default: {True})
    """
    space = 10
    center_point = (int(width/2), int(height/2))

    print(center_point)

    while True:
        img = np.zeros((height, width, channels), np.uint8)
        top_left_point = (center_point[0]-space, center_point[1]-space)
        right_bottom_point = (center_point[0]+space, center_point[1]+space)
        # thickness = randint(3, size)
        thickness = -1
        cv2.rectangle(img, top_left_point, right_bottom_point, (1,1,1), thickness)

        area = (space*2)*(space*2)
        print("area: ", area)
        print("percent: ", (area*100)/ (512*512))

        final_img = (img)*255
        # cv2.imshow("mask", final_img)
        # cv2.imwrite("./mask_levels/black_masks/mask_" + str(area) + ".jpg", final_img)
        # cv2.waitKey(0)
        space += 10
        if space >= width/2 or space >= height/2:
            break


def random_mask_rectangles_around(height, width, channels=3, percent_from=10., percent_to=20., only_rec = True, short_rec= False):
    """Generate the random mask based on percentage of the entire image

    Arguments:
        height {int} -- height of the image
        width {int} -- width of the image

    Keyword Arguments:
        channels {int} -- chanel of the image (default: {3})
        percent_from {float} -- how many percent <from> (default: {10.})
        percent_to {float} -- how many percent <to> (default: {20.})
        only_rec {bool} -- only draw rectangles (default: {True})
    """
    space = 56
    center_point = (int(width/2), int(height/2))

    print(center_point)

    while True:
        img = np.zeros((height, width, channels), np.uint8)
        # top_left_point = (center_point[0]-space, center_point[1]-space)
        # right_bottom_point = (center_point[0]+space, center_point[1]+space)

        temp = 56
        top_left_point = (temp, temp)
        right_bottom_point = (temp + space, temp + space)

        # thickness = randint(3, size)
        thickness = -1

        # cv2.rectangle(img, (temp, temp), (temp + space, temp + space), (1,1,1), thickness)
        # cv2.rectangle(img, (temp*3, temp), (temp*3 + space, temp + space), (1,1,1), thickness)
        # cv2.rectangle(img, (temp*5, temp), (temp*5 + space, temp + space), (1,1,1), thickness)
        # cv2.rectangle(img, (temp*7, temp), (temp*7 + space, temp + space), (1,1,1), thickness)

        # cv2.rectangle(img, (temp, temp*3), (temp + space, temp*3 + space), (1,1,1), thickness)
        cv2.rectangle(img, (temp*3, temp*3), (temp*3 + space, temp*3 + space), (1,1,1), thickness)
        cv2.rectangle(img, (temp*5, temp*3), (temp*5 + space, temp*3 + space), (1,1,1), thickness)
        # cv2.rectangle(img, (temp*7, temp*3), (temp*7 + space, temp*3 + space), (1,1,1), thickness)

        # cv2.rectangle(img, (temp, temp*5), (temp + space, temp*5 + space), (1,1,1), thickness)
        cv2.rectangle(img, (temp*3, temp*5), (temp*3 + space, temp*5 + space), (1,1,1), thickness)
        cv2.rectangle(img, (temp*5, temp*5), (temp*5 + space, temp*5 + space), (1,1,1), thickness)
        # cv2.rectangle(img, (temp*7, temp*5), (temp*7 + space, temp*5 + space), (1,1,1), thickness)

        # cv2.rectangle(img, (temp, temp*7), (temp + space, temp*7 + space), (1,1,1), thickness)
        # cv2.rectangle(img, (temp*3, temp*7), (temp*3 + space, temp*7 + space), (1,1,1), thickness)
        # cv2.rectangle(img, (temp*5, temp*7), (temp*5 + space, temp*7 + space), (1,1,1), thickness)
        # cv2.rectangle(img, (temp*7, temp*7), (temp*7 + space, temp*7 + space), (1,1,1), thickness)

        area = 4
        print("area: ", area)
        print("percent: ", (area*100)/ (512*512))

        final_img = (img)*255
        cv2.imshow("mask", final_img)
        cv2.imwrite("./mask_levels/masks_around/mask_" + str(area) + ".jpg", final_img)
        cv2.waitKey(0)
        # temp += space
        if space >= width/2 or space >= height/2:
            break


def crop_images(path):
    img = cv2.imread(path)



    width, height = 800, 640
    img = cv2.resize(img,(width, height))
    cv2.imshow("img", img)
    cv2.waitKey(0)
    # crop the image using array slices -- it's a NumPy array
    # after all!
    cropped = img[0:height, 0:int(width*80/100)]
    cv2.imshow("cropped", cropped)
    cv2.waitKey(0)

def random_mask_rectangles(height, width, channels=3, percent_from=10., percent_to=20., only_rec = True, short_rec= False):
    """Generate the random mask based on percentage of the entire image

    Arguments:
        height {int} -- height of the image
        width {int} -- width of the image

    Keyword Arguments:
        channels {int} -- chanel of the image (default: {3})
        percent_from {float} -- how many percent <from> (default: {10.})
        percent_to {float} -- how many percent <to> (default: {20.})
        only_rec {bool} -- only draw rectangles (default: {True})
    """
    while True:
        img = np.zeros((height, width, channels), np.uint8)
        # Draw random rectangles
        for _ in range(randint(1, 20)):
            x1, x2 = randint(1, width), randint(1, width)
            y1, y2 = randint(1, height), randint(1, height)
            # thickness = randint(3, size)
            thickness = -1
            if short_rec == True:
                if (-width/3 < (x1 - x2) < width/3) and -height/3 < (y1 - y2) < height/3:
                    cv2.rectangle(img, (x1,y1), (x2,y2), (1,1,1), thickness)
            else:
                cv2.rectangle(img, (x1,y1), (x2,y2), (1,1,1), thickness)

        if (only_rec != True):
            # Set size scale
            size = int((width + height) * 0.03)
            if width < 64 or height < 64:
                raise Exception("Width and Height of mask must be at least 64!")

            # Draw random lines
            for _ in range(randint(1, 10)):
                x1, x2 = randint(1, width), randint(1, width)
                y1, y2 = randint(1, height), randint(1, height)
                thickness = randint(3, size)
                cv2.line(img,(x1,y1),(x2,y2),(1,1,1),thickness)

            # Draw random circles
            for _ in range(randint(1, 10)):
                x1, y1 = randint(1, width), randint(1, height)
                radius = randint(3, size)
                cv2.circle(img,(x1,y1),radius,(1,1,1), -1)

            # Draw random ellipses
            for _ in range(randint(1, 15)):
                x1, y1 = randint(1, width), randint(1, height)
                s1, s2 = randint(1, width), randint(1, height)
                a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
                thickness = randint(3, size)
                cv2.ellipse(img, (x1,y1), (s1,s2), a1, a2, a3,(1,1,1), thickness)

        if (percent_from <= count_nonblack_np(img) < percent_to):
            print ("percent: ", count_nonblack_np(img))
            break

    return 1-img

def random_all_mask_irregular(height=512, width=512, channels=3, IRREGULAR_MASK_DATASET = "./all_masks"):
    """Load irregular mask for training and testing instead of generating it

    Arguments:
        height {int} -- height of the image
        width {int} -- width of the image

    Keyword Arguments:
        channels {int} -- chanel (default: {3})
        IRREGULAR_MASK_DATASET {str} -- masks foder dir (default: {"./data/testing_mask_dataset"})
    """
    number = random.randint(1, 10) # Integer from 1 to 10, endpoints included
    print(number)

    # percent
    if number > 7:
        image_mask = random_mask_rectangles(height,width, channels, percent_from=10., percent_to=50., only_rec=False)
        return image_mask*255

    # randomly choose a folder(or file) from the folder
    foldername = random.choice(os.listdir(IRREGULAR_MASK_DATASET))

    if os.path.isdir(IRREGULAR_MASK_DATASET + "/" + foldername):
        # in case of a directory
        filename = random.choice(os.listdir(IRREGULAR_MASK_DATASET + "/" + foldername))
        image_mask = cv2.imread(IRREGULAR_MASK_DATASET + "/" + foldername + "/" + filename)
        print(IRREGULAR_MASK_DATASET + "/" + foldername + "/" + filename)

    else:
        # in case of a file
        image_mask = cv2.imread(IRREGULAR_MASK_DATASET + "/" + foldername)
        print(IRREGULAR_MASK_DATASET + "/" + foldername)

    image_mask = cv2.resize(image_mask,(512,512))
    image_mask[image_mask <=128] = 128
    image_mask[image_mask > 128] = 0
    image_mask[image_mask > 0] = 255
    image_mask = image_mask/255.

    return image_mask


def convert_white_black (input_folder, output_folder):
    dir_files = os.listdir(input_folder)
    dir_files.sort()

    # t = time.time()
    for file_inter in dir_files:

        base_file_name = os.path.basename(file_inter)
        print("Processing: " + input_folder + "/" + base_file_name)

        image_mask = cv2.imread(input_folder + "/" + base_file_name)
        image_mask[image_mask <= 128] = 128
        image_mask[image_mask > 128] = 0
        image_mask[image_mask > 0] = 255

        # print(image_mask)

        cv2.imwrite(output_folder + "/" + base_file_name, image_mask)

        # cv2.imshow("image_mask", image_mask)
        # cv2.waitKey(0)




if __name__ == '__main__':
    # print(get_last_file_name("./data"))

    # crop_images("D:/102GOPR_noise/GOPR0537/raindrop0987.jpg")
    # illumination_correction()
    # random_mask_rectangles_center(512, 512, short_rec=False)

    # random_mask_rectangles_around(512, 512, short_rec=False)

    convert_white_black("D:/[PROJECTS]/qd-imd/qd_imd/train", "D:/[PROJECTS]/qd-imd/qd_imd/train_black")

    # image_mask = random_all_mask_irregular()
    # cv2.imshow("mask", image_mask)
    # cv2.waitKey(0)
