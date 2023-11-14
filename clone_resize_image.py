import cv2
import os

from os import listdir
from os.path import isfile, join
import numpy as np


def get_last_file_name(folder_path):
    """
    Get the last file name in a folder base on the first number in file name
    Only pay attention to files (not folders)

    :param folder_path: folder path to get the last file name
    :return: string of the last file name (or empty string)
    :example: ['101_abc.h5', '92_abc.h5'] -> '101_abc.h5'
    """

    only_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

    only_files.sort()
    for item in only_files:
        new_image = cv2.imread(folder_path + "/" + item)

        vis = np.concatenate((new_image, new_image), axis=1)
        cv2.imshow("Hello", vis)
        cv2.imwrite("output_pconv_loned/" + item, vis)
        # cv2.waitKey(0)
    print(only_files)


def resize(folder_path_source, folder_path_dest, width = 512, height=512):
    only_files = [f for f in listdir(folder_path_source) if isfile(join(folder_path_source, f))]

    only_files.sort()
    for item in only_files:
        new_image = cv2.imread(folder_path_source + "/" + item)
        print(folder_path_source + "/" + item)

        h, w, c = new_image.shape
        print('width:  ', w)
        print('height: ', h)
        print('channel:', c)

        dest_image = cv2.resize(new_image, (width, height))

        cv2.imshow("dest_image", dest_image)
        cv2.imwrite(folder_path_dest + "/" + item, dest_image)
        cv2.waitKey(3)
    print(only_files)


if __name__ == '__main__':
    # get_last_file_name("./output_pconv")

    # resize("D:/107GOPR/GOPR1107",
    #        "D:/107GOPR/GOPR1107_512")
    # resize("D:/tmp/a", "D:/tmp/resized")
    resize("D:/tmp/a/standard_mask", "D:/tmp/a/standard_mask_resized")
