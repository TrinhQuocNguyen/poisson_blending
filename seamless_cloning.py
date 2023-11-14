#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os

def seamless_cloning( obj,im, mask, mode = 'mixed_clone'):
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


if __name__ == '__main__':
    print("seamless_cloning")
    mask_file = "D:/104GOPR/mask_demo/mask_06.jpg"
    obj_image = "D:/105GOPR/Untitled.png"

    input_folder = "D:/105GOPR/GOPR1031_new_02"
    output_images = "D:/105GOPR/GOPR1031_new_512_blended_02"
    output_video = "D:/105GOPR/GOPR1031_new_512_blended_02/GOPR1031_new_512_blended_02.mp4"

    dir_files = os.listdir(input_folder)
    dir_files.sort()

    # write video
    w = 512
    h = 512
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # out = cv2.VideoWriter('output.avi', fourcc, 30, (w, h))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_video, fourcc, 30, (w, h))

    # t = time.time()                                                                                                                                                                                                                           
    for file_inter in dir_files:

        base_file_name = os.path.basename(file_inter)

        image_s = cv2.imread(input_folder + "/" + base_file_name)
        mask_s = cv2.imread(mask_file)
        obj = cv2.imread(obj_image)
        image_s = cv2.resize(image_s, (512,512))
        mask_s = cv2.resize(mask_s, (512,512))
        obj = cv2.resize(obj, (512,512))

        # cv2.imshow("Dest", obj)
        # cv2.waitKey(0)

        dest_cloned_normal = seamless_cloning( obj,image_s, mask_s, 'normal_clone')
        cv2.imshow("Dest", dest_cloned_normal)
        cv2.imshow("image_s", image_s)
        cv2.imshow("obj", obj)
        cv2.imshow("mask_s", mask_s)
        cv2.waitKey(1)

        cv2.imwrite(output_images + "/" + base_file_name, dest_cloned_normal)
        # save a frame
        out.write(dest_cloned_normal)

        sharpened = unsharp_mask(dest_cloned_normal)

        # cv2.imwrite("D:/blender_output/nosie_GOPR0279_line_512/output_seamless_unsharp/" + base_file_name, sharpened)


