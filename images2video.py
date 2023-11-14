#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os

import super_hero



def images2video(out_video_path, images_folder_path, frame=60, w=512*2, h=512):
    dir_files = os.listdir(images_folder_path)
    dir_files.sort()

    # write video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(out_video_path, fourcc, frame, (w, h))

    for file_inter in dir_files:
        base_file_name = os.path.basename(file_inter)

        # input_image = cv2.imread(images_folder_path + "/" + base_file_name)
        input_image = cv2.imdecode(np.fromfile(images_folder_path + "/" + base_file_name, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        # output_image = input

        # cv2.putText(input_image, "Camera 1", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 256, 256), 2, cv2.LINE_AA)
        # cv2.putText(input_image, "Camera 2", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 256), 2, cv2.LINE_AA)


        input_image = cv2.resize(input_image, (w, h))

        print("PROCESSING: ", images_folder_path + "/" + base_file_name)
        # cv2.imshow("input_image", input_image)
        # cv2.imwrite(out_video_path + '/'+ base_file_name, output_image)
        cv2.waitKey(1)

        # save a frame
        out.write(input_image)


def images2images(out_image_path, in_image_path, new_w=256, new_h=256):
    dir_files = os.listdir(in_image_path)
    dir_files.sort()

    for file_inter in dir_files:
        base_file_name = os.path.basename(file_inter)

        # input_image = cv2.imread(images_folder_path + "/" + base_file_name)
        input_image = cv2.imdecode(np.fromfile(in_image_path + "/" + base_file_name, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        output_image = cv2.resize(input_image, (new_w, new_h))
        # output_image = input

        print("PROCESSING: ", in_image_path + "/" + base_file_name)
        cv2.imshow("input_image", input_image)
        cv2.imwrite(out_image_path + '/'+ base_file_name, output_image)
        cv2.waitKey(1)

def images2imagesOKNG(out_image_path, in_image_path, OK=True):
    dir_files = os.listdir(in_image_path)
    dir_files.sort()

    for file_inter in dir_files:
        base_file_name = os.path.basename(file_inter)

        # input_image = cv2.imread(images_folder_path + "/" + base_file_name)
        input_image = cv2.imdecode(np.fromfile(in_image_path + "/" + base_file_name, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        print(input_image)
        print(input_image.shape)
        if OK==True:
            cv2.putText(input_image, "OK", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 256, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(input_image, "NG", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 256), 2, cv2.LINE_AA)
        # output_image = input

        print("PROCESSING: ", in_image_path + "/" + base_file_name)
        cv2.imshow("input_image", input_image)
        # cv2.imwrite(out_image_path + '/'+ base_file_name, input_image)
        cv2.waitKey(1)


def fix_video_format(input_video_path="", output_video_path="",  skip_frame=1, frame=30, w =512,  h=512):
    cap = cv2.VideoCapture(input_video_path)
    n = 1
    inv_f = 0
    makedirs(output_video_path+ "/tmp")
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print("Can not read this frame!!")
            if inv_f >100:
                print("The number of invalid frames is over 100 !!")
                break
            else:
                inv_f +=1
                continue
        if n % skip_frame == 0:
            cv2.waitKey(1)
            print("PROCESSING: " + output_video_path + "/" + '{0:06}'.format(n) + '.png')
            cv2.imwrite(output_video_path+ "/tmp" + "/" + "eye_" + '{0:06}'.format(n) + '.png', img)

            cv2.imshow("img", img)
        n += 1

    images2video(output_video_path+"/output.mp4", output_video_path+ "/tmp", frame, w, h)


def video2images(input_video_path="", output_images_path="", skip_frame=1):
    cap = cv2.VideoCapture(input_video_path)
    n = 0
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print("Can not read this frame!!")
            continue
        if n%skip_frame==0:
            # if n < 60* 8 * 60 - 200:
            #     print("Skipped!")
            #     n += 1
            #     continue

            # cv2.imshow("img", img)
            cv2.waitKey(1)
            print("PROCESSING: " + output_images_path + "/" + '{0:06}'.format(n) + '.png')
            # cv2.imwrite(output_images_path + "/" +'{0:06}'.format(n)+'.jpg', img)

            # img = cv2.resize(img, (1920, 1080))
            # if (n > 216) and (n < 340):
            #     cv2.putText(img, "Abnormal", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            # else:
            # cv2.putText(img, '{0:06}'.format(n), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # h = 720
            # w = 1155
            # y = 0
            # x = 0
            #
            # img = img[y:y + h, x:x + w]
            # img = cv2.resize(img, (1920, 1080))

            # cv2.imwrite(output_images_path + "/" + "eye_" + '{0:06}'.format(n) + '.png', img)

            cv2.imshow("img", img)
        n += 1
        print(n)


def video2video(input_video_path, out_video_path, frame, w, h):
    cap = cv2.VideoCapture(input_video_path)
    n = 0
    # write video
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(out_video_path, fourcc, frame, (w, h))
    while cap.isOpened():
        ret, img = cap.read()
        # img = cv2.resize(img, (w, h))

        if not ret:
            print("Can not read this frame!!")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            break

        print("PROCESSING frame: " + str(n))
        # cv2.imwrite("C:/Users/Trinh/Downloads/askdjaklsdjklasdklj_20220220/02.Tracking_and_HBOE/output" + "/" + 'test.png', img)

        # img = cv2.resize(img, (w, h))
        # cv2.imwrite(output_images_path + "/" + "raindrop_" +'{0:06}'.format(n)+'.jpg', img)
        # out_img = cv2.imdecode(np.fromfile("C:/Users/Trinh/Downloads/askdjaklsdjklasdklj_20220220/02.Tracking_and_HBOE/output" + "/" + 'test.png', dtype=np.uint8), cv2.IMREAD_UNCHANGED)

        cv2.imshow("img", img)
        cv2.waitKey(1)

        # save a frame
        out_img = cv2.resize(img, (w, h))
        out.write(out_img)
        n += 1

        # if n == 132:
        #     break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def demo_video_2_old(path_01, path_02, output_video, text_01="", text_02="", vertical=False, fps= 30, w=512, h=512, with_arrow = False):
    dir_files = os.listdir(path_01)
    dir_files.sort()

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    if with_arrow:
        img_arrow = cv2.imread("./demo_supporter/arrow_large.png")
        # height, width, channels = img_arrow.shape
        print(img_arrow.shape)
        if vertical == False:
            out = cv2.VideoWriter(output_video, fourcc, fps, (w * 2 + img_arrow.shape[1], h))
            print((w * 2 + img_arrow.shape[1], h))
        else:
            return
    else:
        if vertical == False:
            out = cv2.VideoWriter(output_video, fourcc, fps, (w * 2, h))
        else:
            out = cv2.VideoWriter(output_video, fourcc, fps, (w, h * 2))

    # t = time.time()
    for file_inter in dir_files:
        base_file_name = os.path.basename(file_inter)
        image_01 = cv2.imread(path_01 + "/" + base_file_name)
        image_02 = cv2.imread(path_02 + "/" + base_file_name)


        if with_arrow == True:
            if vertical == False:
                dest_image = np.concatenate((image_01,img_arrow, image_02), axis=1)

        else:
            if vertical == False:
                dest_image = np.concatenate((image_01, image_02), axis=1)
            else:
                dest_image = np.concatenate((image_01, image_02), axis=0)
            cv2.putText(dest_image, text_01, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 256, 256), 1, cv2.LINE_AA)
            cv2.putText(dest_image, text_02, (w + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 256), 1, cv2.LINE_AA)


        # cv2.imwrite("D:/107GOPR/GOPR1115_512_01/tmp" + "/" + base_file_name, dest_image)

        # save a frame
        # out.write(dest_image)

        # dest_image.resize((261, 705))
        print("Processing: ", base_file_name)

        # cv2.imshow("dest_image", dest_image)
        # cv2.waitKey(1)

def demo_video_2(path_01, path_02, output_video, text_01="", text_02="", vertical=False, fps= 30, with_arrow = False,
                 arrow_path = "", with_logo = False, logo_path = "", resized = False, w0=512, h0=512):
    dir_files = os.listdir(path_01)
    dir_files.sort()
    # dir_files.sort(key=lambda f: int(filter(str.isdigit, f)))
    # dir_files = sorted(dir_files, key=lambda x: int(os.path.splitext(x)[0]))
    print(dir_files)

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # get shape
    w = w0
    h = h0

    if resized:
        w = w0
        h = h0
    else:
        # calculate
        first_img = cv2.imread(path_01 + "/" + dir_files[0])
        print(path_01 + "/" + dir_files[0])
        h, w, c = first_img.shape
        print(h, w, c)

    if with_arrow:
        hr = h
        wr = int(w/8)
        blank_image = np.zeros((hr, wr, 3), np.uint8)
        img_arrow = cv2.imread(arrow_path)
        # resize arrow
        hr0, wr0, cr0 = img_arrow.shape
        print(img_arrow.shape)
        img_arrow = cv2.resize(img_arrow, (wr, int(wr*hr0/wr0)))

        x_offset = 0
        y_offset = int(hr/2 - img_arrow.shape[0]/2)

        blank_image[y_offset:y_offset+img_arrow.shape[0], x_offset:x_offset+img_arrow.shape[1]] = img_arrow
        if with_logo:
            img_logo = cv2.imread(logo_path)
            hr1, wr1, cr1 = img_logo.shape

            img_logo = cv2.resize(img_logo, (wr, int(wr*hr1/wr1)))
            y_offset_logo = int(hr - img_logo.shape[0])
            blank_image[y_offset_logo:y_offset_logo+img_logo.shape[0], x_offset:x_offset+img_logo.shape[1]] = img_logo


        # blank_image = cv2.resize(blank_image, (int(wr/4), int(hr/4)))
        # cv2.imshow("blank", blank_image)
        # cv2.waitKey(0)

        if vertical == False:
            out = cv2.VideoWriter(output_video, fourcc, fps, (w * 2 + blank_image.shape[1], h))
            print((w * 2 + blank_image.shape[1], h))
        else:
            return
    else:
        if vertical == False:
            out = cv2.VideoWriter(output_video, fourcc, fps, (w * 2, h))
        else:
            out = cv2.VideoWriter(output_video, fourcc, fps, (w, h * 2))

    # t = time.time()
    for file_inter in dir_files:
        base_file_name = os.path.basename(file_inter)
        image_01 = cv2.imread(path_01 + "/" + base_file_name)
        image_02 = cv2.imread(path_02 + "/" + base_file_name)
        if resized:
            image_01 = cv2.resize(image_01, (w0, h0))
            image_02 = cv2.resize(image_02, (w0, h0))


        if with_arrow == True:
            if vertical == False:
                dest_image = np.concatenate((image_01,blank_image, image_02), axis=1)

        else:
            if vertical == False:
                dest_image = np.concatenate((image_01, image_02), axis=1)
                cv2.putText(dest_image, text_01, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 256, 256), 1, cv2.LINE_AA)
                cv2.putText(dest_image, text_02, (w + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 256), 1,
                            cv2.LINE_AA)
                # cv2.putText(dest_image, base_file_name, (w + 90, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 256), 1,
                #            cv2.LINE_AA)
            else:
                dest_image = np.concatenate((image_01, image_02), axis=0)
                cv2.putText(dest_image, text_01, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 256, 256), 2, cv2.LINE_AA)
                cv2.putText(dest_image, text_02, (30, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 256), 2, cv2.LINE_AA)
                # cv2.putText(dest_image, base_file_name, (30, h + 90 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 256), 2,
                #            cv2.LINE_AA)

        if with_logo:
            img_logo = cv2.imread(logo_path)
            hr1, wr1, cr1 = img_logo.shape

            # img_logo = cv2.resize(img_logo, (100, int(100*hr1/wr1)))
            # y_offset_logo = 0
            # x_offset = 720
            # dest_image[y_offset_logo:y_offset_logo+img_logo.shape[0], x_offset:x_offset+img_logo.shape[1]] = img_logo

        # save a frame
        out.write(dest_image)
        # dest_image.resize((261, 705))
        print("Processing: ", base_file_name)
        # dest_image= cv2.resize(dest_image, (1280, 1440))
        cv2.imshow("dest_image", dest_image)
        cv2.waitKey(1)

def demo_alighting(path_01, path_02, num_01, num_02, output_video, text_01="", text_02="", vertical=False, fps= 30, with_arrow = False,
                 arrow_path = "", with_logo = False, logo_path = "", resized = False, w0=512, h0=512):
    dir_files = os.listdir(path_01)
    dir_files.sort()
    # dir_files = sorted(dir_files, key=lambda x: int(os.path.splitext(x)[0]))
    print(dir_files)

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # get shape
    w = w0
    h = h0

    if resized:
        w = w0
        h = h0
    else:
        # calculate
        first_img = cv2.imread(path_01 + "/" + dir_files[0])
        print(path_01 + "/" + dir_files[0])
        h, w, c = first_img.shape
        print(h, w, c)

    if with_arrow:
        hr = h
        wr = int(w/8)
        blank_image = np.zeros((hr, wr, 3), np.uint8)
        img_arrow = cv2.imread(arrow_path)
        # resize arrow
        hr0, wr0, cr0 = img_arrow.shape
        print(img_arrow.shape)
        img_arrow = cv2.resize(img_arrow, (wr, int(wr*hr0/wr0)))

        x_offset = 0
        y_offset = int(hr/2 - img_arrow.shape[0]/2)

        blank_image[y_offset:y_offset+img_arrow.shape[0], x_offset:x_offset+img_arrow.shape[1]] = img_arrow
        if with_logo:
            img_logo = cv2.imread(logo_path)
            hr1, wr1, cr1 = img_logo.shape

            img_logo = cv2.resize(img_logo, (wr, int(wr*hr1/wr1)))
            y_offset_logo = int(hr - img_logo.shape[0])
            blank_image[y_offset_logo:y_offset_logo+img_logo.shape[0], x_offset:x_offset+img_logo.shape[1]] = img_logo


        # blank_image = cv2.resize(blank_image, (int(wr/4), int(hr/4)))
        # cv2.imshow("blank", blank_image)
        # cv2.waitKey(0)

        if vertical == False:
            out = cv2.VideoWriter(output_video, fourcc, fps, (w * 2 + blank_image.shape[1], h))
            print((w * 2 + blank_image.shape[1], h))
        else:
            return
    else:
        if vertical == False:
            out = cv2.VideoWriter(output_video, fourcc, fps, (w * 2, h))
        else:
            out = cv2.VideoWriter(output_video, fourcc, fps, (w, h * 2))

    # t = time.time()
    for file_inter in dir_files:
        base_file_name = os.path.basename(file_inter)
        image_01 = cv2.imread(path_01 + "/" + base_file_name)
        image_02 = cv2.imread(path_02 + "/" + base_file_name)
        if resized:
            image_01 = cv2.resize(image_01, (w0, h0))
            image_02 = cv2.resize(image_02, (w0, h0))


        if with_arrow == True:
            if vertical == False:
                dest_image = np.concatenate((image_01,blank_image, image_02), axis=1)

        else:
            if vertical == False:
                dest_image = np.concatenate((image_01, image_02), axis=1)
                cv2.putText(dest_image, text_01, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 256, 256), 1, cv2.LINE_AA)
                cv2.putText(dest_image, text_02, (w + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 256), 1,
                            cv2.LINE_AA)
                # cv2.putText(dest_image, base_file_name, (w + 90, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 256), 1,
                #            cv2.LINE_AA)
            else:
                dest_image = np.concatenate((image_01, image_02), axis=0)
                cv2.putText(dest_image, text_01, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 256, 256), 2, cv2.LINE_AA)
                cv2.putText(dest_image, text_02, (30, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 256), 2, cv2.LINE_AA)
                # cv2.putText(dest_image, base_file_name, (30, h + 90 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 256), 2,
                #            cv2.LINE_AA)

        if with_logo:
            img_logo = cv2.imread(logo_path)
            hr1, wr1, cr1 = img_logo.shape

            img_logo = cv2.resize(img_logo, (200, int(200*hr1/wr1)))
            y_offset_logo = 0
            x_offset = 820
            dest_image[y_offset_logo:y_offset_logo+img_logo.shape[0], x_offset:x_offset+img_logo.shape[1]] = img_logo

        # save a frame
        out.write(dest_image)
        # dest_image.resize((261, 705))
        print("Processing: ", base_file_name)
        dest_image= cv2.resize(dest_image, (1280, 1440))
        cv2.imshow("dest_image", dest_image)
        cv2.waitKey(1)

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


def compare_4(output_path, path_01, path_02, path_03, path_04, text_01="", text_02="", text_03="", text_04="",  w0=512, h0=512):
    dir_files = os.listdir(path_01)
    dir_files.sort()

    # t = time.time()
    for file_inter in dir_files:
        base_file_name = os.path.basename(file_inter)
        image_01 = cv2.imread(path_01 + "/" + base_file_name)
        image_02 = cv2.imread(path_02 + "/" + base_file_name)
        image_03 = cv2.imread(path_03 + "/" + base_file_name)
        image_04 = cv2.imread(path_04 + "/" + base_file_name)

        dest_image_top = np.concatenate((image_01, image_02), axis=1)
        dest_image_bottom = np.concatenate((image_03, image_04), axis=1)
        dest_image = np.concatenate((dest_image_top, dest_image_bottom), axis=0)

        cv2.putText(dest_image, text_01, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(dest_image, text_02, (w0 + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(dest_image, text_03, (30, 30 + h0), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(dest_image, text_04, (w0 + 30, 30 + h0), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

        dest_image = cv2.resize(dest_image, (1920, 1080))
        cv2.imwrite(output_path+"/"+base_file_name,dest_image)
        cv2.imshow("dest_image", dest_image)

        cv2.waitKey(1)


def demo_video_4(path_01, path_02, path_03, path_04, output_video, text_01="", text_02="", text_03="", text_04="",
                 fps=30, with_arrow=False,
                 arrow_path="", with_logo=False, logo_path="", resized=False, w0=512, h0=512):
    dir_files = os.listdir(path_01)
    dir_files.sort()
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    if with_arrow:
        hr = h0
        wr = int(w0/8)
        blank_image = np.zeros((hr, wr, 3), np.uint8)
        img_arrow = cv2.imread(arrow_path)
        # resize arrow
        hr0, wr0, cr0 = img_arrow.shape
        print(img_arrow.shape)
        img_arrow = cv2.resize(img_arrow, (wr, int(wr*hr0/wr0)))

        x_offset = 0
        y_offset = int(hr/2 - img_arrow.shape[0]/2)

        blank_image[y_offset:y_offset+img_arrow.shape[0], x_offset:x_offset+img_arrow.shape[1]] = img_arrow

        out = cv2.VideoWriter(output_video, fourcc, fps, (w0* 2 + blank_image.shape[1], h0 * 2))
        print((w0 * 2 + blank_image.shape[1], h0))
    else:
        out = cv2.VideoWriter(output_video, fourcc, fps, (w0 * 2, h0 * 2))

    # t = time.time()
    for file_inter in dir_files:
        base_file_name = os.path.basename(file_inter)
        image_01 = cv2.imread(path_01 + "/" + base_file_name)
        image_02 = cv2.imread(path_02 + "/" + base_file_name)
        image_03 = cv2.imread(path_03 + "/" + base_file_name)
        image_04 = cv2.imread(path_04 + "/" + base_file_name)
        if with_arrow:
            dest_image_top = np.concatenate((image_01,blank_image, image_02), axis=1)
            dest_image_bottom = np.concatenate((image_03,blank_image, image_04), axis=1)
            dest_image = np.concatenate((dest_image_top, dest_image_bottom), axis=0)

            cv2.putText(dest_image, text_01, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(dest_image, text_02, (w0 + blank_image.shape[1] + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(dest_image, text_03, (30, 30 + h0), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(dest_image, text_04, (w0 + blank_image.shape[1] + 30, 30 + h0), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            dest_image_top = np.concatenate((image_01, image_02), axis=1)
            dest_image_bottom = np.concatenate((image_03, image_04), axis=1)
            dest_image = np.concatenate((dest_image_top, dest_image_bottom), axis=0)

            cv2.putText(dest_image, text_01, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(dest_image, text_02, (w0 + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(dest_image, text_03, (30, 30 + h0), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(dest_image, text_04, (w0 + 30, 30 + h0), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

        dest_image = cv2.resize(dest_image, (1920, 1080))
        cv2.imshow("dest_image", dest_image)
        # cv2.imwrite("C:/Users/Trinh/Downloads/final" + "/" + base_file_name, dest_image)
        out.write(dest_image)

        # if base_file_name == "raindrop_000085.png":
        #     break
        cv2.waitKey(1)



def demo_video_abnormal_4(path_01, path_02, path_03, path_04, output_video, text_01="", text_02="", text_03="", text_04="",
                 fps=30, with_arrow=False,
                 arrow_path="", with_logo=False, logo_path="", resized=False, w0=512, h0=512):
    dir_files = os.listdir(path_01)
    dir_files.sort()
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    n = 0

    if with_arrow:
        hr = h0
        wr = int(w0/8)
        blank_image = np.zeros((hr, wr, 3), np.uint8)
        img_arrow = cv2.imread(arrow_path)
        # resize arrow
        hr0, wr0, cr0 = img_arrow.shape
        print(img_arrow.shape)
        img_arrow = cv2.resize(img_arrow, (wr, int(wr*hr0/wr0)))

        x_offset = 0
        y_offset = int(hr/2 - img_arrow.shape[0]/2)

        blank_image[y_offset:y_offset+img_arrow.shape[0], x_offset:x_offset+img_arrow.shape[1]] = img_arrow

        out = cv2.VideoWriter(output_video, fourcc, fps, (w0* 2 + blank_image.shape[1], h0 * 2))
        print((w0 * 2 + blank_image.shape[1], h0))
    else:
        out = cv2.VideoWriter(output_video, fourcc, fps, (w0 * 2, h0 * 2))

    # t = time.time()
    for file_inter in dir_files:
        base_file_name = os.path.basename(file_inter)
        image_01 = cv2.imread(path_01 + "/" + base_file_name)
        image_02 = cv2.imread(path_02 + "/" + base_file_name)
        image_03 = cv2.imread(path_03 + "/" + base_file_name)
        image_04 = cv2.imread(path_04 + "/" + base_file_name)
        if with_arrow:
            dest_image_top = np.concatenate((image_01,blank_image, image_02), axis=1)
            dest_image_bottom = np.concatenate((image_03,blank_image, image_04), axis=1)
            dest_image = np.concatenate((dest_image_top, dest_image_bottom), axis=0)

            cv2.putText(dest_image, text_01, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(dest_image, text_02, (w0 + blank_image.shape[1] + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(dest_image, text_03, (30, 30 + h0), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(dest_image, text_04, (w0 + blank_image.shape[1] + 30, 30 + h0), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            flag = 1
            if (type(image_01) is not np.ndarray):
                image_01 = np.zeros((h0, w0, 3), np.uint8)
            if (type(image_02) is not np.ndarray):
                image_02 = np.zeros((h0, w0, 3), np.uint8)
            if (type(image_03) is not np.ndarray):
                image_03 = np.zeros((h0, w0, 3), np.uint8)
            if (type(image_04) is not np.ndarray):
                image_04 = np.zeros((h0, w0, 3), np.uint8)
                flag = 0


            dest_image_top = np.concatenate((image_01, image_02), axis=1)
            dest_image_bottom = np.concatenate((image_03, image_04), axis=1)
            dest_image = np.concatenate((dest_image_top, dest_image_bottom), axis=0)

            cv2.putText(dest_image, text_01, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 2, cv2.LINE_AA)
            if flag == 1:
                cv2.putText(dest_image, "Abnormal", (w0 + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(dest_image, text_02, (w0 + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 2,
                            cv2.LINE_AA)

            cv2.putText(dest_image, text_03, (30, 30 + h0), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(dest_image, text_04, (w0 + 30, 30 + h0), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 2, cv2.LINE_AA)


        # dest_image = cv2.resize(dest_image, (1920, 1080))
        cv2.imshow("dest_image", dest_image)
        n += 1
        # cv2.imwrite("C:/Users/Trinh/Downloads/final" + "/" + base_file_name, dest_image)
        out.write(dest_image)

        # if base_file_name == "raindrop_000085.png":
        #     break
        cv2.waitKey(1)

def alpha_blending(input_folder_path, output_images_path,
                   mask_path='D:/tmp/GOPR1031_new_512_blended_02/170/demo_tokyo/mask_tokyo_01.png'):
    dir_files = os.listdir(input_folder_path)
    dir_files.sort()

    # Read the images
    foreground = cv2.imread(mask_path)
    foreground = foreground.astype(float)

    alpha = cv2.imread(mask_path)
    alpha = normalize_mask(alpha)
    # cv2.imwrite("D:/tmp/a/IMG_2999/abc.png", alpha)

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float) / 255

    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha, foreground)

    n = 0
    for file_inter in dir_files:
        base_file_name = os.path.basename(file_inter)

        background = cv2.imread(input_folder_path + "/" + base_file_name)

        # Convert uint8 to float
        background = background.astype(float)
        # Multiply the background with ( 1 - alpha )
        background = cv2.multiply(1.0 - alpha, background)

        # Add the masked foreground and background.
        outImage = cv2.add(foreground, background)
        # outImage = cv2.subtract(foreground, background)

        outImage[background==0] = 0

        # Display image
        cv2.imshow("src", foreground / 255)
        cv2.imshow("outImg", outImage / 255)
        # cv2.imwrite(output_images_path + "/" + base_file_name, outImage)

        cv2.imwrite(output_images_path + "/" + "eye_" +'{0:06}'.format(n)+'.png', outImage)
        n += 1
        cv2.waitKey(1)


def copy_D1_D2(path_in="", path_out=""):
    import os.path
    from os import path
    import shutil

    if path_in=="" or path_out=="":
        print("Please check your path")
    else:
        # Processing
        dir_folders = os.listdir(path_in)
        for folder in dir_folders: # In each routeX folder
            full_path = path_in + "/"+folder
            if os.path.isdir(full_path):
                D1_folder_path = full_path + "/D1"
                D2_folder_path = full_path + "/D2"
                if path.exists(D1_folder_path):
                    # make dir
                    new_D1_folder_path = path_out + "/" + folder + "/D1"
                    # copy data
                    shutil.copytree(D1_folder_path, new_D1_folder_path)
                    print("COPYING FROM: " + D1_folder_path + " TO: " + new_D1_folder_path)

                if path.exists(D2_folder_path):
                    # make dir
                    new_D2_folder_path = path_out + "/" + folder + "/D2"
                    # copy data
                    shutil.copytree(D2_folder_path, new_D2_folder_path)
                    print("COPYING FROM: " + D2_folder_path + " TO: " + new_D2_folder_path)







def rename(images_folder_path):
    dir_files = os.listdir(images_folder_path)
    dir_files.sort()

    for file_inter in dir_files:
        base_file_name = os.path.basename(file_inter)

        input_image = cv2.imread(images_folder_path + "/" + base_file_name)
        # input_image = cv2.resize(input_image, (512, 512))
        new_name = base_file_name[:15] + base_file_name[16:]
        # cv2.imwrite(output_images_path + "/" + "raindrop_" +'{0:06}'.format(n)+'.jpg', outImage)
        print(new_name)
        cv2.imwrite("D:/tmp/GOPR1031_new_512_blended_02/abc/" + new_name, input_image)

        # print("PROCESSING")


def write_standard_masks(images_folder_path, output_standard_mask_path, standard_mask_path):
    """
    Copy and generate masks from the standard mask:
    input:  raindrop_01.jpg
            raindrop_02.jpg
    standard_mask:  stand_mask.jpg

    :rtype: object
    output: raindrop_01.jpg # mask
            raindrop_02.jpg # mask
    """
    dir_files = os.listdir(images_folder_path)
    dir_files.sort()

    standard_mask = cv2.imread(standard_mask_path)
    standard_mask = normalize_mask(standard_mask)
    # standard_mask = cv2.resize(standard_mask, (512,512))

    for file_inter in dir_files:
        base_file_name = os.path.basename(file_inter)
        cv2.imwrite(output_standard_mask_path + "/" + base_file_name, standard_mask)
        print("Writing: ", output_standard_mask_path, "/", base_file_name)



def write_white_black(input_file_path, output_file_path):
    img = cv2.imread(input_file_path)
    dest = super_hero.convert_white_black(img)
    print("writing: ", output_file_path)
    cv2.imwrite(output_file_path, dest)



def crop_and_save(path_01, path_02, x1, y1, x2, y2):
    """

    Top-left (x1, y1)
    Right-bottom (x2, y2)

    """
    # x1 = 210
    # y1 = 100
    # x2 = 310
    # y2 = 200
    dir_files = os.listdir(path_01)
    dir_files.sort()

    # t = time.time()
    for file_inter in dir_files:
        base_file_name = os.path.basename(file_inter)
        img = cv2.imread(path_01 + "/" + base_file_name)

        cropped = img[y1:y2, x1:x2]

        # x_offset = 313
        # y_offset = 147
        # (313, 147, 1324, 916)

        # x1_overlay = 320
        # y1_overlay = 240
        # x2_overlay = 640
        # y2_overlay = 480
        # cropped = cv2.resize(cropped, (int(x2_overlay - x1_overlay), int(y2_overlay- y1_overlay)))
        #
        # img[y1_overlay:y2_overlay, x1_overlay:x2_overlay] = cropped

        cv2.imshow('cropped', cropped)
        cv2.imshow('img', img)
        # cv2.imwrite(path_02 + "/" + base_file_name, cropped)
        cv2.waitKey(1)

def crop_video_and_save(input_video_path, out_video_path, path_02, x1, y1, x2, y2):
    """
    Top-left (x1, y1)
    Right-bottom (x2, y2)

    """
    cap = cv2.VideoCapture(input_video_path)
    n = 0
    # write video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(out_video_path, fourcc, 30, (1920, 1080))
    while cap.isOpened():
        ret, img = cap.read()
        # img = cv2.resize(img, (w, h))

        if not ret:
            print("Can not read this frame!!")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        print("PROCESSING frame: " + str(n))
        cropped = img[y1:y2, x1:x2]
        cv2.imshow('cropped', cropped)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.imshow('img', img)

        cv2.imwrite(path_02 + "/" + "eye_" + '{0:06}'.format(n) + '.png', cropped)
        cv2.waitKey(1)

        # save a frame
        # out.write(img)
        n += 1

        # if n == 600:
        #     break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()



def crop_and_save_video(input_video_path, output_images_path, x1, y1, x2, y2):
    """
    Top-left (x1, y1)
    Right-bottom (x2, y2)
    """
    x1 = 190
    y1 = 100
    x2 = 380
    y2 = 200
    x1_overlay = 180
    y1_overlay = 240
    x2_overlay = 640
    y2_overlay = 480

    cap = cv2.VideoCapture(input_video_path)
    n = 1
    while cap.isOpened():
        ret, img = cap.read()

        if not ret:
            print("Can not read this frame!!")
            continue
        cv2.waitKey(1)
        cropped = img[y1:y2, x1:x2]
        cropped = cv2.resize(cropped, (int(x2_overlay - x1_overlay), int(y2_overlay- y1_overlay)))

        img[y1_overlay:y2_overlay, x1_overlay:x2_overlay] = cropped
        cv2.waitKey(1)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

        cv2.imshow('img', img)

        print("PROCESSING: " + output_images_path + "/eye_" + '{0:06}'.format(n) + '.jpg')
        cv2.imwrite(output_images_path + "/" + "eye_" + '{0:06}'.format(n) + '.jpg', img)

        n += 1


def makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def crop_to_4_and_save_into_folder(input_path, output_path,height, width):
    """

    Top-left (x1, y1)
    Right-bottom (x2, y2)

    """

    dir_files = os.listdir(input_path)
    dir_files.sort()

    path_01 = output_path + '/' + '01_tl'
    path_02 = output_path + '/' + '02_tr'
    path_03 = output_path + '/' + '03_bl'
    path_04 = output_path + '/' + '04_br'

    makedirs(path_01)
    makedirs(path_02)
    makedirs(path_03)
    makedirs(path_04)

    # t = time.time()
    for file_inter in dir_files:
        base_file_name = os.path.basename(file_inter)
        img = cv2.imread(input_path + "/" + base_file_name)

        # cropped_01 = img[y1:y2, x1:x2]
        cropped_01 = img[0:512, 0:512]
        cropped_02 = img[0:512, 512:1024]
        cropped_03 = img[512:1024, 0:512]
        cropped_04 = img[512:1024, 512:1024]

        cv2.imshow('cropped_01', cropped_01)
        cv2.imshow('cropped_02', cropped_02)
        cv2.imshow('cropped_03', cropped_03)
        cv2.imshow('cropped_04', cropped_04)
        cv2.imwrite(path_01 + "/" + base_file_name, cropped_01)
        cv2.imwrite(path_02 + "/" + base_file_name, cropped_02)
        cv2.imwrite(path_03 + "/" + base_file_name, cropped_03)
        cv2.imwrite(path_04 + "/" + base_file_name, cropped_04)
        cv2.waitKey(1)

def merge_to_1_and_save_into_folder(input_path, output_path,height, width):
    """

        Top-left (x1, y1)
        Right-bottom (x2, y2)

        """

    dir_files = os.listdir(input_path)
    dir_files.sort()

    path_01 = output_path + '/' + '01_top_left'
    path_02 = output_path + '/' + '02_top_right'
    path_03 = output_path + '/' + '03_bottom_left'
    path_04 = output_path + '/' + '04_bottom_right'

    makedirs(path_01)
    makedirs(path_02)
    makedirs(path_03)
    makedirs(path_04)

    # t = time.time()
    for file_inter in dir_files:
        base_file_name = os.path.basename(file_inter)
        img = cv2.imread(input_path + "/" + base_file_name)

        # cropped_01 = img[y1:y2, x1:x2]
        cropped_01 = img[0:512, 0:512]
        cropped_02 = img[0:512, 512:1024]
        cropped_03 = img[512:1024, 0:512]
        cropped_04 = img[512:1024, 512:1024]

        cv2.imshow('cropped_01', cropped_01)
        cv2.imshow('cropped_02', cropped_02)
        cv2.imshow('cropped_03', cropped_03)
        cv2.imshow('cropped_04', cropped_04)
        cv2.imwrite(path_01 + "/" + base_file_name, cropped_01)
        cv2.imwrite(path_02 + "/" + base_file_name, cropped_02)
        cv2.imwrite(path_03 + "/" + base_file_name, cropped_03)
        cv2.imwrite(path_04 + "/" + base_file_name, cropped_04)
        cv2.waitKey(0)



def separate_img_into_folder(father_folder, sub_folder_01, sub_folder_02, keywork="real_A"):
    dir_files = os.listdir(father_folder)
    dir_files.sort()
    for file_inter in dir_files:
        base_file_name = os.path.basename(file_inter)
        img = cv2.imread(father_folder + "/" + base_file_name)
        print("proccessing: ", base_file_name)
        if base_file_name.find(keywork) > 0:
            # copy
            cv2.imwrite(sub_folder_01 + "/" + base_file_name, img)
            cv2.waitKey(1)
        else:
            # copy
            cv2.imwrite(sub_folder_02 + "/" + base_file_name, img)
            cv2.waitKey(1)


def overlay_offset(path_01, path_02, save_path, x1, y1, x2, y2):
    dir_files = os.listdir(path_01)
    dir_files.sort()
    for file_inter in dir_files:
        base_file_name = os.path.basename(file_inter)
        l_img = cv2.imread(path_01 + "/" + base_file_name)
        s_img = cv2.imread(path_02 + "/" + base_file_name)
        # x_offset = 313
        # y_offset = 147
        # (313, 147, 1324, 916)
        l_img[y1:y2, x1:x2] = s_img
        cv2.imwrite(save_path + "/" + base_file_name, l_img)
        cv2.imshow("l_img", l_img)
        cv2.waitKey(1)


def resize_images(input_path, output_path, width = 1920, height=1080):
    dir_files = os.listdir(input_path)
    dir_files.sort()
    for file_inter in dir_files:
        base_file_name = os.path.basename(file_inter)
        img = cv2.imdecode(np.fromfile(input_path + "/" + base_file_name, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        # img = cv2.imread(input_path + "/" + base_file_name)
        out_img = cv2.resize(img, (width, height))
        cv2.imwrite(output_path + "/" + base_file_name, out_img)
        cv2.imshow("out_img", out_img)
        cv2.waitKey(1)

def resize_images_keep_ration(input_path, output_path, width=1920, height=1080, ratio_base_height=True):
    dir_files = os.listdir(input_path)
    dir_files.sort()
    for file_inter in dir_files:
        base_file_name = os.path.basename(file_inter)
        img = cv2.imdecode(np.fromfile(input_path + "/" + base_file_name, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        # height, width, number of channels in image
        height_ori = img.shape[0]
        width_ori = img.shape[1]
        channels_ori = img.shape[2]

        # img = cv2.imread(input_path + "/" + base_file_name)

        if ratio_base_height:
            width_new = int(height*width_ori/height_ori)
            out_img = cv2.resize(img, (width_new, height))
        else:
            height_new = int(width * height_ori / width_ori)
            out_img = cv2.resize(img, (width, height_new))

        cv2.imwrite(output_path + "/" + base_file_name, out_img)
        cv2.imshow("out_img", out_img)
        cv2.waitKey(1)

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            print("appending: ", fullPath)
            allFiles.append(fullPath)

    return allFiles


def select_file_size(input_path, output_path, min_width = 340, min_height = 340):
    allFiles = getListOfFiles(input_path)
    all_good_files = list()
    for file in allFiles:
        img = cv2.imread(file)
        height, width, channels = img.shape
        print("Processing file: ", file, "  with width x height x channel: ", width, " x ", height, " x ", channels)

        if width < min_width or height < min_height:
            continue
        else:
            all_good_files.append(file)
            base_file_name = os.path.basename(file)
            cv2.imwrite(output_path + "/" + base_file_name, img)


def filter_mean_brightness(input_path, output_path, mean_threshold = 60.0, is_bright = False):
    allFiles = getListOfFiles(input_path)
    all_good_files = list()
    for file in allFiles:
        img = cv2.imread(file)

        mean_value = img.mean()
        print("file: ",file, " has mean_value: ",mean_value)
        cv2.imshow("img", img)
        cv2.waitKey(2)
        if is_bright: # bright images
            if mean_value > mean_threshold:
                all_good_files.append(file)
                base_file_name = os.path.basename(file)
                cv2.imwrite(output_path + "/" + base_file_name, img)
        else: # dark images
            if mean_value < mean_threshold:
                all_good_files.append(file)
                base_file_name = os.path.basename(file)
                cv2.imwrite(output_path + "/" + base_file_name, img)





def get_brightness_value(input_path, mean_threshold):
    brightness = 0
    img = cv2.imread(input_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img,img, mask= mask)

    cv2.imshow('frame',img)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    print("mean: ", img.mean())

    print("brightness: ", brightness)

    # cv2.imshow("img", img)
    cv2.waitKey(0)
    return brightness


def one_click(write_demo = False):

    # path settings
    main_path = "D:/tmp/a/IMG_2999"

    standard_mask_path = main_path + "/standard_mask_pro.png"
    input_path = main_path + "/input"
    masked_path = main_path + "/masked"
    output_path = main_path + "/output/epoch106_20200422174532"
    mask_path = main_path + "/mask"
    demo_video_path = main_path + "/Video_demo_05.mp4"

    ######### processing
    if not write_demo:
        # 01. normalize the mask
        standard_mask = cv2.imread(standard_mask_path)
        standard_mask = normalize_mask(standard_mask)
        cv2.imwrite(standard_mask_path, standard_mask)

        # 02. write standard masks
        write_standard_masks(input_path, mask_path, standard_mask_path)

        # 03. write masked images
        alpha_blending(input_path,
                       masked_path,
                       standard_mask_path)
    else:
        # demo
        demo_video_2(
            masked_path,
            output_path,
            demo_video_path,
            "Input",
            "Output",
            vertical=False,
            fps=15,
            w=512,
            h=512,
            with_arrow=True
        )



if __name__ == '__main__':
    print("### RUNNING IMAGE2VIDEO ###")
    # one_click(True)
    # select_file_size("//NAS2/Kojin/trinh/luxeyeai_englightengan/final_dataset/ExDark", "//NAS2/Kojin/trinh/luxeyeai_englightengan/final_dataset/ExDark_out", 340, 340)

    # get_brightness_value("//NAS2/Kojin/trinh/luxeyeai_englightengan/final_dataset/Dark_A_340/2015_03491.jpg")
    # get_brightness_value("//NAS2/Kojin/trinh/luxeyeai_englightengan/final_dataset/testB/normal00004.png")

    # filter_mean_brightness("//NAS2/Kojin/trinh/luxeyeai_englightengan/final_dataset/Dark_A_340",
    #                        "//NAS2/Kojin/trinh/luxeyeai_englightengan/final_dataset/Dark_A_340_filer_brightness")

    # select_file_size("D:/[DATA]/place2_val_large/val_large", "D:/[DATA]/place2_val_large/val_large_340", 340, 340)
    # resize_images_keep_ration("//NAS2/Kojin/trinh/luxeye_test_data/luxeye_test_unsplash_jpg", "//NAS2/Kojin/trinh/luxeye_test_data/resized", 1920, 1080, True)
    # resize_images("//NAS2/Kojin/trinh/luxeye_test_data/luxeye_test_unsplash_jpg", "//NAS2/Kojin/trinh/luxeye_test_data/resized", 1280, 720)


    # filter_mean_brightness("D:/[DATA]/place2_val_large/val_large_340",
    #                        "D:/[DATA]/place2_val_large/val_large_340_filter_brightness", 80.0, True)

    # images2video(out_video_path="D:/tmp/GOPR1031_new_512_blended_02/170/demo_tokyo/02.Tokyo_resized_02.MOV",
    #              images_folder_path="D:/tmp/GOPR1031_new_512_blended_02/170/demo_tokyo/resize_images",
    #              frame=15,
    #              w=544,
    #              h=512,
    #              )


    #
    # images2video("C:/Users/Trinh/Downloads/60fps.mp4",
    #              "C:/Users/Trinh/Downloads/60fps", 30, 1280, 960)

    # images2video("D:/tmp/GOPR1031_new_512_blended_02/input/input.avi", "D:/tmp/GOPR1031_new_512_blended_02/input", 30, 512, 512)
    # images2video("D:/tmp/GOPR1031_new_512_blended_02/standard_masks/masks.avi", "D:/tmp/GOPR1031_new_512_blended_02/standard_masks", 30, 512, 512)


    # rename("D:/tmp/GOPR1031_new_512_blended_02/standard_masks")



    # alpha_blending("C:/Users/Trinh/Downloads/drive_demo",
    #                "C:/Users/Trinh/Downloads/masked",
    #                "C:/Users/Trinh/Downloads/masks/raindrop_000000.jpg")

    video2video("C:/Users/Trinh/Downloads/_0803/009_230801160000_0100_FF_00_28.n3r",
                "C:/Users/Trinh/Downloads/_0803/009_230801160000_0100_FF_00_28.mp4", 30, 1280, 720)

    # video2video("D:/OneDrive - 株式会社サイバーコア/サイバーコア技術紹介資料_JRW向け/27. 無人レジ_shop.mp4", "D:/OneDrive - 株式会社サイバーコア/サイバーコア技術紹介資料_JRW向け/27. 無人レジ_shop2.mp4", 60, 1920, 1080)

    # write_standard_masks
    # ("D:/107GOPR/GOPR1115_512_01/input", "D:/107GOPR/GOPR1115_512_01/masks_cloned/"
    #                      , "D:/107GOPR/GOPR1115_512_01/standard_mask.jpg")
    #
    # video2images("C:/Users/Trinh/Downloads/0522_Signage_5_24_x264_out2.mp4",
    #            "C:/Users/Trinh/Downloads/0522_Signage_5_24_x264_out2")

    # images2video("D:/PROJECTS/ROLE-raindrop-maker/35fps_22.mp4",
    #              "D:/PROJECTS/ROLE-raindrop-maker/output", 25, 1920, 1080)


    # images2video("C:/Users/Trinh/Downloads/dts_image_classification_dataset_v1/abv.mp4",
    #              "C:/Users/Trinh/Downloads/dts_image_classification_dataset_v1/0_OK", 3, 512, 512)

    # images2imagesOKNG("C:/Users/Trinh/Downloads/dts_image_classification_dataset_v1/0_OK",
    #              "C:/Users/Trinh/Downloads/dts_image_classification_dataset_v1/0", True)

    # fix_video_format("C:/Users/Trinh/Downloads/road_01_out.mp4","C:/Users/Trinh/Downloads",1,30, 1280, 720)
    # video2video("C:/Users/Trinh/Downloads/dts_image_classification_dataset_v1/ObjectDetection.mp4", "C:/Users/Trinh/Downloads/dts_image_classification_dataset_v1/ObjectDetection2.mp4", 3, 512, 512)
    # video2video("C:/Users/Trinh/Downloads/abccc.mp4",
    #             "C:/Users/Trinh/Downloads/TBS_Traffic.mp4",
    #             30, int(1280), int(720))
    # video2video("C:/Users/Trinh/Downloads/combination/scene.MP4", "C:/Users/Trinh/Downloads/combination/scene_nor.mp4", 30, 1920, 1080)


    # crop_video_and_save("C:/Users/Trinh/Downloads/221209_動画撮影/221209_動画撮影/no02.MTS", "C:/Users/Trinh/Downloads/221209_動画撮影/221209_動画撮影/no02_out.mp4",
    #               "C:/Users/Trinh/Downloads/no02_small", 365,580,1440,920)


    # crop_and_save("C:/Users/Trinh/Downloads/no02", "C:/Users/Trinh/Downloads/no02_2", 365,580,1440,920)
    # crop_and_save("C:/Users/Trinh/Downloads/no02", "C:/Users/Trinh/Downloads/no02_2", 0,0,500,1000)
    # crop_and_save_video("C:/Users/Trinh/Downloads/samplemovie.avi", "C:/Users/Trinh/Downloads/label_images", 313,147,1324,916)

    # separate_img_into_folder("D:/densoten/images", "D:/densoten/real_A", "D:/densoten/fake_B", "real_A")

    # overlay_offset("D:/densoten/20200124_100cm_01","D:/densoten/fake_B", "D:/densoten/overlay", 313,147,1324,916)

    # alpha_blending("D:/[DATA]/raindrop_koiwa/input",
    #                "D:/[DATA]/raindrop_koiwa/masked",
    #                "D:/[DATA]/raindrop_koiwa/eye_000900_mask.png")

    # demo_video_2(
    #     "C:/Users/Trinh/Downloads/output_GH011846_Trim3",
    #     "C:/Users/Trinh/Downloads/output_GH011681_Trim3",
    #     "C:/Users/Trinh/Downloads/abcde.mp4",
    #     "Camera 1",
    #     "Camera 2",
    #     vertical=False,
    # )
    # resize_images("//NAS2/Kojin/koseki/DataBackup/Material/静画/雪", "//NAS2/Kojin/trinh/luxeye_test_dataset_full_HD")
    # super_hero.cloning_images("D:/[DATA]/raindrop_koiwa/eye_000900_mask.png", "D:/[DATA]/raindrop_koiwa/mask", 4730)
    # video2images("C:/Users/Trinh/Downloads/03.Traffic_sign_incar.mp4", "C:/Users/Trinh/Downloads/03.Traffic_sign_incar", 1)
    # write_standard_masks("C:/Users/Trinh/Downloads/drive_demo_out/03_bl","C:/Users/Trinh/Downloads/drive_demo_out/03_bl_masks", "C:/Users/Trinh/Downloads/drive_demo_out/03.png")
    # write_standard_masks("C:/Users/Trinh/Downloads/drive_demo_out/04_br","C:/Users/Trinh/Downloads/drive_demo_out/04_br_masks", "C:/Users/Trinh/Downloads/drive_demo_out/04.png")



    # demo_video_2(
    #     # "//UBUNTU170/mnt/ELECOM/DATA/luxeyeai/videos/dark_car/input",
    #     # "//UBUNTU170/mnt/ELECOM/DATA/luxeyeai/videos/dark_car/output",
    #
    #     "C:/Users/Trinh/Downloads/car_crash",
    #     "C:/Users/Trinh/Downloads/output",
    #     "C:/Users/Trinh/Downloads/depth.mp4",
    #     "",
    #     "",
    #
    #     vertical=False,
    #     fps=24,
    #     with_arrow=True,
    #     arrow_path="./demo_supporter/arrow4.png",
    #     with_logo=True,
    #     logo_path="./demo_supporter/logo2.png",
    #     resized=True,
    #     w0=640,
    #     h0=480,
    # )
    # compare_4(
    #     "D:/Google Drive/AwesomeProjects/Olympus_OCR/02.From_Team/2023_05_19/Forte/compare/thick",
    #
    #     "D:/Google Drive/AwesomeProjects/Olympus_OCR/02.From_Team/2023_05_19/Forte/v1_thick",
    #     "D:/Google Drive/AwesomeProjects/Olympus_OCR/02.From_Team/2023_05_19/Forte/v2_thick",
    #     "D:/Google Drive/AwesomeProjects/Olympus_OCR/02.From_Team/2023_05_19/Forte/v3_thick",
    #     "D:/Google Drive/AwesomeProjects/Olympus_OCR/02.From_Team/2023_05_19/Forte/v3_thick",
    #     "v1",
    #     "v2",
    #     "v3",
    #     "v3",
    #     w0=2560,
    #     h0=1440
    # )

    # video2video("D:/DATA/hachinohe/demo_10m.mov", "D:/DATA/hachinohe/demo_10m_out.mp4", 30, int(720*2), int(480*2))
    # video2video("C:/Users/Trinh/Downloads/demo_f3m_out.mp4", "C:/Users/Trinh/Downloads/demo_f3m_out2.mp4", 30, int(960), int(540))
    # demo_video_4(
    #             "C:/Users/Trinh/Downloads/drive_demo_out/01_tl_out",
    #             "C:/Users/Trinh/Downloads/drive_demo_out/02_tr_out",
    #             "C:/Users/Trinh/Downloads/drive_demo_out/03_bl_out",
    #             "C:/Users/Trinh/Downloads/drive_demo_out/04_br_out",
    #             "C:/Users/Trinh/Downloads/drive_demo_out/01_top_left.mp4",
    #
    #             "",
    #             "",
    #             "",
    #             "",
    #             fps=15,
    #             with_arrow=False,
    #             arrow_path="./demo_supporter/arrow4.png",
    #             with_logo=False,
    #             logo_path="./demo_supporter/logo.png",
    #             resized=False,
    #             w0=512,
    #             h0=512,
    #              )

    # video2images(
    #     "C:/Users/Trinh/Downloads/_0803/008_230803182959_0101_FF_00_28.n3r",
    #     "C:/Users/Trinh/Downloads/_0803/tmp")

    # demo_video_abnormal_4(
    #             "C:/Users/Trinh/Downloads/FPGA/Heaven/MVI_0161",
    #             "C:/Users/Trinh/Downloads/FPGA/Heaven/MVI_0158",
    #             "D:/[PROJECTS]/opencv-object-tracking/MVI_0161",
    #             "D:/[PROJECTS]/opencv-object-tracking/MVI_0158_with_box",
    #             "C:/Users/Trinh/Downloads/final.mp4",
    #
    #             "Normal",
    #             "Normal",
    #             "",
    #             "",
    #             fps=20,
    #             with_arrow=False,
    #             arrow_path="./demo_supporter/arrow4.png",
    #             with_logo=False,
    #             logo_path="./demo_supporter/logo.png",
    #             resized=True,
    #             w0=960,
    #             h0=540,
    #              )
    # images2imagesOKNG("C:/Users/Trinh/Downloads/abc/images/",
    #                   "C:/Users/Trinh/Downloads/abc/images/",
    #                   True)
    # copy_D1_D2("C:/Users/Trinh/Downloads/data/20210311","C:/Users/Trinh/Downloads/data/20210311_dest" )

    # write_white_black("D:/107GOPR/GOPR1107_01/mask_02.jpg", "D:/107GOPR/GOPR1107_01/mask_02_wb.jpg")

    # # crop_to_4_and_save_into_folder("C:/Users/Trinh/Downloads/drive_demo", "C:/Users/Trinh/Downloads/drive_demo_out", 1024, 1024).









