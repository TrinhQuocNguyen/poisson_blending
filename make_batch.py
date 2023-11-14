#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os

import super_hero


def images2video(out_video_path, images_folder_path, frame=60, w=512 * 2, h=512):
    dir_files = os.listdir(images_folder_path)
    dir_files.sort()

    # write video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(out_video_path, fourcc, frame, (w, h))

    for file_inter in dir_files:
        base_file_name = os.path.basename(file_inter)

        input_image = cv2.imread(images_folder_path + "/" + base_file_name)
        # input_image = cv2.resize(input_image, (512, 512))

        print("PROCESSING")

        cv2.imshow("input_image", input_image)
        cv2.waitKey(1)

        # save a frame
        out.write(input_image)


def video2images(input_video_path, output_images_path):
    cap = cv2.VideoCapture(input_video_path)
    n = 0
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print("Can not read this frame!!")
            continue

        # if n < 60* 8 * 60 - 200:
        #     print("Skipped!")
        #     n += 1
        #     continue

        # cv2.imshow("img", img)
        cv2.waitKey(1)
        print("PROCESSING: " + output_images_path + "/" + '{0:06}'.format(n) + '.jpg')
        # cv2.imwrite(output_images_path + "/" +'{0:06}'.format(n)+'.jpg', img)

        # img = cv2.resize(img, (512, 512))
        cv2.imwrite(output_images_path + "/" + "raindrop_" + '{0:06}'.format(n) + '.jpg', img)
        cv2.imshow("img", img)

        n += 1


def video2video(input_video_path, out_video_path, frame=60, w=1920, h=1080):
    cap = cv2.VideoCapture(input_video_path)
    n = 0
    # write video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(out_video_path, fourcc, frame, (w, h))
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print("Can not read this frame!!")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        print("PROCESSING frame: " + str(n))
        # cv2.imwrite(output_images_path + "/" +'{0:06}'.format(n)+'.jpg', img)

        # img = cv2.resize(img, (512, 512))
        # cv2.imwrite(output_images_path + "/" + "raindrop_" +'{0:06}'.format(n)+'.jpg', img)

        cv2.waitKey(1)
        cv2.imshow("img", img)
        # save a frame
        out.write(img)
        n += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def make_batch(input_path="D:/tmp/GOPR1031_new_512_blended_02/input",
               output_path="D:/tmp/GOPR1031_new_512_blended_02/170/batches", number_of_frames=20, number_of_overlap=10):
    dir_files = os.listdir(input_path)
    dir_files.sort()
    image_count = 0
    batch_count = 0
    length = len(dir_files)
    number_of_batches = int(length / number_of_overlap) + 1  # 126
    start_point = number_of_frames - number_of_overlap  #
    print(number_of_batches)
    from_frame = 0
    # making sub-folders
    for b in range(0, number_of_batches):
        dirName = output_path + "/" + 'batch_{0:06}'.format(b * start_point) + "/" + "JPEGImages"
        # Create target Directory if don't exist
        if not os.path.exists(dirName):
            os.mkdir(dirName)
            print("Directory ", dirName, " Created ")

        # copy into path: start_point -> start_point + number_of_frames
        for f in range(b * start_point, b * start_point + number_of_frames):
            image_01 = cv2.imread(input_path + "/" + 'raindrop_{0:06}'.format(f) + ".jpg")

            cv2.imwrite(dirName + "/" + 'raindrop_{0:06}'.format(f) + ".jpg", image_01)
            cv2.imshow("image_01", image_01)
            cv2.waitKey(1)


def re_batch(input_path="D:/tmp/GOPR1031_new_512_blended_02/170/batches_output",
             output_path="D:/tmp/GOPR1031_new_512_blended_02/170/re_batch", number_of_frames=20, number_of_overlap=10):


    start_point = number_of_frames - number_of_overlap  #
    for b in range(0, 125):
        dirName = input_path + "/" + 'batch_{0:06}'.format(b * start_point) + "/result_0000"
        dir_files = os.listdir(dirName)
        dir_files.sort()
        count = 0

        for file_inter in dir_files:
            base_file_name = os.path.basename(file_inter)
            input_image = cv2.imread(dirName + "/" + base_file_name)

            cv2.imwrite(output_path + '/raindrop_{0:06}'.format(b*start_point + count) + ".jpg", input_image)
            print(b*start_point + count)
            count +=1
            cv2.imshow("input_image", input_image)
            cv2.waitKey(1)

    # # copy images into batches
    # for file_inter in dir_files:
    #     base_file_name = os.path.basename(file_inter)
    #     image_01 = cv2.imread(input_path + "/" + base_file_name)
    #
    #     if image_count % number_of_frames == 0:
    #         cv2.imwrite(output_path + "/batch_{0:06}".format(image_count) + "/" + base_file_name, image_01)
    #         batch_count += 1
    #
    #     image_count += 1
    #
    #     cv2.imshow("image_01", image_01)
    #     cv2.waitKey(1)


def write_cof():
    file1 = open("myfile.txt", "a+")


    for i in range(0, 126):
        f = i*10
        L1 = ["        },\n"]
        L2 = ["        {\n"]
        L3 = ["            \"type\": \"MaskedFrameDataLoader\",\n"]
        L4 = ["            \"args\":{\n"]
        L5 = ["                \"name\": \"batch_" + '{0:06}'.format(f) + "\",\n"]
        L6 = ["                \"root_videos_dir\": \"../dataset/batches/batch_" + '{0:06}'.format(f) + "/\",\n"]
        L62 = ["                \"root_masks_dir\": \"../dataset/raindrop_mask/\",\n"]
        L63 = ["                \"root_outputs_dir\": \"../batch_" + '{0:06}'.format(f) + "\",\n"]
        L7 = ["                \"dataset_args\": {\n"]
        L8 = ["                    \"type\": \"video\",\n"]
        L9 = ["                    \"w\": 512,\n"]
        L10 = ["                    \"h\": 512,\n"]
        L11 = ["                    \"sample_length\": 30,\n"]
        L12 = ["                    \"random_sample\": false,\n"]
        L13 = ["                    \"random_sample_mask\": false,\n"]
        L14 = ["                    \"mask_type\": \"fg\",\n"]
        L15 = ["                    \"mask_dilation\": 0\n"]
        L16 = ["                },\n"]
        L17 = ["                \"batch_size\": 1,\n"]
        L18 = ["                \"shuffle\": false,\n"]
        L19 = ["                \"validation_split\": 0.0,\n"]
        L20 = ["                \"num_workers\": 2\n"]
        L21 = ["            }\n"]

        # \n is placed to indicate EOL (End of Line)
        file1.writelines(L1)
        file1.writelines(L2)
        file1.writelines(L3)
        file1.writelines(L4)
        file1.writelines(L5)
        file1.writelines(L6)
        file1.writelines(L62)
        file1.writelines(L63)
        file1.writelines(L7)
        file1.writelines(L8)
        file1.writelines(L9)
        file1.writelines(L10)
        file1.writelines(L11)
        file1.writelines(L12)
        file1.writelines(L13)
        file1.writelines(L14)
        file1.writelines(L15)
        file1.writelines(L16)
        file1.writelines(L17)
        file1.writelines(L18)
        file1.writelines(L19)
        file1.writelines(L20)
        file1.writelines(L21)



if __name__ == '__main__':
    print("images2video")
    # write_cof()

    # make_batch(
    #     "D:/tmp/GOPR1031_new_512_blended_02/input",
    #     "D:/tmp/GOPR1031_new_512_blended_02/170/batches",
    #     20,
    #     10,
    # )

    re_batch(
        "D:/tmp/GOPR1031_new_512_blended_02/170/batches_output",
        "D:/tmp/GOPR1031_new_512_blended_02/170/re_batch",
        20,
        10,
    )
