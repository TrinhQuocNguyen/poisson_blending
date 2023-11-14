import numpy as np
import cv2
import os

from airium import Airium
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

        print("PROCESSING: ", base_file_name)
        cv2.imshow("input_image", input_image)
        # cv2.imwrite(out_video_path + '/'+ base_file_name, output_image)
        cv2.waitKey(1)

        # save a frame
        out.write(input_image)

def write_html(left_folder, right_folder):
    dir_files = os.listdir(left_folder)
    dir_files.sort()

    base_file_name_list = []

    for file_inter in dir_files:
        base_file_name = os.path.basename(file_inter)
        base_file_name_list.append(file_inter)

    # for f1 in base_file_name_list:
    #     print(f1)

    a = Airium()

    a('<!DOCTYPE html>')
    with a.html(lang="pl"):
        with a.head():
            a.meta(charset="utf-8")
            a.title(_t="Cybercore")

        with a.body():
            with a.h1(id="id23409231", klass='main_header', style="color: red; text-align:center"):
                a("Result Comparision: CC-CompactAI and YoloV4")
            with a.table(id='table_372', border="1px solid"):
                with a.tr(klass='header_row'):
                    a.th(_t='File Name', style="background-color : yellow; color:red")
                    a.th(_t='CC-CompactAI', style="background-color : yellow; color:red")
                    a.th(_t='YoloV4', style="background-color : yellow; color:red")
                for i in range(0, len(base_file_name_list) - 1):
                    with a.tr():
                        a.td(_t=str(i) + ". " + base_file_name_list[i])
                        with a.td():
                            a.img(src='./cc-compact/' + str(base_file_name_list[i]), alt='alt text', width='680')
                        with a.td():
                            a.img(src='./yolov4/' + str(base_file_name_list[i]), alt='alt text', width='680')
                    #
                    # with a.tr():
                    #     a.td(_t='2.')
                    #     a.td(_t='Roland', id='rmd')
                    #     a.td(_t='Mendel')
            # for i in range(0, len(base_file_name_list)-1):
            #     with a.div():
            #         with a.h3(klass='body_text'):
            #             a(str(i) + ". " + base_file_name_list[i])
            #         a.img(src='./cc-compact/' + str(base_file_name_list[i]), alt='alt text', width='680')
            #         a.img(src='./yolov4/' + str(base_file_name_list[i]), alt='alt text', width='680')


    html = str(a)  # casting to string extracts the value

    print(html)
    with open('//NAS5/Level4/4100_顧客_準最高機密/NTTCom/20211020_CrowdHuman_compare/open_me.html', 'w') as f:
        f.write(str(html))

def generate_html(left_folder, right_folder):
    dir_files = os.listdir(left_folder)
    dir_files.sort()

    for file_inter in dir_files:
        base_file_name = os.path.basename(file_inter)
        print(left_folder + "/"+ base_file_name)

        # input_image = cv2.imread(images_folder_path + "/" + base_file_name)
        # input_image = cv2.imdecode(np.fromfile(images_folder_path + "/" + base_file_name, dtype=np.uint8), cv2.IMREAD_UNCHANGED)



if __name__ == '__main__':
    write_html("//NAS5/Level4/4100_顧客_準最高機密/NTTCom/20211020_CrowdHuman_compare/cc-compact", "cc-yolov4")
    # generate_html("//NAS5/Level4/4100_顧客_準最高機密/NTTCom/20211020_CrowdHuman_compare/cc-compact", "cc-yolov4")
    cv2.waitKey(0)
