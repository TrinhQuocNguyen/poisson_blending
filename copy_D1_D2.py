

import os.path
from os import path
import shutil


def copy_D1_D2(path_in="", path_out=""):

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


if __name__ == '__main__':
    print("### RUNNING COPY ###")
    copy_D1_D2("C:/Users/Trinh/Downloads/data/20210311","C:/Users/Trinh/Downloads/data/20210311_dest")