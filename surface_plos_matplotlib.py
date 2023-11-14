
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os
import scipy.misc

def show_3d_surface():
    # generate some sample data
    import scipy.misc
    lena = cv2.imread("./data/score.png", 0)

    # downscaling has a "smoothing" effect
    # lena = cv2.resize(lena, (128,128))

    # create the x and y coordinate arrays (here we just use pixel indices)
    xx, yy = np.mgrid[0:lena.shape[0], 0:lena.shape[1]]

    # create the figure
    fig = plt.figure(figsize=[5.12, 5.12])
    ax = fig.gca(projection='3d')

    ax.set_xlabel('bb width', fontsize=10, rotation=0, color = (0.3,0.0,0.9))
    ax.set_ylabel('bb height', fontsize=10, rotation=0, color = (0.1,0.5,1.0))
    ax.set_zlabel('intensity', fontsize=10, rotation=0, color = (1.0,0.0,1.0))

    ax.plot_surface(xx, yy, lena ,rstride=1, cstride=1, cmap=plt.cm.jet,
                    linewidth=0)

    # show it
    plt.title('Spatial Reliability Map  \n', fontsize = 20, color = (0.0,1.0,1.0))
    plt.show()
    # fig.savefig('./data/score_3d.png')

def show_3d_surface_folder(images_folder_path, output_path):
    dir_files = os.listdir(images_folder_path)
    dir_files.sort()
    # generate some sample data
    for file_inter in dir_files:

        base_file_name = os.path.basename(file_inter)

        lena = cv2.imread(images_folder_path + "/" +base_file_name, 0)
        # input_image = cv2.resize(input_image, (512, 512))

        # cv2.imshow("input_image", lena)
        # cv2.waitKey(1)

        # downscaling has a "smoothing" effect

        # create the x and y coordinate arrays (here we just use pixel indices)
        xx, yy = np.mgrid[0:lena.shape[0], 0:lena.shape[1]]

        # create the figure
        fig = plt.figure(figsize=[5.12, 5.12])
        ax = fig.gca(projection='3d')

        ax.set_zlim([0, 255])

        ax.set_xlabel('bb width', fontsize=10, rotation=0, color = (0.3,0.0,0.9))
        ax.set_ylabel('bb height', fontsize=10, rotation=0, color = (0.1,0.5,1.0))
        ax.set_zlabel('intensity', fontsize=10, rotation=0, color = (1.0,0.0,1.0))

        ax.plot_surface(xx, yy, lena ,rstride=1, cstride=1, cmap=plt.cm.jet,
                        linewidth=0)

        # show it
        plt.title('Correlation Response \n', fontsize = 20, color = (0.0,1.0,1.0))
        # plt.show()
        print("Processing: " + base_file_name)
        fig.savefig(output_path + '/' + base_file_name)
        plt.close()

if __name__ == '__main__':
    # show_3d_surface()
    show_3d_surface_folder("D:/[PROJECTS]/pyCFTrackers/dataset/test/AA/score", "D:/[PROJECTS]/pyCFTrackers/dataset/test/AA/correlation")


