import os 



# paths
ffmped_path = "D:/[PROJECTS]/tutorials/learn_opencv/ffmpeg/bin/ffmpeg"
videos_path = "D:/102GOPR_ori/"
extracted_path = "D:/102GOPR_ori/"
file_end_with = '.MP4'



file_names = os.listdir(videos_path)
file_names.sort()
for i in file_names:
    if i.endswith('.MP4'):
        # print(i)
        s = "D:/[PROJECTS]/tutorials/learn_opencv/ffmpeg/bin/ffmpeg -ss 00:00:00 -t 00:20:40 -i " + videos_path \
        + i \
        + " -qmin 1 -q:v 1 -r 30.0 "+  extracted_path \
        + i[:-4] \
        + "/" + i[4:-4]\
        + "_raindrop%6d.jpg"
        print(s)
        f = open(extracted_path + "www.sh", "a")
        f.write(s + "\n")

        if not os.path.exists(extracted_path +i[:-4]):

            print(extracted_path +i[:-4])
            os.makedirs(extracted_path +i[:-4])

# print(os.listdir('D:/tmp/janken_data'))

