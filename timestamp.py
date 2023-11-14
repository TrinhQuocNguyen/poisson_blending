import cv2

video = ""
camera = cv2.VideoCapture(video)
frame_num = 0
while True:
    ret, im = camera.read()
    time_calculate_speed = camera.get(cv2.CAP_PROP_POS_MSEC) / 1000
    if ret != True:
        print("Done of video at frame {}. Continue..".format(frame_num))
        break

    frame_num+=1