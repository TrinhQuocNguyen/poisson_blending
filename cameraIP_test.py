import urllib.request
import cv2
import numpy as np

# PARAMETERS
USER_NAME = "taiyoyuden"
PASSWORD = "Taiyoyuden1234"
IP_ADDRESS = "192.168.111.13"
FRAME_RATE = "30"
WIDTH = "1920"
HEIGHT = "1080"


# path = r"http://" + USER_NAME + ":" + PASSWORD + "@" + IP_ADDRESS + "/cgi-bin/mjpeg?framerate=" + FRAME_RATE + "&resolution=" + WIDTH + "x" + HEIGHT


# path = r'http://taiyoyuden:Taiyoyuden1234@192.168.111.13/cgi-bin/mjpeg?framerate=30&resolution=1920x1080'
# path = "http://taiyoyuden:Taiyoyuden1234@192.168.111.13/cgi-bin/mjpeg?framerate=30&resolution=1920x1080"
path = "http://taiyoyuden:Taiyoyuden1234@192.168.111.13/cgi-bin/mjpeg"

print(path)
print(type(path))

cap = cv2.VideoCapture(path)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    height, width, channels = frame.shape
    print("width: ", width)
    print("height: ", height)

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()