import cv2
import numpy as np
import time

# cap = cv2.VideoCapture(0)
#
# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#
#     # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Display the resulting frame
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("D:/103GOPR_ori/GOPR1043.MP4")
time.sleep(2)

fps1 = 10

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (512, 512))

        font = cv2.FONT_HERSHEY_SIMPLEX

        fps1 += 1
        cv2.putText(frame, 'Fps: ' + str(fps1), (50, 50), font, 1, (255,255,255), 2,cv2.LINE_AA)

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.waitKey(16)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        fps = cap.get(cv2.CAP_PROP_FPS)
        print(fps)

cap.release()
cv2.destroyAllWindows()



