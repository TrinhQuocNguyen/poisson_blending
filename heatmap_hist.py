import numpy as np
import cv2
import matplotlib.pyplot as plt
# use it if you wonna write video or ffmpeg 
# from skvideo.io import FFmpegWriter 

start = 1
duration = 10
fps = '30'
cap = cv2.VideoCapture(0)
outfile = 'heatmap.mp4'


while True:
    try:
        # _, f = cap.read()
        f = cv2.imread("D:/[PROJECTS]/temp/New folder/img_5_2018-11-26-17-32-57.png")

        f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        f = cv2.GaussianBlur(f, (11, 11), 2, 2)
        cnt = 0
        res = 0.05*f
        res = res.astype(np.float64)
        break
    except:
        print('s')


fgbg = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold=100,
                                          detectShadows=True)



import matplotlib.pyplot as plt
import numpy as np

# a = cv2.imread("rain/o_raindrop0351.jpg")
a = cv2.imread("D:/102GOPR_noise/GOPR0537/raindrop0066.jpg")
a = cv2.resize(a, (512, 512))

a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
plt.imshow(a, cmap='hot', interpolation='nearest')
plt.show()


# https://stackoverflow.com/questions/47899253/how-to-plot-3d-histogram-of-an-image-in-opencv/47899963#47899963
# img = cv2.imread("rain/o_raindrop0351.jpg")
img = cv2.imread("D:/102GOPR_noise/GOPR0537/raindrop0066.jpg")
img = cv2.resize(a, (512, 512))

b,g,r = cv2.split(img)
fig = plt.figure(figsize=(8,4))

ax = fig.add_subplot(121)
ax.imshow(img[...,::-1])

ax = fig.add_subplot(122)
for x, c in zip([b,g,r], ["b", "g", "r"]):
    xs = np.arange(256)
    ys = cv2.calcHist([x], [0], None, [256], [0,256])
    ax.plot(xs, ys.ravel(), color=c)

ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import cv2

# img = cv2.imread("rain/o_raindrop0351.jpg")
img = cv2.imread("D:/102GOPR_noise/GOPR0537/raindrop0066.jpg")
img = cv2.resize(a, (512, 512))



hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv)
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
for x, c, z in zip([h,s,v], ['r', 'g', 'b'], [30, 20, 10]):
    xs = np.arange(256)
    ys = cv2.calcHist([x], [0], None, [256], [0,256])
    cs = [c] * len(xs)
    cs[0] = 'c'
    ax.bar(xs, ys.ravel(), zs=z, zdir='y', color=cs, alpha=0.8)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()


# writer = FFmpegWriter(outfile, outputdict={'-r': fps})
#writer = FFmpegWriter(outfile)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
cnt = 0
sec = 0
while True:
    # if sec == duration: break
    cnt += 1
    if cnt % int(fps) == 0:
        print(sec)
        sec += 1
    # ret, frame = cap.read()
    # if not ret: break
    frame = cv2.imread("D:/[PROJECTS]/temp/New folder/img_5_2018-11-26-17-32-57.png")


    fgmask = fgbg.apply(frame, None, 0.01)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # if cnt == 30: res
    gray = cv2.GaussianBlur(gray, (11, 11), 2, 2)
    gray = gray.astype(np.float64)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = fgmask.astype(np.float64)
    res += (40 * fgmask + gray) * 0.01
    res_show = res / res.max()
    res_show = np.floor(res_show * 255)
    res_show = res_show.astype(np.uint8)
    res_show = cv2.applyColorMap(res_show, cv2.COLORMAP_JET)
    cv2.imshow('s', res_show)
    # if sec < start: continue
    #    try:
    #        writer.writeFrame(res_show)
    #    except:
    #        writer.close()
    #        break


    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

#writer.close()
cap.release()
cv2.destroyAllWindows()