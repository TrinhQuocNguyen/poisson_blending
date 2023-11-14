import cv2

if __name__ == '__main__':
    vidPath = 'D:/DATA/hachinohe/0522_Signage_5_24_x264.mp4'
    shotsPath = 'D:/DATA/hachinohe/chucks/%d.mp4' # output path (must be avi, otherwize choose other codecs)
    segRange = [(0,447465),(447466,894930),(894931,1342398), (1342399, 1789861-1)] # a list of starting/ending frame indices pairs

    cap = cv2.VideoCapture(vidPath)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(n_frames)

    # fourcc = int(cv2.VideoWriter_fourcc('X','V','I','D')) # XVID codecs
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    for idx,(begFidx,endFidx) in enumerate(segRange):
        print("writing: ", idx)
        writer = cv2.VideoWriter(shotsPath%idx,fourcc,fps,size)
        cap.set(cv2.CAP_PROP_POS_FRAMES,begFidx)
        ret = True # has frame returned
        while(cap.isOpened() and ret and writer.isOpened()):
            ret, frame = cap.read()
            frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
            if frame_number < endFidx:
                writer.write(frame)
            else:
                break
        writer.release()