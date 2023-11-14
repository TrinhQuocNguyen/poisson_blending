import numpy as np
import cv2

# basic settings
settings = {
    'sensitivity': 'high',  # motion sensitivity
    'skip_frames': 1,  # how many frames to skip: 1 = process every frame
}

# extended settings for low sensitivity
if settings['sensitivity'] == 'low':
    settings['gaussian_size'] = (9, 9)
    settings['difference_threshold'] = 0.05  # between 0 (track every small change) to 1 (track nothing)
    settings['heatmap_threshold'] = 0.5  # between 0 (track every small change) to 1 (track nothing)
    settings['heatmap_decay'] = 30  # number of frames the heatmap decays (goes cold without movement)

# extended settings for mid sensitivity
elif settings['sensitivity'] == 'mid':
    settings['gaussian_size'] = (7, 7)
    settings['difference_threshold'] = 0.01  # between 0 (track every small change) to 1 (track nothing)
    settings['heatmap_threshold'] = 0.4  # between 0 (track every small change) to 1 (track nothing)
    settings['heatmap_decay'] = 30  # number of frames the heatmap decays (goes cold without movement)

# extended settings for high sensitivity
elif settings['sensitivity'] == 'high':
    settings['gaussian_size'] = None  # unused!
    settings['difference_threshold'] = 0.00  # between 0 (track every small change) to 1 (track nothing)
    settings['heatmap_threshold'] = 0.25  # between 0 (track every small change) to 1 (track nothing)
    settings['heatmap_decay'] = 10  # number of frames the heatmap decays (goes cold without movement)

else:
    raise Exception('Error - Invalid settings')


# -----------------------------------------------------------------------------
def get_luminance(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    frame = frame[:, :, 0].astype('float32')
    frame = frame / 255.0

    return frame


# -----------------------------------------------------------------------------
def generate_preview(frame, heatmap_bin):
    # mask to show only red colors
    mask = ((1 - heatmap_bin) * 255).astype('uint8')

    # red color :3
    red = np.zeros_like(frame)
    red[:, :, 2] = 255

    # now mask the red color so only the masked part is visible
    red = cv2.bitwise_and(red, red, mask=mask)

    # mask the preview so we get the same region too
    prev = cv2.bitwise_and(frame, frame, mask=mask)

    # combine both to create a reddish tint
    red = (0.6 * red + 0.4 * prev).astype('uint8')

    # invert the mask and apply it to the preview
    frame = cv2.bitwise_and(frame, frame, mask=(255 - mask))

    # now combine red color and original image
    frame = cv2.bitwise_or(frame, red)

    return frame


# -----------------------------------------------------------------------------
def process_video(filename):
    # visual preview
    cv2.namedWindow("WINDOW_1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("WINDOW_2", cv2.WINDOW_NORMAL)
    vid = cv2.VideoCapture(filename)

    # save output to file
    fps = vid.get(cv2.CAP_PROP_FPS)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('H', '2', '6', '4')
    output_1 = cv2.VideoWriter("output_1.mp4", fourcc, 30, (width, height))
    output_2 = cv2.VideoWriter("output_2.mp4", fourcc, 30, (width, height))

    # read first frame
    ret, last_frame = vid.read()
    last_frame = get_luminance(last_frame)

    # prepare heatmap for movement detection
    heatmap = np.zeros_like(last_frame)
    heatmap = heatmap.astype('float32')

    # process frames
    frame_count = 0
    while True:

        ret, frame = vid.read()
        if not ret:
            break

        # display original frame
        cv2.imshow("WINDOW_1", frame)
        output_1.write(frame)

        # save a copy
        preview = frame.copy()

        # process every nth frame
        frame_count += 1
        if frame_count % settings['skip_frames'] == 0:

            frame = get_luminance(frame)
            if settings['gaussian_size'] is not None:
                frame = cv2.GaussianBlur(frame, settings['gaussian_size'], 0)

            diff = cv2.absdiff(last_frame, frame)
            last_frame = frame

            # there must be at least this much change to set it as "changed"
            _, diff = cv2.threshold(diff, settings['difference_threshold'], 1, cv2.THRESH_TOZERO)

            # normalize 0~1
            heatmap += diff * 4

            # decay            
            heatmap -= 1 / settings['heatmap_decay']

            # clamp
            heatmap = np.clip(heatmap, 0, 1)
            _, heatmap_bin = cv2.threshold(heatmap, settings['heatmap_threshold'], 1, cv2.THRESH_BINARY_INV)

        # generate preview, save and display
        preview = generate_preview(preview, heatmap_bin)
        output_2.write(preview)

        cv2.imshow("WINDOW_2", preview)
        key = cv2.waitKey(1)


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    process_video("data/video.mp4")
