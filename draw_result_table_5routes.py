import numpy as np
import cv2


def draw_result_table(img = '', current_frame ='', date_time='', minute_period='',  ttruck_1='', large_1='', small_1='', ttruck_2='', large_2='', small_2='', ttruck_3='', large_3='', small_3=''
                      , ttruck_4='', large_4='', small_4='', ttruck_5='', large_5='', small_5=''):
    # Create a black image
    dest = img.copy()

    # Draw a diagonal blue line with thickness of 5 px

    # Background
    bg_top_left_x = 5
    bg_top_left_y = 5

    bg_bottom_right_x = 650
    bg_bottom_right_y = 420
    bg_width = bg_bottom_right_x - bg_top_left_x
    bg_height = bg_bottom_right_y - bg_top_left_y
    padding = 5
    time_padding_y = 30

    content_ncol = 5
    content_nrow = 16

    # color panel (bgr)
    bg_color = (224, 224, 235)
    time_color = (0, 0, 180)
    frame_color = (0, 0, 0)
    content_color = (0, 200, 0)

    ########### DRAW ###########
    dest = cv2.rectangle(dest, (bg_top_left_x, bg_top_left_y), (bg_bottom_right_x, bg_bottom_right_y), bg_color, -1)

    # time text
    cv2.putText(dest, "Current Frame: ", (bg_top_left_x + padding * 2, time_padding_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                time_color, 2)
    cv2.putText(dest, str(current_frame), (bg_top_left_x + padding * 2 + 200, time_padding_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                time_color, 2)

    cv2.putText(dest, "Date Time: ", (bg_top_left_x + padding * 2, time_padding_y * 2 + padding),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, time_color, 2)
    cv2.putText(dest, date_time, (bg_top_left_x + padding * 2 + 200, time_padding_y * 2 + padding),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, time_color, 2)

    cv2.putText(dest, "Minute Period: ", (bg_top_left_x + padding * 2, time_padding_y * 3 + padding * 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, time_color, 2)
    cv2.putText(dest, minute_period, (bg_top_left_x + padding * 2 + 200, time_padding_y * 3 + padding * 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, time_color, 2)

    # Result table
    content_top_left_x = bg_top_left_x + padding * 2
    content_top_left_y = time_padding_y * 3 + padding * 4

    content_bottom_right_x = bg_width - padding
    content_bottom_right_y = bg_height - padding

    content_width = content_bottom_right_x - content_top_left_x
    content_height = content_bottom_right_y - content_top_left_y

    content_row_padding = int(content_height / 16)
    content_col_padding = int(content_width / 5)

    print("content_top_left_x, content_top_left_y: ", content_top_left_x, content_top_left_y)
    print("content_bottom_right_x, content_bottom_right_y:", content_bottom_right_x, content_bottom_right_y)

    print("content_width: ", content_width)
    print("content_height: ", content_height)

    dest = cv2.rectangle(dest, (content_top_left_x, content_top_left_y),
                         (content_bottom_right_x, content_bottom_right_y), frame_color, 1)

    # Draw colums
    dest = cv2.rectangle(dest, (content_top_left_x, content_top_left_y),
                         (content_bottom_right_x - content_col_padding * 4, content_bottom_right_y), frame_color, 1)
    dest = cv2.rectangle(dest, (content_top_left_x, content_top_left_y),
                         (content_bottom_right_x - content_col_padding * 2, content_bottom_right_y), frame_color, 1)

    # Draw rows
    dest = cv2.rectangle(dest, (content_top_left_x, content_top_left_y),
                         (content_bottom_right_x, content_bottom_right_y - content_row_padding * 15), frame_color, 1)
    dest = cv2.rectangle(dest, (content_top_left_x, content_top_left_y),
                         (content_bottom_right_x, content_bottom_right_y - content_row_padding * 12), frame_color, 1)
    dest = cv2.rectangle(dest, (content_top_left_x, content_top_left_y),
                         (content_bottom_right_x, content_bottom_right_y - content_row_padding * 9), frame_color, 1)
    dest = cv2.rectangle(dest, (content_top_left_x, content_top_left_y),
                         (content_bottom_right_x, content_bottom_right_y - content_row_padding * 6), frame_color, 1)
    dest = cv2.rectangle(dest, (content_top_left_x, content_top_left_y),
                         (content_bottom_right_x, content_bottom_right_y - content_row_padding * 3), frame_color, 1)


    # Draw small rows
    dest = cv2.rectangle(dest, (content_top_left_x + content_col_padding, content_top_left_y),
                         (content_bottom_right_x, content_bottom_right_y - content_row_padding * 14), frame_color, 1)
    dest = cv2.rectangle(dest, (content_top_left_x + content_col_padding, content_top_left_y),
                         (content_bottom_right_x, content_bottom_right_y - content_row_padding * 13), frame_color, 1)
    dest = cv2.rectangle(dest, (content_top_left_x + content_col_padding, content_top_left_y),
                         (content_bottom_right_x, content_bottom_right_y - content_row_padding * 11), frame_color, 1)
    dest = cv2.rectangle(dest, (content_top_left_x + content_col_padding, content_top_left_y),
                         (content_bottom_right_x, content_bottom_right_y - content_row_padding * 10), frame_color, 1)
    dest = cv2.rectangle(dest, (content_top_left_x + content_col_padding, content_top_left_y),
                         (content_bottom_right_x, content_bottom_right_y - content_row_padding * 8), frame_color, 1)
    dest = cv2.rectangle(dest, (content_top_left_x + content_col_padding, content_top_left_y),
                         (content_bottom_right_x, content_bottom_right_y - content_row_padding * 7), frame_color, 1)
    dest = cv2.rectangle(dest, (content_top_left_x + content_col_padding, content_top_left_y),
                         (content_bottom_right_x, content_bottom_right_y - content_row_padding * 5), frame_color, 1)
    dest = cv2.rectangle(dest, (content_top_left_x + content_col_padding, content_top_left_y),
                         (content_bottom_right_x, content_bottom_right_y - content_row_padding * 4), frame_color, 1)
    dest = cv2.rectangle(dest, (content_top_left_x + content_col_padding, content_top_left_y),
                         (content_bottom_right_x, content_bottom_right_y - content_row_padding * 2), frame_color, 1)
    dest = cv2.rectangle(dest, (content_top_left_x + content_col_padding, content_top_left_y),
                         (content_bottom_right_x, content_bottom_right_y - content_row_padding * 1), frame_color, 1)

    # Draw texts
    # Draw the headers
    cv2.putText(dest, "Route", (content_top_left_x + padding * 6, content_top_left_y + padding * 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)
    cv2.putText(dest, "Vehicle Type",
                (content_top_left_x + content_col_padding + padding * 6, content_top_left_y + padding * 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)
    cv2.putText(dest, "Number Of Vehicles",
                (content_top_left_x + content_col_padding * 3 + padding * 6, content_top_left_y + padding * 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)
    # Draw the main routes
    cv2.putText(dest, "1",
                (content_top_left_x + padding * 10, content_top_left_y + content_row_padding * 2 + padding * 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)
    cv2.putText(dest, "2",
                (content_top_left_x + padding * 10, content_top_left_y + content_row_padding * 5 + padding * 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)
    cv2.putText(dest, "3",
                (content_top_left_x + padding * 10, content_top_left_y + content_row_padding * 8 + padding * 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)
    cv2.putText(dest, "4",
                (content_top_left_x + padding * 10, content_top_left_y + content_row_padding * 11 + padding * 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)
    cv2.putText(dest, "5",
                (content_top_left_x + padding * 10, content_top_left_y + content_row_padding * 14 + padding * 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)

    # Draw every route text
    cv2.putText(dest, "T-Truck", (
    content_top_left_x + content_col_padding + padding * 6, content_top_left_y + content_row_padding * 1 + padding * 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)
    cv2.putText(dest, "Large", (
    content_top_left_x + content_col_padding + padding * 6, content_top_left_y + content_row_padding * 2 + padding * 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)
    cv2.putText(dest, "Small", (
    content_top_left_x + content_col_padding + padding * 6, content_top_left_y + content_row_padding * 3 + padding * 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)
    cv2.putText(dest, "T-Truck", (
    content_top_left_x + content_col_padding + padding * 6, content_top_left_y + content_row_padding * 4 + padding * 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)
    cv2.putText(dest, "Large", (
    content_top_left_x + content_col_padding + padding * 6, content_top_left_y + content_row_padding * 5 + padding * 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)
    cv2.putText(dest, "Small", (
    content_top_left_x + content_col_padding + padding * 6, content_top_left_y + content_row_padding * 6 + padding * 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)
    cv2.putText(dest, "T-Truck", (
    content_top_left_x + content_col_padding + padding * 6, content_top_left_y + content_row_padding * 7 + padding * 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)
    cv2.putText(dest, "Large", (
    content_top_left_x + content_col_padding + padding * 6, content_top_left_y + content_row_padding * 8 + padding * 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)
    cv2.putText(dest, "Small", (
    content_top_left_x + content_col_padding + padding * 6, content_top_left_y + content_row_padding * 9 + padding * 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)
    cv2.putText(dest, "T-Truck", (
    content_top_left_x + content_col_padding + padding * 6, content_top_left_y + content_row_padding * 10 + padding * 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)
    cv2.putText(dest, "Large", (
    content_top_left_x + content_col_padding + padding * 6, content_top_left_y + content_row_padding * 11 + padding * 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)
    cv2.putText(dest, "Small", (
    content_top_left_x + content_col_padding + padding * 6, content_top_left_y + content_row_padding * 12 + padding * 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)
    cv2.putText(dest, "T-Truck", (
    content_top_left_x + content_col_padding + padding * 6, content_top_left_y + content_row_padding * 13 + padding * 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)
    cv2.putText(dest, "Large", (
    content_top_left_x + content_col_padding + padding * 6, content_top_left_y + content_row_padding * 14 + padding * 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)
    cv2.putText(dest, "Small", (
    content_top_left_x + content_col_padding + padding * 6, content_top_left_y + content_row_padding * 15 + padding * 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)

    # Draw every route value
    cv2.putText(dest, ttruck_1, (content_top_left_x + content_col_padding * 3 + padding * 6,
                            content_top_left_y + content_row_padding * 1 + padding * 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                content_color, 2)
    cv2.putText(dest, large_1, (content_top_left_x + content_col_padding * 3 + padding * 6,
                            content_top_left_y + content_row_padding * 2 + padding * 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                content_color, 2)
    cv2.putText(dest, small_1, (content_top_left_x + content_col_padding * 3 + padding * 6,
                            content_top_left_y + content_row_padding * 3 + padding * 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                content_color, 2)
    cv2.putText(dest, ttruck_2, (content_top_left_x + content_col_padding * 3 + padding * 6,
                            content_top_left_y + content_row_padding * 4 + padding * 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                content_color, 2)
    cv2.putText(dest, large_2, (content_top_left_x + content_col_padding * 3 + padding * 6,
                            content_top_left_y + content_row_padding * 5 + padding * 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                content_color, 2)
    cv2.putText(dest, small_2, (content_top_left_x + content_col_padding * 3 + padding * 6,
                            content_top_left_y + content_row_padding * 6 + padding * 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                content_color, 2)
    cv2.putText(dest, ttruck_3, (content_top_left_x + content_col_padding * 3 + padding * 6,
                            content_top_left_y + content_row_padding * 7 + padding * 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                content_color, 2)
    cv2.putText(dest, large_3, (content_top_left_x + content_col_padding * 3 + padding * 6,
                            content_top_left_y + content_row_padding * 8 + padding * 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                content_color, 2)
    cv2.putText(dest, small_3, (content_top_left_x + content_col_padding * 3 + padding * 6,
                            content_top_left_y + content_row_padding * 9 + padding * 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                content_color, 2)
    cv2.putText(dest, ttruck_4, (content_top_left_x + content_col_padding * 3 + padding * 6,
                            content_top_left_y + content_row_padding * 10 + padding * 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                content_color, 2)
    cv2.putText(dest, large_4, (content_top_left_x + content_col_padding * 3 + padding * 6,
                            content_top_left_y + content_row_padding * 11 + padding * 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                content_color, 2)
    cv2.putText(dest, small_4, (content_top_left_x + content_col_padding * 3 + padding * 6,
                            content_top_left_y + content_row_padding * 12 + padding * 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                content_color, 2)
    cv2.putText(dest, ttruck_5, (content_top_left_x + content_col_padding * 3 + padding * 6,
                            content_top_left_y + content_row_padding * 13 + padding * 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                content_color, 2)
    cv2.putText(dest, large_5, (content_top_left_x + content_col_padding * 3 + padding * 6,
                            content_top_left_y + content_row_padding * 14 + padding * 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                content_color, 2)
    cv2.putText(dest, small_5, (content_top_left_x + content_col_padding * 3 + padding * 6,
                            content_top_left_y + content_row_padding * 15 + padding * 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                content_color, 2)

    return dest


if __name__ == '__main__':
    print("### DRAWING THE TABLE RESULT ###")
    img = cv2.imread("C:/Users/Trinh/Downloads/abc/raindrop_000000.png")
    current_frame = 125
    date_time ="2020-10-05 09"
    minute_period = "15-30"

    ttruck_1 = 1
    large_1 = 2
    small_1 = 3
    ttruck_2 = 4
    large_2 = 5
    small_2 = 6
    ttruck_3 = 7
    large_3 = 8
    small_3 = 19
    ttruck_4 = 23
    large_4 = 85
    small_4 = 55
    ttruck_5 = 24
    large_5 = 15
    small_5 = 89

    dest = draw_result_table(img, str(current_frame), date_time, minute_period, str(ttruck_1), str(large_1), str(small_1),
                             str(ttruck_2), str(large_2), str(small_2), str(ttruck_3), str(large_3), str(small_3),
                             str(ttruck_4), str(large_4), str(small_4), str(ttruck_5), str(large_5), str(small_5)
                             )
    cv2.imshow("dest", dest)
    cv2.imwrite("C:/Users/Trinh/Downloads/abc/output.png", dest)
    cv2.waitKey(0)
