import numpy as np
import cv2


def draw_config_table(img = '', g_target_vehicle = '', g_speed = '', g_preceding = '', g_inter_dis = '', h_target_vehicle = '', h_speed = '', h_preceding = '', h_inter_dis = ''):
    # Create a black image
    dest = img.copy()

    # Background
    bg_top_left_x = 15
    bg_top_left_y = 5
    start_plus_y = 65

    bg_bottom_right_x = 700
    bg_bottom_right_y = 200

    padding = 5
    time_padding_y = 0
    bg_width = bg_bottom_right_x - bg_top_left_x
    bg_height = bg_bottom_right_y - bg_top_left_y

    content_ncol = 5
    content_nrow = 4

    # color panel (bgr)
    bg_color = (224, 224, 235)
    frame_color = (0, 0, 0)
    content_color = (0, 200, 0)

    ########### DRAW ###########
    # Draw Background
    dest = cv2.rectangle(dest, (bg_top_left_x, bg_top_left_y + start_plus_y), (bg_bottom_right_x, bg_bottom_right_y), bg_color, -1)

    # Result table
    content_top_left_x = bg_top_left_x + padding * 1
    content_top_left_y = time_padding_y * 1 + padding * 3 + start_plus_y

    content_bottom_right_x = bg_width + padding*2
    content_bottom_right_y = bg_height - padding

    content_width = content_bottom_right_x - content_top_left_x
    content_height = content_bottom_right_y - content_top_left_y

    content_row_padding = int(content_height / content_nrow)
    content_col_padding = int(content_width / content_ncol)

    print("content_top_left_x, content_top_left_y: ", content_top_left_x, content_top_left_y)
    print("content_bottom_right_x, content_bottom_right_y:", content_bottom_right_x, content_bottom_right_y)

    print("content_width: ", content_width)
    print("content_height: ", content_height)
    # Draw the biggest frame
    dest = cv2.rectangle(dest, (content_top_left_x, content_top_left_y),
                         (content_bottom_right_x, content_bottom_right_y), frame_color, 1)

    ###### Draw columns ######
    dest = cv2.rectangle(dest, (content_top_left_x, content_top_left_y),
                         (content_bottom_right_x - content_col_padding * 4, content_bottom_right_y), frame_color, 1)
    dest = cv2.rectangle(dest, (content_top_left_x, content_top_left_y),
                         (content_bottom_right_x - content_col_padding * 3, content_bottom_right_y), frame_color, 1)
    dest = cv2.rectangle(dest, (content_top_left_x, content_top_left_y),
                         (content_bottom_right_x - content_col_padding * 2, content_bottom_right_y), frame_color, 1)
    dest = cv2.rectangle(dest, (content_top_left_x, content_top_left_y),
                         (content_bottom_right_x - content_col_padding * 1, content_bottom_right_y), frame_color, 1)

    ###### Draw rows ######
    dest = cv2.rectangle(dest, (content_top_left_x, content_top_left_y),
                         (content_bottom_right_x, content_bottom_right_y - content_row_padding * 2), frame_color, 1)
    dest = cv2.rectangle(dest, (content_top_left_x, content_top_left_y),
                         (content_bottom_right_x, content_bottom_right_y - content_row_padding * 1), frame_color, 1)

    ###### Draw texts ######
    # First Line
    cv2.putText(dest, "Target Vehicle",
                (content_top_left_x + content_col_padding + padding * 1, content_top_left_y + padding * 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)
    cv2.putText(dest, "Speed [km/h]",
                (content_top_left_x + content_col_padding * 2 + padding * 1, content_top_left_y + padding * 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)
    cv2.putText(dest, "Preceding",
                (content_top_left_x + content_col_padding * 3 + padding * 1, content_top_left_y + padding * 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)
    cv2.putText(dest, "Internal",
                (content_top_left_x + content_col_padding * 4 + padding * 1, content_top_left_y + padding * 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)

    # Second Line
    cv2.putText(dest, "Type",
                (content_top_left_x + content_col_padding + padding * 1, content_top_left_y + content_row_padding * 1 + padding * 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)
    cv2.putText(dest, "Vehicle Type",
                (content_top_left_x + content_col_padding * 3 + padding * 1, content_top_left_y + content_row_padding * 1 + padding * 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)
    cv2.putText(dest, "Distance [m]",
                (content_top_left_x + content_col_padding * 4 + padding * 1, content_top_left_y + content_row_padding * 1 + padding * 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)

    # Third Line
    cv2.putText(dest, "Gouryu",
                (content_top_left_x + padding * 8, content_top_left_y + content_row_padding * 2 + padding * 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)
    cv2.putText(dest, g_target_vehicle, (
                content_top_left_x + content_col_padding * 1 + padding * 4, content_top_left_y + content_row_padding * 2 + padding * 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)
    cv2.putText(dest, g_speed, (
                content_top_left_x + content_col_padding * 2 + padding * 4, content_top_left_y + content_row_padding * 2 + padding * 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)
    cv2.putText(dest, g_preceding, (
                content_top_left_x + content_col_padding * 3 + padding * 4, content_top_left_y + content_row_padding * 2 + padding * 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)
    cv2.putText(dest, g_inter_dis, (
                content_top_left_x + content_col_padding * 4 + padding * 4, content_top_left_y + content_row_padding * 2 + padding * 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)

    # Fourth Line
    cv2.putText(dest, "Honsen",
                (content_top_left_x + padding * 8, content_top_left_y + content_row_padding * 3 + padding * 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)
    cv2.putText(dest, h_target_vehicle, (
                content_top_left_x + content_col_padding * 1 + padding * 4, content_top_left_y + content_row_padding * 3 + padding * 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)
    cv2.putText(dest, h_speed, (
                content_top_left_x + content_col_padding * 2 + padding * 4, content_top_left_y + content_row_padding * 3 + padding * 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)
    cv2.putText(dest, h_preceding, (
                content_top_left_x + content_col_padding * 3 + padding * 4, content_top_left_y + content_row_padding * 3 + padding * 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)
    cv2.putText(dest, h_inter_dis, (
                content_top_left_x + content_col_padding * 4 + padding * 4, content_top_left_y + content_row_padding * 3 + padding * 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, content_color, 2)

    return dest


if __name__ == '__main__':
    print("### DRAWING THE CONFIG TABLE ###")
    img = cv2.imread("./data/Truck_ViewC_0.jpg")

    g_target_vehicle = "T-Truck"
    g_speed = "80"
    g_preceding = "Free"
    g_inter_dis = "50"
    h_target_vehicle = "Large"
    h_speed = "90"
    h_preceding = "Small"
    h_inter_dis = "60"

    dest = draw_config_table(img, g_target_vehicle, g_speed, g_preceding, g_inter_dis, h_target_vehicle, h_speed, h_preceding, h_inter_dis)
    cv2.imshow("dest", dest)
    cv2.imwrite("C:/Users/Trinh/Downloads/abc/output.png", dest)
    cv2.waitKey(0)
