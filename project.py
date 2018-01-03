import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import time
import test
import os
import expansion
from PIL import Image, ImageChops


real_vertical_lines = []
real_vertical_lines_angle = []
real_horizontal_lines = []
real_horizontal_lines_angle = []
row = 0
column = 0

real_col = 10
real_row = 10
# ----------------------------------------- #
# ----------------------------------------- #
crop_width = 10

sss_num = 1536


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((10,10)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    return im

def determine_row_col(gray):
    edges = cv2.Canny(gray, 50, 150, apertureSize = 3)

    lines = cv2.HoughLines(edges, 1, np.pi/180, 130)

    row_error = 20

    row = 0
    column = 0

    vertical_lines = []
    horizontal_lines = []

    if lines == None:
        lines = []

    for line in lines:
        rho, theta = line[0]
        if theta >= 1.5 and theta <= 1.6:
            horizontal_lines.append((rho,theta))
        elif theta >= 0 and theta <= 0.1:
            vertical_lines.append((rho,theta))

    vertical_lines.sort(key = lambda x: abs(x[0]))
    horizontal_lines.sort(key = lambda x: abs(x[0]))
    vertical_lines = [(abs(i),k) for i,k in vertical_lines]
    horizontal_lines = [(abs(i),k) for i,k in horizontal_lines]


    for val, theta in vertical_lines:
        total_val = 0
        total_angle = 0
        count = 0
        for m in vertical_lines:
            if m[0] <= val + row_error and m[0] >= val - row_error:
                total_val += m[0]
                total_angle += m[1]
                count += 1
        total_val = round(total_val / count)
        total_angle = total_angle / count
        if total_val not in real_vertical_lines:
            real_vertical_lines.append(total_val)
            real_vertical_lines_angle.append(total_angle)

    for val, theta in horizontal_lines:
        total_val = 0
        total_angle = 0
        count = 0
        for m in horizontal_lines:
            if m[0] <= val + row_error and m[0] >= val - row_error:
                total_val += m[0]
                total_angle += m[1]
                count += 1
        total_val = round(total_val / count)
        total_angle = total_angle / count
        if total_val not in real_horizontal_lines:
            real_horizontal_lines.append(total_val)
            real_horizontal_lines_angle.append(total_angle)

    for rho, theta in zip(real_vertical_lines, real_vertical_lines_angle):
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(gray, (x1, y1), (x2,y2), (0,0,255), 2)

    for rho, theta in zip(real_horizontal_lines, real_horizontal_lines_angle):
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(gray, (x1, y1), (x2,y2), (0,255,0), 2)

    return (math.floor((row -1)/ 2), math.floor((column -1)/ 2))


def draw(row , col):
    x_width = 50
    y_height = 50
    space = 20
    line_width = 2
    total_width = x_width * col + line_width * (col + 1)
    total_height = y_height * row + line_width * (row + 1)
    width_resolution = total_width + 2 * space
    height_resolution = total_height + 2 * space

    img = np.ones((height_resolution, width_resolution), np.uint8)
    img = 255 * img

    # draw first vertical line
    top_corner = [space, space]
    bottom_corner = [space, total_height]
    cv2.line(img, tuple(top_corner), tuple(bottom_corner), 0, line_width)
    top_corner[0] = top_corner[0] + line_width
    bottom_corner[0] = bottom_corner[0] + line_width


    # draw vertical line
    for i in range(col):
        top_corner[0] = top_corner[0] + x_width
        bottom_corner[0] = bottom_corner[0] + x_width
        cv2.line(img, tuple(top_corner), tuple(bottom_corner), 0, line_width)
#        top_corner[0] = top_corner[0] + line_width
#        bottom_corner[0] = bottom_corner[0] + line_width

    # draw first horizontal line
    top_corner = [space, space]
    bottom_corner = [total_width, space]
    cv2.line(img, tuple(top_corner), tuple(bottom_corner), 0, line_width)
    top_corner[1] = top_corner[1] + line_width
    bottom_corner[1] = bottom_corner[1] + line_width

    for i in range(row):
        top_corner[1] = top_corner[1] + y_height
        bottom_corner[1] = bottom_corner[1] + y_height
        cv2.line(img, tuple(top_corner), tuple(bottom_corner), 0, line_width)
#        top_corner[1] = top_corner[1] + line_width
#        bottom_corner[1] = bottom_corner[1] + line_width


    return img

def get_coordinate(pos):
    x,y = pos
    space = 20
    x_width = 50
    y_width = 50
    line_width = 2
    total_x = space + (y) * x_width
    total_x = total_x + x_width//2

    total_y = space + (x) * y_width
    total_y = total_y + y_width//2

    return (total_x, total_y)

# pos is tupe contains (x,y) cordinate
def draw_X(image, pos):
    center = get_coordinate(pos)
    space = 10
    line1_pos1 = (center[0] - space, center[1] - space)
    line1_pos2 = (center[0] + space, center[1] + space)
    line2_pos1 = (center[0] - space, center[1] + space)
    line2_pos2 = (center[0] + space, center[1] - space)
    cv2.line(image, line1_pos1, line1_pos2, 0, 5)
    cv2.line(image, line2_pos1, line2_pos2, 0, 5)

def draw_O(image, pos):
    cv2.circle(image, get_coordinate(pos), 15, 0, 5)



cv2.namedWindow('image', cv2.WINDOW_NORMAL)


def detection_segment(matrix, img):
    global sss_num
    for i in range(row):
        for j in range(column):
            first_row = real_horizontal_lines[i]
            second_row = real_horizontal_lines[i+1]
            first_column = real_vertical_lines[j]
            second_column = real_vertical_lines[j+1]
            #print('At position :' + str(i) + '  ' + str(j), sep = ' ')
            seg = img[first_row+crop_width:second_row-crop_width,first_column+crop_width:second_column-crop_width]
            #cv2.imshow('image_before_dilate', seg)

            #kernel = np.zeros((5,5), np.uint8)
            #seg = cv2.dilate(seg, kernel, iterations = 1)

            #seg = expansion.expan(seg)

            for aaa in range(seg.shape[0]):
                for bbb in range(seg.shape[1]):
                    if seg[aaa][bbb] <= 230:
                        seg[aaa][bbb] = 0
            cv2.imshow('image_dilate', seg)
            cv2.imwrite('segment.jpg', seg)

            time.sleep(0.3)
            cv2.waitKey(1)
            #cv2.imwrite('training_data/cross/cross' + str(sss_num) + '.jpg', seg)
            #sss_num += 1


            #pil_img = Image.open('segment.jpg')
            #im = trim(pil_img)
            #im.save('segment.jpg', 'JPEG')

            result = test.test_project('segment.jpg')
            matrix[i][j] = result


def draw_real_matrix(matrix, image):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] == 0:
                draw_O(image, (i,j))
            elif matrix[i][j] == 1:
                draw_X(image, (i,j))



if __name__ == '__main__':
    # first is capture image
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv2.imread('100-blocks-1.png')
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = preprocess(img)
        determine_row_col(frame)

        cv2.imshow('real_image', gray)
        cv2.namedWindow('row_column_draw', cv2.WINDOW_NORMAL)
        cv2.imshow('row_column_draw', frame)
        row = len(real_horizontal_lines) - 1
        column = len(real_vertical_lines) -1

        k = cv2.waitKey(1)
        if row != real_row or column != real_col:
            real_vertical_lines = []
            real_vertical_lines_angle = []
            real_horizontal_lines = []
            real_horizontal_lines_angle = []
            continue

        # draw computer
        matrix = np.zeros((row,column))
        computer_draw = draw(row, column)

        # draw real
        detection_segment(matrix, gray)

        draw_real_matrix(matrix, computer_draw)

        cv2.imshow('computer_draw', computer_draw)

        k = cv2.waitKey(1)
        if k == 27: # Esc to escape
            break


        real_vertical_lines = []
        real_vertical_lines_angle = []
        real_horizontal_lines = []
        real_horizontal_lines_angle = []


    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #cv2.imshow('image', computer_draw)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
