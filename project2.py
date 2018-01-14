import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import time
import test
import os
import expansion


row = 0
column = 0
real_col = 10
real_row = 10

crop_width = 8


def preprocess(gray):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    dilated = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
    diff1 = 255 - cv2.subtract(dilated, gray)

    median = cv2.medianBlur(dilated, 15)
    diff2 = 255 - cv2.subtract(median, gray)

    normed = cv2.normalize(diff2, None, 0, 255, cv2.NORM_MINMAX)

    res = np.hstack((gray, dilated, diff1, median, diff2, normed))
    return normed

def determine_corner(gray, img):

    row = 0
    column = 0
    dst = cv2.cornerHarris(gray, 2, 1, 0.05)

    dst = cv2.dilate(dst, None)
    print(dst)
    print(dst.shape)
    print(dst > 0.01*dst.max())
    time.sleep(2)
    print(60*'*')

    img[dst > 0.01*dst.max()] = [0,0,255]



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
        print(bottom_corner)
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
    for i in range(row):
        for j in range(column):
            first_row = real_horizontal_lines[i]
            second_row = real_horizontal_lines[i+1]
            first_column = real_vertical_lines[j]
            second_column = real_vertical_lines[j+1]
            print(first_row, second_row, first_column, second_column)
            print(i,j, sep = '-')
            seg = img[first_row+crop_width:second_row-crop_width,first_column+crop_width:second_column-crop_width]
            cv2.imshow('image', seg)
            # expansion
            #seg = expansion.expan(seg)
            kernel = np.ones((5,5), np.uint8)
            seg = cv2.dilate(seg, kernel, iterations = 1)
            cv2.imshow('image_dilate', seg)
            time.sleep(0)
            cv2.waitKey(1)
            cv2.imwrite('segment.jpg', seg)
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
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray = preprocess(img)
        determine_corner(gray, frame)

        cv2.imshow('real_image', gray)
        cv2.namedWindow('row_column_draw', cv2.WINDOW_NORMAL)
        cv2.imshow('row_column_draw', frame)
        k = cv2.waitKey(1)
        continue

        print(row, column)

        if row != real_row or column != real_col:
            continue

        # draw computer
        matrix = np.zeros((row,column))
        computer_draw = draw(row, column)

        # draw real
        detection_segment(matrix, gray)

        draw_real_matrix(matrix, computer_draw)

        cv2.imshow('computer_draw', computer_draw)

        k = cv2.waitKey(300)
        if k == 27: # Esc to escape
            break


