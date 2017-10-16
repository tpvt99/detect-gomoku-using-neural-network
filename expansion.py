import numpy
import cv2

def expan(z):
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if z[i][j] <= 10:
                range_i = [k for k in range(i-1,i+1)]
                range_j = [m for m in range(j-1,j+1)]
                for k in range_i:
                    for m in range_j:
                        if k >= 115:
                            k = 115
                        if m >= 115:
                            m = 115
                        z[k][m] = 0

    return z


if __name__ == '__main__':
    image = cv2.imread('image1.jpg', 0)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] >= 127:
                image[i][j] = 255
            else:
                image[i][j] = 0
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', image)
    cv2.waitKey(0)
