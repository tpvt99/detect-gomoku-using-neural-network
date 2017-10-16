import os
import cv2

path = '/home/web/openCV/project1/training_data/cross'
os.chdir(path)
m = 1
k = 1025
for i in os.listdir():
    img = cv2.imread(i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] <= 248:
                img[i][j] = 255
    cv2.imwrite('../white_sm/white' + str(k) + '.jpg', img)
    k += 1
    m +=1
    if m == 1024:
        break
