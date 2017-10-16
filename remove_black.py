import cv2
import numpy as np

img = cv2.imread('segment.jpg', 0)
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img)
ret,thresh = cv2.threshold(img ,1, 255, cv2.THRESH_BINARY)

i, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(contours)
print(hierarchy)
cnt = contours[0]
x,y,w,h = cv2.boundingRect(cnt)

crop = img[y:y+h,x:x+w]
cv2.imshow('image', crop)
cv2.waitKey(0)
exit(0)
