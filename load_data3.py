import cv2
import os
import numpy as np
from PIL import Image, ImageFilter
import math

path = './training_data'
path_circle = path + '/circle'
path_cross = path + '/cross'
path_white = path + '/white'

training_inputs = []
training_results = []

test_inputs = []
test_results = []

circle_result = np.zeros((3,1))
circle_result[0] = 1
cross_result = np.zeros((3,1))
cross_result[1] = 1
white_result = np.zeros((3,1))
white_result[2] = 1


def imageprepare(argv):    
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255)) #creates white canvas of 28x28 pixels
    
    if width > height: #check which dimension is bigger
        #Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20/width*height),0)) #resize height according to ratio width
        if (nheight == 0): #rare case but minimum is 1 pixel
            nheight = 1  
        # resize and sharpen
        img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight)/2),0)) #caculate horizontal pozition
        newImage.paste(img, (4, wtop)) #paste resized image on white canvas
    else:
        #Height is bigger. Heigth becomes 20 pixels. 
        nwidth = int(round((20/height*width),0)) #resize width according to ratio height
        if (nwidth == 0): #rare case but minimum is 1 pixel
            nwidth = 1
         # resize and sharpen
        img = im.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth)/2),0)) #caculate vertical pozition
        newImage.paste(img, (wleft, 4)) #paste resized image on white canvas    
    #newImage.save("sample.png")
    tv = list(newImage.getdata()) #get pixel values    
    #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255-x)/255 for x in tv] 
    return tva
    #print(tva)

def resize(img):
    if type(img) == list:
        a = round(math.sqrt(len(img)))
        img = np.array(img)
        img = img.reshape((a,a,1))
    return img

def load_data():
    global training_inputs
    global test_inputs
    global training_results
    global test_results

    for i in os.listdir(path_circle):
        img = resize(imageprepare(path_circle + '/' + i))
        training_inputs.append(img)
        training_results.append(circle_result)

        test_inputs.append(img)
        test_results.append(circle_result)

    for i in os.listdir(path_cross):
        img = resize(imageprepare(path_cross + '/' + i))
        training_inputs.append(img)
        training_results.append(cross_result)

        test_inputs.append(img)
        test_results.append(cross_result)

    for i in os.listdir(path_white):
        img = resize(imageprepare(path_white + '/' + i))
        training_inputs.append(img)
        training_results.append(white_result)

        test_inputs.append(img)
        test_results.append(white_result)

    training_inputs = np.array(training_inputs)
    test_inputs = np.array(test_inputs)
    training_results = np.array(training_results)
    test_results = np.array(test_results)

    return training_inputs, training_results, test_inputs, test_results

if __name__ == '__main__':
    training_inputs, training_results, test_inputs, test_results = load_data()
    np.save('training_inputs.npy', training_inputs)
    np.save('training_results.npy', training_results)
    np.save('test_inputs.npy', test_inputs)
    np.save('test_results.npy', test_results)

