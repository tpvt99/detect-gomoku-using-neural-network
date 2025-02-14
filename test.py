import numpy as np
import cv2
from PIL import Image, ImageFilter
import training_vs2


weights = np.load('weights.npy')
biases = np.load('biases.npy')


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


def feedforward(a):
    for b, w in zip(biases, weights):
        a = sigmoid(np.dot(w, a) + b)
    return a

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def test_project(img):
    image = imageprepare(img)
    img = np.reshape(image, (784,1))
    zzz = np.reshape(img, (28,28))
    #cv2.namedWindow('ahihihi', cv2.WINDOW_NORMAL)
    #cv2.imshow('ahihihi', zzz)
    #cv2.waitKey(1)
    m = feedforward(img)
    return np.argmax(m)
