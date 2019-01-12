# Using Python 3.6.5
# Using OpenCV 4.0.0
# Using Numpy 1.15.4
# Using macOS 10.14.1

import sys
import numpy as np
import cv2

# Reading in the Train png and making a copy
im =  cv2.imread('Images/Train/tiu_train.png')
im_copy = im.copy()

## Drawing Contours
## https://docs.opencv.org/3.0.0/d4/d73/tutorial_py_contours_begin.html
## https://medium.com/@gsari/digit-recognition-with-opencv-and-python-cbf962f7e2d0
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(imgray,(5,5),0)
thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

## Finding the contours

# Page 99, https://media.readthedocs.org/pdf/opencv-python-tutroals/latest/opencv-python-tutroals.pdf 
# Describing the "findContours" function
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)


samples = np.empty((0,100))
responses = []
keys = [i for i in range(48,58)]

print(contours)

for cont in contours:
    if cv2.contourArea(cont)>50:
        [x,y,w,h] = cv2.boundingRect(cont)

        if h>28:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            cv2.imshow('norm',im)
            key = cv2.waitKey(0)

            if key == 27:
                sys.exit()
            elif key in keys:
                responses.append(int(chr(key)))
                sample = roismall.reshape((1,100))
                samples = np.append(samples,sample,0)

responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size),1)
print("Training complete!")


np.savetxt('generalsamples.data',samples)
np.savetxt('generalresponses.data',responses)









### Preparing the data for training ###
## https://stackoverflow.com/questions/9413216/simple-digit-recognition-ocr-in-opencv-python  ##