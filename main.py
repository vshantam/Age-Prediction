#--------------------------Vedio Feed usng OpenCv ------------------------------------

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 19:41:35 2018

@author: Shantam Vijayputra
"""

#Required Modules

import numpy as np
import cv2
import os
import pickle

'''The OPenCv performs same operation to feed whether it is an image or Vedio.The way vedio works is with THe frames per sec (FPS) and each frame is still an image so it is basically the same thing or we can say that Looping the continuos capturing of image leads to the formation of images.'''

#For primary   Webcam Feed :- 0
#For secondary Webcam Feed :- 1

#printing doc
print(__doc__)

#Capturing Images.
cap=cv2.VideoCapture(0) #For Primary webcam

#To use recoded vedio as a feed 
'''cap=cv2.VedioCapture('Location of the vedio')'''


#To save each frames 
#fourcc=cv2.VedioWriter("Output_name.avi",fourcc,20.0,(720,640))#(720,640)==720x640 pixel values.It depends on the Webcam quality.

#Loading the cascade classifier files
face_cascade=cv2.CascadeClassifier('C://Users/Haarcascades_Datasets/haarcascade_frontalface_default.xml ')#copy the locations
eye_cascade=cv2.CascadeClassifier('C://Users/Haarcascades_Datasets/haarcascade_eye.xml ')#copy the locations

#looping through the webcam feed
while 1:
        
        #reading the frame
        ret, img = cap.read()
        
        #conversion of grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #detection of facial coordinates
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        #creating rectangles
        for (x,y,w,h) in faces:
                
                #in face
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                
                #extracting the facial part
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                
                #reshaping for prediction
                simg = cv2.resize(roi_gray,(10,10))
                
                #flattening
                simg = simg.flatten().reshape(-1,1)
                
                #transpose
                simg = simg.T/10.0
                
                #loading the classifier
                clf = pickle.load(open("c://Users/Shantam Vijayputra/Desktop/clf.pkl","rb"))
                
                #predicting the value
                res = clf.predict(simg)
                
                #reduction for noise
                if res//2 >15:
                        print("Predicted Age is :{}".format(abs(res)//2))
                        
                #detection of eyes
                eyes = eye_cascade.detectMultiScale(roi_gray)
                
                #looping through eye coordinates
                for (ex,ey,ew,eh) in eyes:
                        
                        #creating rectangles
                        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        #displaying the image
        cv2.imshow('img',img)
        
        #wait key
        k = cv2.waitKey(30) & 0xff
        if k == 27:
                break
        
#releasing the webcamfeed
cap.release()

#closing all the window
cv2.destroyAllWindows()
