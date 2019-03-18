#--------------------------Vedio Feed usng OpenCv -----------------------------------

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
import math
import keras.models
import imutils

#The OPenCv performs same operation to feed whether it is an image or Vedio.The way vedio works is with THe frames per sec (FPS) and each frame is #still an image so it is basically the same thing or we can say that Looping the continuos capturing of image leads to the formation of images.'''

#For primary   Webcam Feed :- 0
#For secondary Webcam Feed :- 1

#printing doc
print(__doc__)

def find_marker(image):
	# convert the image to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 35, 125)

	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
	contours,hierachy=cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	c = max(contours, key = cv2.contourArea)

	# compute the bounding box of the of the paper region and return it
	return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth

# initialize the known distance from the camera to the object, which
# in this case is 24 inches
KNOWN_DISTANCE = 14

# initialize the known object width, which in this case, the piece of
# paper is 12 inches wide
KNOWN_WIDTH = 5

# initialize the list of images that we'll be using
#IMAGE_PATHS = []

# load the furst image that contains an object that is KNOWN TO BE 2 feet
# from our camera, then find the paper marker in the image, and initialize
# the focal length
image = cv2.imread("training_pic/trainpic.jpg")
marker = find_marker(image)
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH


#Capturing Images.
cap=cv2.VideoCapture(0) #For Primary webcam

#To use recoded vedio as a feed 
'''cap=cv2.VedioCapture('Location of the vedio')'''


#To save each frames 
#fourcc=cv2.VedioWriter("Output_name.avi",fourcc,20.0,(720,640))#(720,640)==720x640 pixel values.It depends on the Webcam quality.

clf1 = keras.models.load_model("Model/gclf.h5py")

clf = pickle.load(open("Model/clf.pkl","rb"))


#Loading the cascade classifier files
face_cascade=cv2.CascadeClassifier('Haarcascades_Datasets/haarcascade_frontalface_default.xml')#copy the locations

eye_cascade=cv2.CascadeClassifier('Haarcascades_Datasets/haarcascade_eye.xml')#copy the locations

#looping through the webcam feed
if cap.isOpened():
	while 1:
        
        	#reading the frame
        	ret, img=cap.read()
	
        	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	
        	im = cv2.resize(img,(64,64))
	
        	bailey = np.expand_dims(im, axis=0)
        	
        	prediction_b = clf1.predict(bailey)
	
        	if math.floor(prediction_b) >=0.15:
	
        	        prediction_b = "Female"
        	        
        	else:
        	        prediction_b = "Male"
        	
	
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
                                
                #predicting the value
                	res = clf.predict(simg)
                
                #reduction for noise
                	if res:
                        	print("Gender :{}\tPredicted Age is :{}".format((prediction_b),abs(res+20)//2))
                        
                #detection of eyes
                	eyes = eye_cascade.detectMultiScale(roi_gray)

                	marker = find_marker(roi_color)
                
                	inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
                
                	cv2.putText(img, "%.2fft" % (inches / 12),(x , y), cv2.FONT_HERSHEY_SIMPLEX,2.0, (0, 255, 255), 1)

                
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
