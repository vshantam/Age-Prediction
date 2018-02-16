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
from PIL import Image

'''The OPenCv performs same operation to feed whether it is an image or Video.The way video works is with THe frames per sec (FPS) and each frame is still an image so it is basically the same thing or we can say that Looping the continuos capturing of image leads to the formation of images.'''

#For primary   Webcam Feed :- 0
#For secondary Webcam Feed :- 1

class Detect(object):

	@classmethod
	def capture(self):
		
		#Capturing Images.
		cap=cv2.VideoCapture(0) #For Primary webcam

		return cap

		#To use recoded vedio as a feed 
		'''cap=cv2.VedioCapture('Location of the vedio')'''

	@classmethod
	def load_clf(self,path):
		
		#loading casade classifier
		return cv2.CascadeClassifier(path)#copy the locations

	@classmethod
	def load_predictor(self,path,_type):

		#Looping the continuos caption .
		regressor = pickle.load(open(path,_type))

		return regressor


	@classmethod
	def detect_and_predict(self,face_cascade,eye_cascade,regressor,cap):
		
		list = []
		count =0
		
		#looping through webcam
		while True:
			#reading the frame
			ret,frame=cap.read()

			#conversion of grayscale
			gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

			#detection of facial coordinated
			faces=face_cascade.detectMultiScale(gray,1.3,3)

			#For changing the Background color of the Vedio feeds.
			for (x,y,w,h) in faces:
		
				#drawing the rectangle
				frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)
		
				#capturing the facal part in grayscale
				roi_gray=gray[x:x+w,y:y+h]
				
				#capturing the facial part in color
				roi_color=frame[y:y+h,x:x+w]
		
				#resizing the facial part
				frame_10 = cv2.resize(roi_gray,(10,10))
		
				#matrix to array
				img = frame_10.flatten().reshape(-1,1)
		
				#conversion of m x 1 matrix to 1 x m
				img= img.T
				img = np.array(img)
				img = img
		
				#prediction
				y_pred1 = regressor.predict(img)
		
				#scaling the values
				list.append(abs(y_pred1))
		
				#eye detection
				eyes=eye_cascade.detectMultiScale(roi_gray)
		
				#looping through eyes
				for (ex,ey,ew,eh) in eyes:
			
					#drawing rectangle on eyes
					cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),4)

				#counter to delay displaying the predicted values 
				if count %5 ==0:
					if  (sum(list)//len(list)) > 14 :
						print("Predicted Age is :{}".format(y_pred1//20))

			#displayng the image
			cv2.imshow("frame",frame)
					      
			#incrementing counter
			count += 1
	


		#	out.write(frame)

			if cv2.waitKey(1) &  0xFF=='q':
				break

		#release webcam
		cap.release()

		#close all window
		cv2.destroyAllWindows()


#calling main method
if __name__ == "__main__":

	#doc printing
	print(__doc__)

	#creating object
	obj = Detect()

	#capturing feed from the camera
	cap = obj.capture()

	#loading classifier pickle  data
	clf = obj.load_predictor("C://Users/Shantam Vijayputra/Desktop/clf.pkl","rb")

	#loading ascade classifier data
	face_cascade = obj.load_clf("C://Users/Haarcascades_Datasets/haarcascade_frontalface_default.xml")
	eye_cascade = obj.load_clf("C://Users/Haarcascades_Datasets/haarcascade_eye.xml")

	#detection and age prediction
	obj.detect_and_predict(face_cascade,eye_cascade,clf,cap)

