# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 22:53:44 2018

@author: Shantam Vijayputra
"""

#importng the libraries
import pandas as pd
import numpy as np
import cv2
import os

#creating classmethod
class Imagescale(object):

    #calling classmethod and defining image resize functio
    @classmethod
    def resizeimage(self):

        #changing the directory tho extracted dataset of wiki images
        os.chdir("c://Users/Shantam Vijayputra/Desktop/Python Projects and Implementations/images/")
        
        #list of all folders tha contain images
        dir = os.listdir()
        
        #looping through all the directories and images
        for i in dir:

            #listing all
            list = os.listdir()

            #for iamge n each dir
            for j in list:
                
                #reading the image
                img = cv2.imread(j)
                
                #resizing the image
                rm = cv2.resize(img,(64,64))
                
                #saving the resized image
                cv2.imwrite('C:/Users/Shantam Vijayputra/Desktop/images/'+str(j),rm)
        
        #returning to the main dataset folder
        return os.chdir('c://Users/Shantam Vijayputra/Desktop/Python Projects and Implementations')


    #calling classmethod and efinig the the crop function
    @classmethod
    def detectfaces_crop(self):
                        
        #the directory location where all the image are scalled and resized
        os.chdir("c://Users/Shantam Vijayputra/Desktop/Python Projects and Implementations/images")
                        
       #list of all the images
        images = os.listdir()
                        
        #detection of faces in the images
                        
        #location of haarcascade dataset
        face_cascade=cv2.CascadeClassifier('C://Users/Haarcascades_Datasets/haarcascade_frontalface_default.xml')
                        
        for image in images:
                        
            #reading the image
            img = cv2.imread(image)
                        
            #conversion of grayscale
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #will display another window with same vedio feed but grey in color .You can always use different 								option and for that refer the official Documentations.
            #detection of face coordinates
                        
            faces=face_cascade.detectMultiScale(gray,1.3,3)
                        
            #detection of eye and creating the rectangle box around faces and eye
                        
            for (x,y,w,h) in faces:

                img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),5)
                        
                #croping the image contains only face
                roi_gray = gray[x:x+w,y:y+h]
                roi_color = img[x:x+w,y:y+h]
                crop_img = img[y:y+h, x:x+w]
                        
                #saving the croped face image
                cv2.imwrite('C:/Users/Shantam Vijayputra/Desktop/Python Project and Implemetations/cropedimages/'+str(image),crop_img)
        
        #returning to the main directory
        return os.chdir('c://Users/Shantam Vijayputra/Desktop/Python Projects and Implementations')

    
    #calling the classmethod and defing function that convert image to gray
    @classmethod
    def clor2gray(self):
        
        #location of the images
        os.chdir('c://Users/Shantam Vijayputra/Desktop/Python Projects and Implementations/cropedimages/')

        #list of croped images
        crops = os.listdir()
                        
        #looping through all the images
        for crop in crops :
                        
            #reading the image
            img = cv2.imread(crop)
                        
            #conversion of the image
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                        
            #saving the converted image
            cv2.imwrite('C:/Users/Shantam Vijayputra/Desktop/Python Projects and Implementations/greyed/'+str(crop),img)
        
        #returning to the main director
        return os.chdir('c://Users/Shantam Vijayputra/Desktop/Python Projects and Implementations')


#calling the main method
if __name__ == "__main__":

    #printing doc
    print(__doc__)
    
    #object creation
    obj = Imagescale()
    
    #calling the object functionalities.
    obj.resizeimage()
    obj.detectfaces_crop()
    obj.color2gray()
    
