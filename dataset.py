# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 19:41:35 2018

@author: Shantam Vijayputra
"""

#importing Libraries.
import cv2
import os

#creating dataset class object.
class Dataset(object):

    #calling init method.
    def __init__(self,l=[],f=[],t=os.listdir()):
        self.l = l
        self.f = f
        self.t = t

    #creating classmethod and defining features dataset.
    @classmethod
    def createdatafeatures(self):
        
        #provide path to your scaled dataset .
        fd = open('c://Users/Shantam Vijayputra/Desktop/Python Projects and Implementations/age_predict--Todo/foo2.csv','a+')
        
        #creating list of images.
        t = os.listdir()
        
        #looping through all the images and extracting features.
        for k in t:
            
            #reading image and conversion to greay scale image.
            img = cv2.imread(k,0)
            
            #image resizing
            img = cv2.resize(img,(10,10))
            
            #conversion of image matrix to aimage array.
            img = img.flatten().reshape(-1,1).transpose()
            
            #writing each array to a csv file.
            for i in img[0]:
                fd.write(str(i)+str(","))
            fd.write("\n")        

        #returning and closing the file.
        return fd.close()
    
    #creating classmethod and creating output dataset.
    @classmethod
    def createdataoutput(self):
        
        #providing location of output dataset csv file.
        fe = open("c://Users/Shantam Vijayputra/Desktop/Python Projects and Implementations/age_predict--Todo/foo11.csv","a+")
        
        #storing the list of images in the directory.
        t = os.listdir()
        
        #looping through all the mages.
        for i in range(len(t)):
            
            #computing the age provided in the image name and writing n the output csv file.
            fe.write(str(abs(int(t[i].split("_")[1][:4])-int(t[i].split("_")[2][:4]))))
            fe.write("\n")

        #returning and closing the file
        return fe.close()

#main method
if __name__ == "__main__":
    
    #displaying the doc method.
    print(__doc__)
    
    #changing the directory to the grey scaled image folder.
    os.chdir("c://Users/Shantam Vijayputra/Desktop/Python Projects and Implementations/age_predict--Todo/greyed")

    #creating object
    obj = Dataset()
    
    #creating object functions.
    obj.createdatafeatures()
    obj.createdataoutput()
