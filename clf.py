# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 23:29:56 2018

@author: Shantam Vijayputra
"""

#importing dataset
import pandas as pd
import pickle
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LinearRegression


#creating classmethod
class classifier(object):

    #calling classmethod and defing function to load dataframe
    @classmethod
    def dataload(self,path):
        
        #loading csv dataset for featurs and output
        return pd.read_csv(path)

    #calling claamethod and defining function to load values from dataframe
    @classmethod
    def loadvalues(self,df1,df2):
        
        #extracting the values
        x = df1.iloc[:,:-1].values
        y = df2.iloc[:,:].values

        #returning loaded values
        return x,y

    #calling clasmethod and defining function for split data to train and test
    @classmethod
    def splitdata(self,x,y,size):
        
        #splitting the dataset into training and testing
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = size, random_state = 0)

        #returning training and testing data
        return X_train, X_test, y_train, y_test


    #calling classmethod for scaling features
    @classmethod
    def scale(self,x):
        
        # Feature Scaling
        sc= StandardScaler()
        x = sc.fit_transform(x)

        #returnng scaled data
        return x

    #calling classmethod and defining function for dimensionality reduction
    @classmethod
    def dimreduction(self,x,n):
        

        #Dimensionality reduction
        kpca = KernelPCA(n_components = n, kernel = 'rbf')
        x = kpca.fit_transform(x)

        #returning data with reduced features
        return x
    

    #calling classmethod for defining clasiffier
    @classmethod
    def clf(self,X_train,y_train):
        
        # Fitting Simple Linear Regression to the Training set
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)

        #returning classifer
        return regressor

    #calling classmethod to define funtion for saving data
    @classmethod
    def save_clf(self,clf,path_with_name,_type):
        
        #dumping the data
        return pickle.dump(clf, open(path_with_name, _type))


#main method
if __name__ == "__main__":

    #printing the doc
    print(__doc__)

    #creating object
    obj = classifier()

    #providing path
    path1 = ""
    path2 = ""

    #loading dataframe
    df1 = obj.dataload(path1)
    df2 = obj.dataload(path2)

    #loading values
    x,y = obj.loadvalues(df1,df2)

    #size of split data
    size = 1/3

    #splitting the data
    X_train, X_test, y_train, y_test = obj.splitdata(x,y,size)

    #scaing the data
    X_train = obj.scale(X_train)
    X_test  = obj.scale(X_test)

    """
    #feature reduction
    X_train = obj.dimreduction(X_train,20)
    X_test  = obj.dimreduction(X_test,20)
    """

    #defining classifer
    clf = obj.clf(X_train,y_train)

    #path_with_name
    path_with_name = ""

    #type
    _type = ""
    
    #saving the clf method
    obj.save_clf(clf,path_with_name,_type)
    

