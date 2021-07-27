# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 17:16:34 2021

@author: Rabbit
"""

import os
import numpy as np
import sklearn

x_train_savepath = "F:/神经网络/鸢尾花/x_train.npy"
y_train_savepath = "F:/神经网络/鸢尾花/y_train.npy"
x_test_savepath = "F:/神经网络/鸢尾花/x_test.npy"
y_test_savepath = "F:/神经网络/鸢尾花/y_test.npy"

def getData():
    
    x_train,y_train = [],[]
    x_test,y_test = [],[]
    
    if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath)and os.path.exists(x_test_savepath) and os.path.exists(y_test_savepath):
        x_train = np.load(x_train_savepath)
        y_train = np.load(y_train_savepath)
        x_test = np.load(x_test_savepath)
        y_test = np.load(y_test_savepath)
        
        
    else:
        data = sklearn.datasets.load_iris()
        
        feature = data["data"]
        target = data["target"]
        
        np.random.seed(116)
        np.random.shuffle(feature)
        np.random.seed(116)
        np.random.shuffle(target)
        
        x_train = feature[:-30]
        y_train = target[:-30]
        x_test = feature[-30:]
        y_test = target[-30:]
        
        np.save(x_train_savepath,x_train)
        np.save(y_train_savepath,y_train)
        np.save(x_test_savepath,x_test)
        np.save(y_test_savepath,y_test)
        
    return x_train,y_train,x_test,y_test
        
        
            
        
    