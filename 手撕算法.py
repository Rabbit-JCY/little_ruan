# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 11:25:29 2021

@author: Rabbit
"""

#导入相关包
import tensorflow as tf
import getData
import os
import matplotlib.pyplot as plt


#加载处理数据集
x_train,y_train,x_test,y_test = [],[],[],[]
x_train,y_train,x_test,y_test = getData.getData()

x_train = tf.constant(x_train, dtype = tf.float32)
y_train = tf.constant(y_train*10, dtype = tf.float32)
y_train = tf.reshape(y_train,(120,1))

x_test = tf.constant(x_test, dtype = tf.float32)
y_test = tf.constant(y_test, dtype = tf.float32)
y_test = tf.reshape(y_test,(30,1))

#设置超参数
w = tf.ones([4,1])
w = tf.Variable(tf.cast(w, dtype = tf.float32))
b = tf.constant(1,dtype = tf.float32)
b = tf.Variable(tf.constant(b))

alpha = 0.001

for i in range(300):
    with tf.GradientTape() as tape:
        loss = abs(sum(tf.matmul(x_train, w) + b - y_train))/len(x_train)
        grad = tape.gradient(loss,(w,b))
        
        w.assign_sub(grad[0]*alpha)
        b.assign_sub(grad[1]*alpha)
        
        print("loss = " , loss.numpy())
        
        
    