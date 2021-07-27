# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 16:12:00 2021

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


#搭建神经网络结构
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

#设置网络参数
model.compile(optimizer=tf.keras.optimizers.Adam(1e-6, 0.9, 0.999),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

#断点续训
checkpoint_save_path = "F:/神经网络/鸢尾花/model.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print("进行断点续训")
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

#训练神经网络
history = model.fit(x_train, y_train, batch_size=8, epochs=20, validation_data=(x_test,y_test), validation_freq=1, callbacks=[cp_callback])

#网络概述
model.summary()

#训练过程可视化
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
