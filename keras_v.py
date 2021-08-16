# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 14:28:13 2020

@author: Lenovo
"""

import tensorflow as tf
import preprocess as pp
import numpy as np
a=pp.preprocess()
a.process()
a.image_50()
a.siglayer(level=1)
acc_list=[]


train=np.reshape(a.collection_50_train,[7494,50,50,1])
siglayer_train=a.sigcollection_train
train_sample=np.append(siglayer_train,train,axis=3)
train_index=np.eye(10)[a.index_train.astype(np.int8)]

test=np.reshape(a.collection_50_test,[3498,50,50,1])
siglayer_test=a.sigcollection_test
test_sample=np.append(siglayer_test,test,axis=3)
test_index=np.eye(10)[a.index.astype(np.int8)]


#input_x=tf.placeholder(tf.float32,[None,50,50,3])
#output_y=tf.placeholder(tf.int32,[None,10])

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (5,5), activation='relu',padding='same',
                         strides=1,input_shape=(50, 50, 3)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(32, (2,2), activation='relu', strides=1,
    padding='valid'),
  tf.keras.layers.MaxPooling2D(3,3),
  tf.keras.layers.Conv2D(16, (5,5), activation='relu',padding='same',
                         strides=1,input_shape=(50, 50, 3)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(1024, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(train_sample, train_index, epochs=10, batch_size=16,validation_data=(test_sample,test_index) )
#test_loss = model.evaluate(test_sample, test_index)

history_dict = history.history
print(history_dict.keys())
# >>dict_keys(['loss', 'val_loss', 'val_acc', 'acc'])

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']