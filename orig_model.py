# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:22:01 2020

@author: Lenovo
"""
import tensorflow as tf
import preprocess as pp
import numpy as np
a=pp.preprocess()
a.process()
a.image_50()
acc_list=[]


train_sample=np.reshape(a.collection_50_train,[7494,50,50,1])

train_index=np.eye(10)[a.index_train.astype(np.int8)]

test_sample=np.reshape(a.collection_50_test,[3498,50,50,1])

test_index=np.eye(10)[a.index.astype(np.int8)]


input_x=tf.placeholder(tf.float32,[None,50,50,1])
output_y=tf.placeholder(tf.int32,[None,10])


conv1=tf.layers.conv2d(
    inputs=input_x,
    filters=30,
    kernel_size=[3,3],
    strides=3,
    padding='valid',
)

pool1=tf.layers.max_pooling2d(
    inputs=conv1,
    pool_size=[2,2],
    strides=2
)

conv2=tf.layers.conv2d(
    inputs=pool1,
    filters=60,
    kernel_size=[2,2],
    strides=1,
    padding='same',
    activation=tf.nn.relu
)

pool2=tf.layers.max_pooling2d(
    inputs=conv2,
    pool_size=[2,2],
    strides=2
)

conv3=tf.layers.conv2d(
    inputs=pool2,
    filters=90,
    kernel_size=[2,2],
    strides=1,
    padding='same',
    activation=tf.nn.relu
)

pool3=tf.layers.max_pooling2d(
    inputs=conv3,
    pool_size=[2,2],
    strides=2
)

conv4=tf.layers.conv2d(
    inputs=pool3,
    filters=120,
    kernel_size=[2,2],
    strides=1,
    padding='same',
    activation=tf.nn.relu
)

pool4=tf.layers.max_pooling2d(
    inputs=conv4,
    pool_size=[2,2],
    strides=2
)
#conv1=tf.layers.conv2d(
#    inputs=input_x,
#    filters=16,
#    kernel_size=[5,5],
#    strides=1,
#    padding='same',
#)
#
#pool1=tf.layers.max_pooling2d(
#    inputs=conv1,
#    pool_size=[2,2],
#    strides=2
#)
#
#conv2=tf.layers.conv2d(
#    inputs=pool1,
#    filters=32,
#    kernel_size=[2,2],
#    strides=1,
#    padding='valid',
#)
#
#pool2=tf.layers.max_pooling2d(
#    inputs=conv2,
#    pool_size=[3,3],
#    strides=3
#)
#
#conv3=tf.layers.conv2d(
#    inputs=pool2,
#    filters=64,
#    kernel_size=[5,5],
#    strides=1,
#    padding='same',
#)
#
#pool3=tf.layers.max_pooling2d(
#    inputs=conv3,
#    pool_size=[2,2],
#    strides=2
#)
#
#flat=tf.reshape(pool3,[-1,4*4*64])
#
#dense=tf.layers.dense(
#    inputs=flat,
#    units=1024,
#    activation=tf.nn.relu
#)
flat=tf.reshape(pool4,[-1,120])
dense=tf.layers.dense(
    inputs=flat,
    units=150,
    activation=tf.nn.relu
)
#dropout=tf.layers.dropout(
#    inputs=dense,
#    rate=0.5,
#)
#

logits=tf.layers.dense(
    inputs=dense,
    units=10
)

loss=tf.losses.softmax_cross_entropy(onehot_labels=output_y,
                                     logits=logits)
train_op=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
#train_op=tf.train.GradientDescentOptimizer(learning_rate=0.03).minimize(loss)

accuracy_op=tf.metrics.accuracy(
    labels=tf.argmax(output_y,axis=1),
    predictions=tf.argmax(logits,axis=1))[1]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for i in range(500000):
        k=np.random.randint(low=0,high=7494,size=10)
        train_loss,_=sess.run([loss,train_op], feed_dict={
                input_x:train_sample[k,:,:,:] , output_y:train_index[k,:]})
        if i%1000==0:
            test_accuracy=sess.run(accuracy_op,{input_x:test_sample,output_y:test_index})
            acc_list.append(test_accuracy)
            print("Step=%d, Train loss=%.4f,[Test accuracy=%.4f]"%(i,train_loss,test_accuracy))
            