"""
Created on Mon Aug 19 13:42:42 2019

@author: haggarwal
"""


import tensorflow as tf
import numpy as np
import misc as sf
from os.path import expanduser
home = expanduser("~")


def convLayer(x, szW,training,i):
    with tf.name_scope('layers'):
        with tf.variable_scope('lay'+str(i)):
            W=tf.get_variable('W',shape=szW,initializer=tf.contrib.layers.xavier_initializer())
            y = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
            if training!='linear':
                y=tf.nn.relu(y)
    return y


def smallUnet(inp,C,training):
    with tf.name_scope('Unet'):
        x=convLayer(inp,(3,3,C,32),training,1)
        x=convLayer(x,(3,3,32,32),training,2)
        x1=convLayer(x,(3,3,32,32),training,3)
        p1=tf.nn.avg_pool(x,[1,2,2,1],[1,2,2,1],'SAME')

        x=convLayer(p1,(3,3,32,64),training,4)
        x=convLayer(x,(3,3,64,64),training,5)
        x2=convLayer(x,(3,3,64,64),training,6)
        p2=tf.nn.avg_pool(x2,[1,2,2,1],[1,2,2,1],'SAME')

        x=convLayer(p2,(3,3,64,96),training,7)
        x=convLayer(x,(3,3,96,96),training,8)
        x=convLayer(x,(3,3,96,96),training,9)

        x=tf.image.resize_nearest_neighbor(x,tf.constant([128,128]))
        x=tf.concat([x2,x],axis=-1)
        x=convLayer(x,(3,3,160,96),training,10)
        x=convLayer(x,(3,3,96,96),training,11)
        x=convLayer(x,(3,3,96,96),training,12)

        x=tf.image.resize_nearest_neighbor(x,tf.constant([256,256]))
        x=convLayer(x,(3,3,96,64),training,13)
        x=convLayer(x,(3,3,64,64),training,14)
        x=convLayer(x,(3,3,64,64),training,15)


        x=tf.concat([x1,x],axis=-1)
        x=convLayer(x,(3,3,96,64),training,16)
        x=convLayer(x,(3,3,64,32),training,17)
        x=convLayer(x,(3,3,32,32),training,18)
        x=convLayer(x,(1,1,32,C//2),'linear',19)

    return x

def bigUnet(inp,C,training):
    with tf.name_scope('Unet'):
        x=convLayer(inp,(3,3,C,32),training,1)
        x=convLayer(x,(3,3,32,32),training,2)
        x1=convLayer(x,(3,3,32,32),training,3)
        p1=tf.nn.avg_pool(x1,[1,2,2,1],[1,2,2,1],'SAME')

        x=convLayer(p1,(3,3,32,64),training,4)
        x=convLayer(x,(3,3,64,64),training,5)
        x2=convLayer(x,(3,3,64,64),training,6)
        p2=tf.nn.avg_pool(x2,[1,2,2,1],[1,2,2,1],'SAME')

        x=convLayer(p2,(3,3,64,128),training,7)
        x=convLayer(x,(3,3,128,128),training,8)
        x3=convLayer(x,(3,3,128,128),training,9)
        p3=tf.nn.avg_pool(x3,[1,2,2,1],[1,2,2,1],'SAME')

        x=convLayer(p3,(3,3,128,256),training,10)
        x=convLayer(x,(3,3,256,256),training,11)
        x=convLayer(x,(3,3,256,256),training,12)


        x=tf.image.resize_nearest_neighbor(x,tf.constant([64,64]))
        x=tf.concat([x3,x],axis=-1)
        x=convLayer(x,(3,3,384,128),training,13)
        x=convLayer(x,(3,3,128,128),training,14)
        x=convLayer(x,(3,3,128,128),training,15)


        x=tf.image.resize_nearest_neighbor(x,tf.constant([128,128]))
        x=tf.concat([x2,x],axis=-1)
        x=convLayer(x,(3,3,192,64),training,16)
        x=convLayer(x,(3,3,64,64),training,17)
        x=convLayer(x,(3,3,64,64),training,18)

        x=tf.image.resize_nearest_neighbor(x,tf.constant([256,256]))
        x=tf.concat([x1,x],axis=-1)
        x=convLayer(x,(3,3,96,32),training,19)
        x=convLayer(x,(3,3,32,32),training,20)
        x=convLayer(x,(3,3,32,32),training,21)

        x=convLayer(x,(1,1,32,C),'linear',22)

    return x
