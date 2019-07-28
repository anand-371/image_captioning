import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import tensorflow as tf
import tensorflow.keras
import numpy as np
import sys
import os
from PIL import Image
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
import keras
from keras.models import model_from_json
from load_data import *
from CNN import *
from RNN import *
from preprocess_text import *
total_train_size=100
num_words = 10000
batch_size=10
trans=[]
count=[0]
data,transfer_values,filenames_train=load_dataset('qwerty.csv')
num_images_train = len(filenames_train)
transfer_values_size=CNN()
transfer_values=transfer_values[:total_train_size]
num_images_train=total_train_size
tokens_train=preprocess(data,total_train_size,num_words)
decoder_model,generator,steps_per_epoch=RNN(num_words,batch_size,tokens_train,transfer_values,transfer_values_size,total_train_size)
optimizer = RMSprop(lr=1e-3)  
decoder_target = tf.placeholder(dtype='int32', shape=(None, None))  
decoder_model.compile(optimizer=optimizer,loss=sparse_cross_entropy,target_tensors=[decoder_target])
optimizer = RMSprop(lr=1e-3)  
decoder_target = tf.placeholder(dtype='int32', shape=(None, None))  
decoder_model.compile(optimizer=optimizer,loss=sparse_cross_entropy,target_tensors=[decoder_target])
decoder_model.fit_generator(generator=generator,steps_per_epoch=steps_per_epoch,epochs=20)