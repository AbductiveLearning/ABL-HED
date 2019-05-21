import os
import numpy as np
import keras
import pickle
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, Reshape, UpSampling2D
from keras import regularizers
from keras import optimizers
from PIL import Image
from functools import partial
import sys



def get_LeNet5_net(labels_num, input_shape = (28, 28, 1)):
    h = input_shape[0]
    w = input_shape[1]
    d = input_shape[2]
    model = Sequential()
    model.add(Conv2D(input_shape=(h, w, d), kernel_size=(5, 5), filters=32, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))
    model.add(BatchNormalization()) # BN is good 
    model.add(Conv2D(kernel_size=(5, 5), filters=64,  activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))
    model.add(BatchNormalization()) # BN is useful
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(labels_num, activation='softmax'))
    return model
    
def get_LeNet5_autoencoder_net(labels_num, input_shape = (28, 28, 1)):
    h = input_shape[0]
    w = input_shape[1]
    d = input_shape[2]
    model = get_LeNet5_net(labels_num, input_shape)
    model.add(Dense(10 * 10, activation='relu'))
    model.add(Dense(h * w *d, activation='relu'))
    return model

def get_cifar10_net(labels_num, input_shape = (28, 28, 3)):
    h = input_shape[0]
    w = input_shape[1]
    d = input_shape[2]
    
    weight_decay = 1e-4
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=(h, w, d)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
     
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
     
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
     
    model.add(Flatten())
    model.add(Dense(labels_num, activation='softmax'))
    

    '''
    model = Sequential()
    model.add(Conv2D(input_shape=(h, w, d), kernel_size=(5, 5), filters=32, activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=2, padding='same'))
    model.add(BatchNormalization()) # BN is good 
    model.add(Conv2D(kernel_size=(5, 5), filters=32,  activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=2, padding='same'))
    model.add(BatchNormalization()) # BN is good 
    model.add(Conv2D(kernel_size=(5, 5), filters=64,  activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=2, padding='same'))
    model.add(BatchNormalization()) # BN is good 
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(labels_num, activation='softmax'))
    '''
    return model

    
    
    
    
def get_autoencoder_net(labels_num, input_shape = (32, 32, 3)):
    h = input_shape[0]
    w = input_shape[1]
    d = input_shape[2]
    
    weight_decay = 1e-4
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=(h, w, d)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0))
     
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0))
     
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0))
     
    model.add(Flatten())
    model.add(Dense(labels_num))
    
    model.add(Dense(4*4*128))
    model.add(Reshape((4, 4, 128)))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(3, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    return model
    
    
def get_mlp_net(logic_output_dim):
    model = Sequential()
    model.add(Dense(int(np.sqrt(logic_output_dim)), input_dim=logic_output_dim, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model
