import os
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
# os.add_dll_directory("C:/cuDNN/cuda/bin")
import tensorflow as tf
from tensorflow.nn import local_response_normalization
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras import layers
import numpy as np
import cv2

# pathNames = []
# image = cv2.imread('./data/recruit.png')

model = Sequential()
firstConvLayer = Conv2D(filters=96,
                 kernel_size=(8, 8),
                 activation='relu',
                 padding='valid',
                 input_shape=(100, 100, 3),
                 strides=(2,2)
                )
model.add(firstConvLayer)
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(5, 5), strides=2))
model.add(Conv2D(filters=256,
                 kernel_size=(4,4),
                 activation='relu',
                 padding='valid',
                 strides=(1,1)
                )
         )
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3), strides=1))
model.add(Conv2D(filters=384,
                 kernel_size=(11,11),
                 activation='relu',
                 padding='valid',
                 strides=1
                )
         )
model.add(Conv2D(filters=384,
                 kernel_size=(1,1),
                 activation='relu',
                 padding='valid',
                 strides=1
                )
         )
model.add(Conv2D(filters=256,
                 kernel_size=(1,1),
                 activation='relu',
                 padding='valid',
                 strides=1
                )
         )
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3), strides=1))
model.add(Flatten())
model.add(Dense(units=4096, activation='relu'))
model.add(Dense(units=4096, activation='softmax'))

model.summary()