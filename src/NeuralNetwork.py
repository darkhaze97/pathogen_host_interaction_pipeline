import os
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
# os.add_dll_directory("C:/cuDNN/cuda/bin")
import tensorflow as tf
from tensorflow.nn import local_response_normalization
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
from keras.optimizers import Adam
from tensorflow.keras import layers
import numpy as np
import cv2

# The image preprocessing code was adapted from:
# https://www.youtube.com/watch?v=_L2uYfVV48I&list=PLZbbT5o_s2xrwRnXk_yCPtnqqo4_u2YGL&index=11
print(os.getcwd())
trainPath = os.getcwd() + '/data/train'
validPath = os.getcwd() + '/data/validation'
testPath = os.getcwd() + '/data/test'

trainBatch = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input) \
             .flow_from_directory(directory=trainPath,
                                  target_size=(100, 100),
                                  classes=['artifact', 'noRecruit', 'recruit'],
                                  batch_size=10)
validBatch = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input) \
             .flow_from_directory(directory=validPath,
                                  target_size=(100, 100),
                                  classes=['artifact', 'noRecruit', 'recruit'],
                                  batch_size=10)
testBatch = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input) \
            .flow_from_directory(directory=testPath,
                                 target_size=(100, 100),
                                 classes=['artifact', 'noRecruit', 'recruit'],
                                 batch_size=10)
print(trainBatch)

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

model.compile(optimizer=Adam(learning_rate=0.0001),
             loss='categorical_crossentropy',
             metrics=['accuracy'])

# model.fit(x=trainBatch, validation_data=validBatch, epochs=100, verbose=2)