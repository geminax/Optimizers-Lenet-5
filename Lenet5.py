import time
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dense, Flatten
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def Lenet5(in_shape,opt):
    inputs = Input(shape=(in_shape))
    c1 = Conv2D(filters=6,kernel_size=(5,5),strides=1,padding='same',activation='relu')(inputs)
    mp1 = MaxPool2D(pool_size=(2,2),strides=2)(c1)
    c2 = Conv2D(filters=16,kernel_size=(5,5),strides=1,activation='relu')(mp1)
    mp2 = MaxPool2D(pool_size=(2,2),strides=2)(c2)
    flt = Flatten()(mp2)
    d1 = Dense(120,activation='relu')(flt)
    d2 = Dense(84,activation='relu')(d1)
    outputs = Dense(10,activation='softmax')(d2)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='Lenet-5')
    model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])
    return model

def main():
    #Load Data
    print('---------------')
    print('Loading Data ...')
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    print('Data Loaded.\n---------------')

    #Preprocessing
    print('Preprocessing Data...')
    x_train = x_train.astype('float32') / 255 #scale to [0,1]
    x_test = x_test.astype('float32') / 255

    x_train = np.expand_dims(x_train, -1) #add trailing dim [1] for each pixel value
    x_test = np.expand_dims(x_test, -1)

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test,10)
    print('Done Preprocessing.\n---------------')

    #Parameters
    input_shape = (28,28,1)
    batch_size = 64
    epochs = 100

    #Optimizers - Change this to whichever optimizer
    # SGD
    # Adagrad
    # Adadelta
    # RMSprop
    # Adam
    # Adamax
    # Nadam
    # with parameters such as momentum and nesterov=bool for SGD
    optimizer = keras.optimizers.SGD(learning_rate=0.01,momentum=0.9,nesterov=False)

    #Build
    model = Lenet5(input_shape, optimizer)

    #Training
    print('Training...')
    callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)
    time0 = time.time()
    model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, callbacks=[callback])
    time1 = time.time()
    print('Finished Training - Time: %s seconds' % (round(time1-time0,4)))
    print('------------------------------------------')

    #Testing
    print('Testing...')
    eval = model.evaluate(x_test,y_test)
    print("Accuracy - ", round(eval[1] * 100,4) , '%')

if __name__ == '__main__':
    main()