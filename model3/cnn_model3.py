# encoding: utf-8
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.layers import concatenate, Activation, BatchNormalization
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Dropout, Dense, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from keras.models import Model

def create_1dcnn(input_shape):

    model = tf.keras.Sequential([
        # cov1
        Conv1D(16, kernel_size=3, input_shape=input_shape,padding='same', name = '1dcov1'),
        BatchNormalization(),
        Activation(tf.nn.relu),
        MaxPooling1D(pool_size=2),
        # cov2
        Conv1D(32, kernel_size=3,padding='same',name = '1dcov2'),
        BatchNormalization(),
        Activation(tf.nn.relu),
        MaxPooling1D(pool_size=2),
        # cov3
        Conv1D(64, kernel_size=3, padding='same', name='1dcov3'),
        BatchNormalization(),
        Activation(tf.nn.relu),
        MaxPooling1D(pool_size=2),
        # fc
        Flatten(),
        Dense(32),
        BatchNormalization(),
        Activation(tf.nn.relu)
        # Dropout
        #Dropout(0.5)
        # Dense(classes, activation=tf.nn.softmax)
    ])

    return model

def create_3dcnn(input_shape):

    model = tf.keras.Sequential([
        # cov1+maxpooling
        Conv3D(16, kernel_size=(3, 3, 3), input_shape=input_shape, strides=(2, 2, 2), padding='same',name='3dcov1'),
        BatchNormalization(),
        Activation(tf.nn.relu),
        MaxPooling3D(pool_size=2),
        # cov2+maxpooling
        Conv3D(32, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same',name='3dcov2'),
        BatchNormalization(),
        Activation(tf.nn.relu),
        MaxPooling3D(pool_size=2),
        # cov3+maxpooling
        Conv3D(64, kernel_size=(3, 3, 3), padding='same',name='3dcov3'),
        BatchNormalization(),
        Activation(tf.nn.relu),
        MaxPooling3D(pool_size=2),
        # fc
        Flatten(),
        Dense(1024),
        Dense(128),
        BatchNormalization(),
        Activation(tf.nn.relu),
        # Dropout
        #Dropout(0.5),
        # Dense(classes, activation=tf.nn.softmax)
    ])

    return model

def cnn_model3(channel, height, width, depth, classes):

    # define two sets of inputs
    input1_shape=(height, width, depth, channel)
    input2_shape=(depth, channel)

    model1 = create_3dcnn(input1_shape)
    model2 = create_1dcnn(input2_shape)

    combined = concatenate([model1.output, model2.output])

    # apply a FC layer and then a regression prediction on the
    # combined outputs
    model = Dense(160, activation="relu")(combined)
    model = Dense(64, activation="relu")(model)
    model = Dropout(0.5)(model)
    model = Dense(classes, activation=tf.nn.softmax)(model)

    # our model will accept the inputs of the two branches and
    # then output a single value
    model = Model(inputs=[model1.input, model2.input], outputs=model)

    return model