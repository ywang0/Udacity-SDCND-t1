#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ast
import csv
import cv2
import math
import numpy as np
import os
import tensorflow as tf
from itertools import product
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential, Model
from keras.layers import (Activation, Flatten, Conv2D, MaxPooling2D, Dense, Dropout,
                          Lambda, Cropping2D, Input, Merge, BatchNormalization)
from keras.layers.merge import Concatenate
from keras.optimizers import Adam, SGD, Nadam
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

#%%

np.random.seed(42)
tf.set_random_seed(42)

#%%
data_dir = 'data/'
log_sources = ['udacity', 'counter_clockwise', 'clockwise',
               'curves', 'recovery', 'turn_after_bridge_2',
               'track2', 'track2_opposite', 'track2_recovery',
               'track2_curves', 'track2_curves_2',
              ]
#log_csv = 'driving_log.csv'
log_csv = 'processed_driving_log.csv'


#%%
def get_log_data(sources):
    data = []
    for source in sources:
        log = os.path.join(data_dir, source, log_csv)
        with open(log) as f:
            reader = csv.reader(f)
            next(reader)    # skip header
            for row in reader:
                data.append([ele.strip() for ele in row])

    return data

#%%
def random_CLF_image(log, correction):
    """
    Return the image andomly selected from 'center', 'left', or 'right' camera.

    Args:
        log (list): A csv parsed list containing the recored data for a video
                    frame, ['center', 'left', 'right', 'steering', 'throttle',
                    'brake', 'speed', 'pre_state', 'avg_steering']
        correction (float): The amount of adjustment if 'left' or 'right' camera
                    is selected.
    Returns:
        The image selected
    """
    steering = float(log[3])
#    which = np.random.choice(3, p=[0.5, 0.25, 0.25])
    which = np.random.randint(3)
    image = cv2.imread(log[which])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if which == 1:
        steering += correction
    elif which == 2:
        steering -= correction

    return image, steering


def random_brightness(image, britness_range=0.5):
    """
    Apply random brightness/darkness to the input image and return the
    resulting image.
    """
    image_ = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_britness = np.random.uniform(1.-britness_range, 1.+britness_range)
    image_[:, :, 2] = np.clip(image_[:, :, 2] * random_britness, 0, 255)
    image_ = cv2.cvtColor(image_, cv2.COLOR_HSV2RGB)

    return image_


def random_flip(image, steering):
    """
    Horizontally flip the input image with the probability of 50% and return
    the resulting image.
    """
    image_ = image.copy()
    if np.random.randint(2):
        image_ = cv2.flip(image_, 1)
        steering = -steering

    return image_, steering

#%%
def data_generator(logs, batch_size=32, zero_drop_prob=0., correction=0.2,
                   training=False, need_shuffle=True):
    """
    Return a data generator. The generator generates pairs of (features, labels)
    where features consists of input images and previous states associated with
    the input images and the labels are steerings.

    Args:
        logs (list of lists): List of csv parsed lists containing the recored
                            data for a video frame.
        zero_drop_prob (float): The percentage of data with almost zero steering
                            angle to be dropped
        correction (float): The amount of steering angle to be adjust for 'left'
                            or 'right' camera images
        training (boolean): True for training steps, False otherwise
        need_shuffle (boolean): The dataset (i.e., logs) will be shuffled at the
                            beginning of each epoch
    """

    n_samples = len(logs)
    if training:
        while 1:
            if need_shuffle:
                logs = np.random.permutation(logs)

            # the following logic is to keep each batch returned has the same
            # batch_size. we don't do this for validation dataset since we don't
            # drop samples in validation steps
            offset = 0
            while offset < n_samples:
                images, steerings, states = [], [], []
                n_2_fill = batch_size
                i = 0
                while n_2_fill and (offset+i) < n_samples:
                    log = logs[offset+i]
                    image, steering = random_CLF_image(log, correction=correction)
                    pre_state = ast.literal_eval(log[7])
                    i += 1
                    if not np.isclose(abs(steering), 0) or \
                            np.random.uniform() > zero_drop_prob:
                        image = random_brightness(image)
                        image, steering = random_flip(image, steering)
                        images.append(image)
                        steerings.append(steering)
                        states.append(pre_state)
                        n_2_fill -= 1
                offset += i
                yield [np.array(images), np.array(states)], np.array(steerings)
    else:
        while 1:
            if need_shuffle:
                logs = np.random.permutation(logs)

            for offset in range(0, n_samples, batch_size):
                images, steerings, states = [], [], []

                batch = logs[offset: offset+batch_size]
                for log in batch:
                    image, steering = random_CLF_image(log, correction=correction)
                    pre_state = ast.literal_eval(log[7])
                    images.append(image)
                    steerings.append(steering)
                    states.append(pre_state)
                yield [np.array(images), np.array(states)], np.array(steerings)

#%%
def resize_img(input):
    from keras.backend import tf as K
    return K.image.resize_images(input, (96, 96))


def nn_model(image_shape, state_shape):
    """
    The definition of the DNN model.
    There are two types of inputs: images and previous states. The images are
    fed to the network's first conv layer and the previous states are combined
    later with the output of the last conv layer and then fed into the top
    fully-connected layers.

    Args:
        images_shape (int tuple): The shape of input image, e.g., (160, 320, 3)
        state_shape (float tuple): The previous state consists of steering,
        throttle, and speed

    Returns:
        The DNN model
    """

    image_inputs = Input(shape=image_shape, name='input_images')
    x = Cropping2D(cropping=((30,20), (0,0)), name='cropping')(image_inputs)
    x = Lambda(resize_img, name='resize')(x)
    x = Lambda(lambda x: (x / 127.) - 1., name='normalization')(x)
    conv1 = Conv2D(16, (3, 3), activation='elu', padding='same', name='conv_1')(x)
#    conv1 = BatchNormalization()(conv1)
#    conv1 = Activation(activation='elu)(conv1)
    pool1 = MaxPooling2D(name='pool_1')(conv1)
    conv2 = Conv2D(32, (3, 3), activation='elu', padding='same', name='conv_2')(pool1)
#    conv2 = BatchNormalization()(conv2)
#    conv2 = Activation(activation='elu', name='conv2')(conv2)
    pool2 = MaxPooling2D(name='pool_2')(conv2)
    conv3 = Conv2D(32, (3, 3), activation='elu', padding='same', name='conv_3')(pool2)
#    conv3 = BatchNormalization()(conv3)
#    conv3 = Activation(activation='elu', name='conv3')(conv3)
    pool3 = MaxPooling2D(name='pool_3')(conv3)
    flat = Flatten(name='flatten')(pool3)
    flat = Dropout(0.2)(flat)

    state_inputs = Input(shape=state_shape, name='input_states')
    merged = Concatenate(name='merged')([flat, state_inputs])
    merged = BatchNormalization(name='batch_norm')(merged)

    fc1 = Dense(64, activation='elu', name='fc_1')(merged)
    fc1 = Dropout(0.2)(fc1)
    fc2 = Dense(10, activation='elu', name='fc_2')(fc1)
    fc2 = Dropout(0.2)(fc2)
    out = Dense(1, name='output')(fc2)

    nadam = Nadam(lr=0.0001)
    model = Model(inputs=[image_inputs, state_inputs], outputs=out)
    model.compile(optimizer=nadam, loss='mse')

    return model

#%%
class SaveModel(keras.callbacks.Callback):
    """
    A callback class to save model and weights after each epoch.
    """
    def __init__(self, model_dir):
        self.epoch_id = 0
        self.model_dir = model_dir
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)

    def on_epoch_end(self, batch, logs={}):
        out_model = os.path.join(self.model_dir,
                                 "model_epoch_{}.h5".format(self.epoch_id))
        out_weights = os.path.join(self.model_dir,
                                   "model_weights_epoch_{}.h5" \
                                   .format(self.epoch_id))
        self.model.save(out_model)
        self.model.save_weights(out_weights)
        self.epoch_id += 1


if __name__ == '__main__':

    # load the log data from csv files and split into training and test sets
    log_data = get_log_data(log_sources)
    log_data = np.random.permutation(log_data)
    train_logs, valid_logs = train_test_split(log_data, test_size=0.2,
                                              random_state=42)
    print("Number of training samples: {}".format(len(train_logs)))
    print("Number of validation samples: {}".format(len(valid_logs)))

    # get image shape and state shape
    sample_gen = data_generator(train_logs, batch_size=1)
    img_batch, label_batch = next(sample_gen)
    image_shape = img_batch[0].shape[1:]
    state_shape = img_batch[1].shape[1:]
    label_shape = label_batch[0].shape

    # define hyperparameters, using lists for grid searching the best paramters
    corrections = [0.2]
    zero_drop_probs = [0.9]
    epochs = [50]

    for zero_drop_prob, correction, epoch in product(
                                    zero_drop_probs, corrections, epochs):
        print("*** zero_drop: {} - correction: {} - epoch: {}".format(
                                            zero_drop_prob, correction, epoch))

        model_dir = 'MODELS_nn16_z{}_c{}_e{}'.format(zero_drop_prob, correction,
                                                     epoch)
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        output_model = os.path.join(model_dir, 'model.h5')
        output_weights = os.path.join(model_dir, 'model_weights.h5')

        # define callbacks
        checkpoint = ModelCheckpoint(os.path.join(model_dir, 'checkpoint.h5'),
                                     save_best_only=True, verbose=1)
        csvlogger = CSVLogger(os.path.join(model_dir, 'training_log.csv'))
        early_stopping = EarlyStopping(monitor="val_loss", patience=3, verbose=1)
        savemodel = SaveModel(model_dir)
        callbacks=[csvlogger, checkpoint, early_stopping, savemodel]

        # create data generators
        train_gen = data_generator(train_logs, zero_drop_prob=zero_drop_prob,
                                   correction=correction, training=True)
        valid_gen = data_generator(valid_logs, correction=correction)

        # create model and fit the model from data generators
        model =  nn_model(image_shape, state_shape)
        print(model.summary())
        model.fit_generator(train_gen,
                            steps_per_epoch=math.ceil(len(train_logs)/32),
                            nb_epoch=epoch,
                            validation_data=valid_gen,
                            validation_steps=math.ceil(len(valid_logs)/32),
                            callbacks=callbacks)

        model.save_weights(output_weights)
        model.save(output_model)

#%%
