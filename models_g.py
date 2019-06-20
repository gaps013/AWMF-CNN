# coding=utf-8
import os
import sys
from random import randint
from time import sleep

import keras
import numpy as np
import tensorflow as tf
from GPUtil import GPUtil

from keras.models import Model
from keras.applications.xception import Xception
from keras.layers.merge import concatenate
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation
from keras.utils import multi_gpu_model, plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, multiply, Lambda, K, BatchNormalization, Conv2D, UpSampling2D, Cropping2D, Convolution2D
from keras.backend.tensorflow_backend import set_session

from deeplabv3plus_model import Deeplabv3  # https://github.com/tensorflow/models/tree/master/research/deeplab
from networks_util_g import AnotherModelCheckpointCallback, extended_to_categorical, dice_coef_op, MaxPoolingWithArgmax2D, MaxUnpooling2D


def set_session_gpu(G, allow_growth=False):
    if G >= 1:
        # Set CUDA_DEVICE_ORDER so the IDs assigned by CUDA match those from nvidia-smi
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        first_flag = True
        # Get the first available GPU
        while True:
            DEVICE_ID_LIST = GPUtil.getAvailable(order='last', limit=G, maxLoad=0.5, maxMemory=0.5, includeNan=False)
            if DEVICE_ID_LIST != []:
                break
            if first_flag:
                print("I'm waiting....")
                first_flag = False
            sleep(randint(300, 600))

        DEVICE_ID_LIST = [str(x) for x in DEVICE_ID_LIST]
        DEVICE_ID = ",".join(DEVICE_ID_LIST)  # grab first element from list

        config = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list=DEVICE_ID, allow_growth=allow_growth))
        set_session(tf.Session(config=config))

    else:
        config = tf.ConfigProto(
            device_count={"GPU": 0},
            log_device_placement=True
        )

        set_session(tf.Session(config=config))


class FCN(object):
    def __init__(self, input_channel_count, output_channel_count, LENGTH, SCALE, weights_path, include_softmax=True):
        self.model_weight = weights_path

        inputs = Input((LENGTH, LENGTH, input_channel_count))
        conv1 = Conv2D(output_channel_count, 2, padding='same')(inputs)
        self.model = Model(inputs=inputs, outputs=conv1)

    def get_model(self, loss=None, opt=None, gpu_num=0, load_weight=False):
        if load_weight:
            self.model.load_weights(self.model_weight)

        if gpu_num >= 2:
            with tf.device("/cpu:0"):
                self.model_template = self.model
        else:
            self.model_template = self.model

        if gpu_num >= 2:
            self.model = multi_gpu_model(self.model_template, gpus=gpu_num)

        if loss is not None and opt is not None:
            self.model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
        return self.model

    def get_batch_size(self, size):
        return int(1310720 / (size[0] * size[1]))

    def plot_model(self, output_path="/dropbox/", network_mode="model"):
        plot_model(self.model, to_file=output_path + network_mode + ".png", show_shapes=True, show_layer_names=False)
        self.model.summary()

    def get_callbacks(self):
        monitor = "val_loss"
        mode = "min"
        patience = 5
        early_stopping = EarlyStopping(patience=patience, verbose=1, monitor=monitor, mode=mode)
        checkpoint = AnotherModelCheckpointCallback(self.model_weight, base_model=self.model_template, monitor=monitor, verbose=1, save_best_only=True, save_weights_only=True, mode=mode)

        return [early_stopping, checkpoint]

    def load_train_data(self, generator1, generator2, dir1, dir2, batch_size, img_height, img_width, num_classes, shuffle=True):
        genX1 = generator1.flow_from_directory(dir1 + "0/",
                                               target_size=(img_height, img_width),
                                               class_mode=None, batch_size=batch_size,
                                               shuffle=shuffle, color_mode="rgb", seed=1)

        genX2 = generator2.flow_from_directory(dir2 + "0/",
                                               target_size=(img_height, img_width),
                                               class_mode=None, batch_size=batch_size,
                                               shuffle=shuffle, color_mode="grayscale", seed=1)

        while True:
            X1i = genX1.next()
            X2i = genX2.next()
            yield X1i, extended_to_categorical(X2i, num_classes)

    def load_test_data(self, generator1, dir1, batch_size, img_height, img_width, num):
        genX1 = generator1.flow_from_directory(dir1 + "0/" + str(num) + "/",
                                               target_size=(img_height, img_width),
                                               class_mode=None, batch_size=batch_size,
                                               shuffle=False, color_mode="rgb", seed=1)
        while True:
            X1i = genX1.next()
            yield X1i

    def train_model(self, train_generator, train_image_list, batch_size, G, valid_generator, valid_image_list, NUM_EPOCH, callbacks):
        history = self.model.fit_generator(train_generator, steps_per_epoch=len(train_image_list) // (batch_size * G),
                                           validation_data=valid_generator, validation_steps=len(valid_image_list) // (batch_size * G),
                                           epochs=NUM_EPOCH, verbose=2, callbacks=callbacks, max_queue_size=10)
        return history

    def predict_model(self, test_generator, test_image_list, steps, verbose=2, workers=1):
        y_pred = self.model.predict_generator(test_generator, steps=steps, verbose=verbose, workers=workers)
        return y_pred


class Deeplabv3plus(FCN):
    def __init__(self, input_channel_count, output_channel_count, LENGTH, SCALE, weights_path, include_softmax=True):
        self.model_weight = weights_path
        self.nb_classes = output_channel_count

        self.CONCATENATE_AXIS = -1
        self.CONV_FILTER_SIZE = 3
        self.first_layer_filter_count = 32

        base_model = Deeplabv3(input_shape=(LENGTH, LENGTH, input_channel_count), classes=output_channel_count, weights='pascal_voc', input_tensor=None, backbone='xception', OS=16, alpha=1.)

        # cropping & upsampling
        if SCALE == 0:
            pass
        elif SCALE == 1:
            x = Cropping2D(cropping=LENGTH // 4)(base_model.output)
            x = UpSampling2D(size=(2, 2))(x)
        elif SCALE == 2:
            x = Cropping2D(cropping=3 * LENGTH // 8)(base_model.output)
            x = UpSampling2D(size=(4, 4))(x)
        elif SCALE == 3:
            x = Cropping2D(cropping=7 * LENGTH // 16)(base_model.output)
            x = UpSampling2D(size=(8, 8))(x)
        elif SCALE == 4:
            x = Cropping2D(cropping=15 * LENGTH // 32)(base_model.output)
            x = UpSampling2D(size=(16, 16))(x)
        elif SCALE == 5:
            x = Cropping2D(cropping=31 * LENGTH // 64)(base_model.output)
            x = UpSampling2D(size=(32, 32))(x)
        elif SCALE == 6:
            x = Cropping2D(cropping=63 * LENGTH // 128)(base_model.output)
            x = UpSampling2D(size=(64, 64))(x)
        else:
            sys.exit()

        if include_softmax:
            if SCALE == 0:
                x = Activation("softmax")(base_model.output)
            else:
                x = Activation("softmax")(x)

        self.model = Model(inputs=base_model.input, outputs=x)

    def get_batch_size(self, size):
        return int(524288 / (size[0] * size[1]))


class Segnet(FCN):
    def __init__(self, input_channel_count, output_channel_count, LENGTH, SCALE, weights_path, include_softmax=True):
        self.model_weight = weights_path
        self.nb_classes = output_channel_count

        self.CONCATENATE_AXIS = -1
        self.CONV_FILTER_SIZE = 3
        self.first_layer_filter_count = 32

        kernel = 3
        pool_size = (2, 2)
        n_labels = output_channel_count

        inputs = Input((LENGTH, LENGTH, input_channel_count))

        conv_1 = Convolution2D(64, (kernel, kernel), padding="same")(inputs)
        conv_1 = BatchNormalization()(conv_1)
        conv_1 = Activation("relu")(conv_1)
        conv_2 = Convolution2D(64, (kernel, kernel), padding="same")(conv_1)
        conv_2 = BatchNormalization()(conv_2)
        conv_2 = Activation("relu")(conv_2)

        pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

        conv_3 = Convolution2D(128, (kernel, kernel), padding="same")(pool_1)
        conv_3 = BatchNormalization()(conv_3)
        conv_3 = Activation("relu")(conv_3)
        conv_4 = Convolution2D(128, (kernel, kernel), padding="same")(conv_3)
        conv_4 = BatchNormalization()(conv_4)
        conv_4 = Activation("relu")(conv_4)

        pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

        conv_5 = Convolution2D(256, (kernel, kernel), padding="same")(pool_2)
        conv_5 = BatchNormalization()(conv_5)
        conv_5 = Activation("relu")(conv_5)
        conv_6 = Convolution2D(256, (kernel, kernel), padding="same")(conv_5)
        conv_6 = BatchNormalization()(conv_6)
        conv_6 = Activation("relu")(conv_6)
        conv_7 = Convolution2D(256, (kernel, kernel), padding="same")(conv_6)
        conv_7 = BatchNormalization()(conv_7)
        conv_7 = Activation("relu")(conv_7)

        pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

        conv_8 = Convolution2D(512, (kernel, kernel), padding="same")(pool_3)
        conv_8 = BatchNormalization()(conv_8)
        conv_8 = Activation("relu")(conv_8)
        conv_9 = Convolution2D(512, (kernel, kernel), padding="same")(conv_8)
        conv_9 = BatchNormalization()(conv_9)
        conv_9 = Activation("relu")(conv_9)
        conv_10 = Convolution2D(512, (kernel, kernel), padding="same")(conv_9)
        conv_10 = BatchNormalization()(conv_10)
        conv_10 = Activation("relu")(conv_10)

        pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)

        conv_11 = Convolution2D(512, (kernel, kernel), padding="same")(pool_4)
        conv_11 = BatchNormalization()(conv_11)
        conv_11 = Activation("relu")(conv_11)
        conv_12 = Convolution2D(512, (kernel, kernel), padding="same")(conv_11)
        conv_12 = BatchNormalization()(conv_12)
        conv_12 = Activation("relu")(conv_12)
        conv_13 = Convolution2D(512, (kernel, kernel), padding="same")(conv_12)
        conv_13 = BatchNormalization()(conv_13)
        conv_13 = Activation("relu")(conv_13)

        pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)

        # decoder

        unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])

        conv_14 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_1)
        conv_14 = BatchNormalization()(conv_14)
        conv_14 = Activation("relu")(conv_14)
        conv_15 = Convolution2D(512, (kernel, kernel), padding="same")(conv_14)
        conv_15 = BatchNormalization()(conv_15)
        conv_15 = Activation("relu")(conv_15)
        conv_16 = Convolution2D(512, (kernel, kernel), padding="same")(conv_15)
        conv_16 = BatchNormalization()(conv_16)
        conv_16 = Activation("relu")(conv_16)

        unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])

        conv_17 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_2)
        conv_17 = BatchNormalization()(conv_17)
        conv_17 = Activation("relu")(conv_17)
        conv_18 = Convolution2D(512, (kernel, kernel), padding="same")(conv_17)
        conv_18 = BatchNormalization()(conv_18)
        conv_18 = Activation("relu")(conv_18)
        conv_19 = Convolution2D(256, (kernel, kernel), padding="same")(conv_18)
        conv_19 = BatchNormalization()(conv_19)
        conv_19 = Activation("relu")(conv_19)

        unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])

        conv_20 = Convolution2D(256, (kernel, kernel), padding="same")(unpool_3)
        conv_20 = BatchNormalization()(conv_20)
        conv_20 = Activation("relu")(conv_20)
        conv_21 = Convolution2D(256, (kernel, kernel), padding="same")(conv_20)
        conv_21 = BatchNormalization()(conv_21)
        conv_21 = Activation("relu")(conv_21)
        conv_22 = Convolution2D(128, (kernel, kernel), padding="same")(conv_21)
        conv_22 = BatchNormalization()(conv_22)
        conv_22 = Activation("relu")(conv_22)

        unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])

        conv_23 = Convolution2D(128, (kernel, kernel), padding="same")(unpool_4)
        conv_23 = BatchNormalization()(conv_23)
        conv_23 = Activation("relu")(conv_23)
        conv_24 = Convolution2D(64, (kernel, kernel), padding="same")(conv_23)
        conv_24 = BatchNormalization()(conv_24)
        conv_24 = Activation("relu")(conv_24)

        unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])

        conv_25 = Convolution2D(64, (kernel, kernel), padding="same")(unpool_5)
        conv_25 = BatchNormalization()(conv_25)
        conv_25 = Activation("relu")(conv_25)

        conv_26 = Convolution2D(n_labels, (1, 1), padding="valid")(conv_25)
        conv_26 = BatchNormalization()(conv_26)

        # cropping & upsampling
        if SCALE == 0:
            pass
        elif SCALE == 1:
            conv_26 = Cropping2D(cropping=LENGTH // 4)(conv_26)
            conv_26 = UpSampling2D(size=(2, 2))(conv_26)
        elif SCALE == 2:
            conv_26 = Cropping2D(cropping=3 * LENGTH // 8)(conv_26)
            conv_26 = UpSampling2D(size=(4, 4))(conv_26)
        elif SCALE == 3:
            conv_26 = Cropping2D(cropping=7 * LENGTH // 16)(conv_26)
            conv_26 = UpSampling2D(size=(8, 8))(conv_26)
        elif SCALE == 4:
            conv_26 = Cropping2D(cropping=15 * LENGTH // 32)(conv_26)
            conv_26 = UpSampling2D(size=(16, 16))(conv_26)
        elif SCALE == 5:
            conv_26 = Cropping2D(cropping=31 * LENGTH // 64)(conv_26)
            conv_26 = UpSampling2D(size=(32, 32))(conv_26)
        elif SCALE == 6:
            conv_26 = Cropping2D(cropping=63 * LENGTH // 128)(conv_26)
            conv_26 = UpSampling2D(size=(64, 64))(conv_26)
        else:
            sys.exit()

        if include_softmax:
            conv_26 = Activation("softmax")(conv_26)

        self.model = Model(inputs=inputs, outputs=conv_26, name="SegNet")

    def get_batch_size(self, size):
        return int(1000000 / (size[0] * size[1]))


class Unet(FCN):
    def __init__(self, input_channel_count, output_channel_count, LENGTH, SCALE, weights_path, include_softmax=True):
        self.model_weight = weights_path
        self.nb_classes = output_channel_count

        self.CONCATENATE_AXIS = -1
        self.CONV_FILTER_SIZE = 3
        self.first_layer_filter_count = 32

        inputs = Input((LENGTH, LENGTH, input_channel_count))

        conv1 = Conv2D(self.first_layer_filter_count, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = keras.layers.advanced_activations.ELU()(conv1)
        conv1 = Conv2D(self.first_layer_filter_count, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = keras.layers.advanced_activations.ELU()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(self.first_layer_filter_count * 2, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = keras.layers.advanced_activations.ELU()(conv2)
        conv2 = Conv2D(self.first_layer_filter_count * 2, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = keras.layers.advanced_activations.ELU()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(self.first_layer_filter_count * 4, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = keras.layers.advanced_activations.ELU()(conv3)
        conv3 = Conv2D(self.first_layer_filter_count * 4, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = keras.layers.advanced_activations.ELU()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(self.first_layer_filter_count * 8, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = keras.layers.advanced_activations.ELU()(conv4)
        conv4 = Conv2D(self.first_layer_filter_count * 8, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = keras.layers.advanced_activations.ELU()(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(self.first_layer_filter_count * 16, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(pool4)
        conv5 = BatchNormalization()(conv5)
        conv5 = keras.layers.advanced_activations.ELU()(conv5)
        conv5 = Conv2D(self.first_layer_filter_count * 16, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = keras.layers.advanced_activations.ELU()(conv5)

        up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=self.CONCATENATE_AXIS)
        conv6 = Conv2D(self.first_layer_filter_count * 8, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(up6)
        conv6 = BatchNormalization()(conv6)
        conv6 = keras.layers.advanced_activations.ELU()(conv6)
        conv6 = Conv2D(self.first_layer_filter_count * 8, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(conv6)
        conv6 = BatchNormalization()(conv6)
        conv6 = keras.layers.advanced_activations.ELU()(conv6)

        up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=self.CONCATENATE_AXIS)
        conv7 = Conv2D(self.first_layer_filter_count * 4, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(up7)
        conv7 = BatchNormalization()(conv7)
        conv7 = keras.layers.advanced_activations.ELU()(conv7)
        conv7 = Conv2D(self.first_layer_filter_count * 4, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(conv7)
        conv7 = BatchNormalization()(conv7)
        conv7 = keras.layers.advanced_activations.ELU()(conv7)

        up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=self.CONCATENATE_AXIS)
        conv8 = Conv2D(self.first_layer_filter_count * 2, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(up8)
        conv8 = BatchNormalization()(conv8)
        conv8 = keras.layers.advanced_activations.ELU()(conv8)
        conv8 = Conv2D(self.first_layer_filter_count * 2, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(conv8)
        conv8 = BatchNormalization()(conv8)
        conv8 = keras.layers.advanced_activations.ELU()(conv8)

        up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=self.CONCATENATE_AXIS)
        conv9 = Conv2D(self.first_layer_filter_count, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(up9)
        conv9 = BatchNormalization()(conv9)
        conv9 = keras.layers.advanced_activations.ELU()(conv9)
        conv9 = Conv2D(self.first_layer_filter_count, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(conv9)
        # crop9 = Cropping2D(cropping=((16, 16), (16, 16)))(conv9)
        conv9 = BatchNormalization()(conv9)
        conv9 = keras.layers.advanced_activations.ELU()(conv9)
        conv9 = Conv2D(output_channel_count, 1, padding='same', kernel_initializer='he_uniform')(conv9)

        # cropping & upsampling
        if SCALE == 0:
            pass
        elif SCALE == 1:
            conv9 = Cropping2D(cropping=LENGTH // 4)(conv9)
            conv9 = UpSampling2D(size=(2, 2))(conv9)
        elif SCALE == 2:
            conv9 = Cropping2D(cropping=3 * LENGTH // 8)(conv9)
            conv9 = UpSampling2D(size=(4, 4))(conv9)
        elif SCALE == 3:
            conv9 = Cropping2D(cropping=7 * LENGTH // 16)(conv9)
            conv9 = UpSampling2D(size=(8, 8))(conv9)
        elif SCALE == 4:
            conv9 = Cropping2D(cropping=15 * LENGTH // 32)(conv9)
            conv9 = UpSampling2D(size=(16, 16))(conv9)
        elif SCALE == 5:
            conv9 = Cropping2D(cropping=31 * LENGTH // 64)(conv9)
            conv9 = UpSampling2D(size=(32, 32))(conv9)
        elif SCALE == 6:
            conv9 = Cropping2D(cropping=63 * LENGTH // 128)(conv9)
            conv9 = UpSampling2D(size=(64, 64))(conv9)
        else:
            sys.exit()

        if include_softmax:
            conv9 = Activation(activation='softmax')(conv9)
        else:
            conv9 = keras.layers.advanced_activations.ELU()(conv9)

        self.model = Model(inputs=inputs, outputs=conv9)


class UnetDilatedrate2(FCN):
    def __init__(self, input_channel_count, output_channel_count, LENGTH, SCALE, weights_path, include_softmax=True):
        self.model_weight = weights_path
        self.nb_classes = output_channel_count

        self.CONCATENATE_AXIS = -1
        self.CONV_FILTER_SIZE = 3
        self.first_layer_filter_count = 32

        inputs = Input((LENGTH, LENGTH, input_channel_count))

        conv1 = Conv2D(self.first_layer_filter_count, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = keras.layers.advanced_activations.ELU()(conv1)
        conv1 = Conv2D(self.first_layer_filter_count, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = keras.layers.advanced_activations.ELU()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(self.first_layer_filter_count * 2, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = keras.layers.advanced_activations.ELU()(conv2)
        conv2 = Conv2D(self.first_layer_filter_count * 2, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = keras.layers.advanced_activations.ELU()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(self.first_layer_filter_count * 4, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = keras.layers.advanced_activations.ELU()(conv3)
        conv3 = Conv2D(self.first_layer_filter_count * 4, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = keras.layers.advanced_activations.ELU()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(self.first_layer_filter_count * 8, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = keras.layers.advanced_activations.ELU()(conv4)
        conv4 = Conv2D(self.first_layer_filter_count * 8, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = keras.layers.advanced_activations.ELU()(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(self.first_layer_filter_count * 16, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform', dilation_rate=2)(pool4)
        conv5 = BatchNormalization()(conv5)
        conv5 = keras.layers.advanced_activations.ELU()(conv5)
        conv5 = Conv2D(self.first_layer_filter_count * 16, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform', dilation_rate=2)(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = keras.layers.advanced_activations.ELU()(conv5)

        up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=self.CONCATENATE_AXIS)
        conv6 = Conv2D(self.first_layer_filter_count * 8, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(up6)
        conv6 = BatchNormalization()(conv6)
        conv6 = keras.layers.advanced_activations.ELU()(conv6)
        conv6 = Conv2D(self.first_layer_filter_count * 8, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(conv6)
        conv6 = BatchNormalization()(conv6)
        conv6 = keras.layers.advanced_activations.ELU()(conv6)

        up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=self.CONCATENATE_AXIS)
        conv7 = Conv2D(self.first_layer_filter_count * 4, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(up7)
        conv7 = BatchNormalization()(conv7)
        conv7 = keras.layers.advanced_activations.ELU()(conv7)
        conv7 = Conv2D(self.first_layer_filter_count * 4, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(conv7)
        conv7 = BatchNormalization()(conv7)
        conv7 = keras.layers.advanced_activations.ELU()(conv7)

        up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=self.CONCATENATE_AXIS)
        conv8 = Conv2D(self.first_layer_filter_count * 2, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(up8)
        conv8 = BatchNormalization()(conv8)
        conv8 = keras.layers.advanced_activations.ELU()(conv8)
        conv8 = Conv2D(self.first_layer_filter_count * 2, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(conv8)
        conv8 = BatchNormalization()(conv8)
        conv8 = keras.layers.advanced_activations.ELU()(conv8)

        up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=self.CONCATENATE_AXIS)
        conv9 = Conv2D(self.first_layer_filter_count, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(up9)
        conv9 = BatchNormalization()(conv9)
        conv9 = keras.layers.advanced_activations.ELU()(conv9)
        conv9 = Conv2D(self.first_layer_filter_count, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(conv9)
        conv9 = BatchNormalization()(conv9)
        conv9 = keras.layers.advanced_activations.ELU()(conv9)
        conv9 = Conv2D(output_channel_count, 1, padding='same', kernel_initializer='he_uniform')(conv9)

        # cropping & upsampling
        if SCALE == 0:
            pass
        elif SCALE == 1:
            conv9 = Cropping2D(cropping=LENGTH // 4)(conv9)
            conv9 = UpSampling2D(size=(2, 2))(conv9)
        elif SCALE == 2:
            conv9 = Cropping2D(cropping=3 * LENGTH // 8)(conv9)
            conv9 = UpSampling2D(size=(4, 4))(conv9)
        elif SCALE == 3:
            conv9 = Cropping2D(cropping=7 * LENGTH // 16)(conv9)
            conv9 = UpSampling2D(size=(8, 8))(conv9)
        elif SCALE == 4:
            conv9 = Cropping2D(cropping=15 * LENGTH // 32)(conv9)
            conv9 = UpSampling2D(size=(16, 16))(conv9)
        elif SCALE == 5:
            conv9 = Cropping2D(cropping=31 * LENGTH // 64)(conv9)
            conv9 = UpSampling2D(size=(32, 32))(conv9)
        elif SCALE == 6:
            conv9 = Cropping2D(cropping=63 * LENGTH // 128)(conv9)
            conv9 = UpSampling2D(size=(64, 64))(conv9)
        else:
            sys.exit()

        if include_softmax:
            conv9 = Activation(activation='softmax')(conv9)
        else:
            conv9 = keras.layers.advanced_activations.ELU()(conv9)

        self.model = Model(inputs=inputs, outputs=conv9)


class FixedWeightingMultiFieldOfViewCNN(FCN):
    def __init__(self, input_channel_count, output_channel_count, LENGTH, LEVEL, weights_path, include_softmax=True):
        self.model_weight = weights_path
        self.nb_classes = output_channel_count
        self.LEVEL = LEVEL

        self.CONCATENATE_AXIS = -1
        self.CONV_FILTER_SIZE = 3
        self.first_layer_filter_count = 32

        network0 = Unet(input_channel_count, output_channel_count, LENGTH, SCALE=0, weights_path=None, include_softmax=include_softmax)
        network1 = Unet(input_channel_count, output_channel_count, LENGTH, SCALE=LEVEL[1] - LEVEL[0], weights_path=None, include_softmax=include_softmax)
        network2 = Unet(input_channel_count, output_channel_count, LENGTH, SCALE=LEVEL[2] - LEVEL[0], weights_path=None, include_softmax=include_softmax)
        self.f0_x = network0.get_model()
        self.f1_x = network1.get_model()
        self.f2_x = network2.get_model()

        if weights_path != "plot":
            print("Load weights", weights_path.replace(str(LEVEL), "[%s]" % LEVEL[0], 1).replace(self.__class__.__name__, "Unet"))
            print("Load weights", weights_path.replace(str(LEVEL), "[%s]" % LEVEL[1], 1).replace(self.__class__.__name__, "Unet"))
            print("Load weights", weights_path.replace(str(LEVEL), "[%s]" % LEVEL[2], 1).replace(self.__class__.__name__, "Unet"))
            self.f0_x.load_weights(weights_path.replace(str(LEVEL), "[%s]" % LEVEL[0], 1).replace(self.__class__.__name__, "Unet"))
            self.f1_x.load_weights(weights_path.replace(str(LEVEL), "[%s]" % LEVEL[1], 1).replace(self.__class__.__name__, "Unet"))
            self.f2_x.load_weights(weights_path.replace(str(LEVEL), "[%s]" % LEVEL[2], 1).replace(self.__class__.__name__, "Unet"))

        x = concatenate([self.f0_x.output, self.f1_x.output, self.f2_x.output], axis=self.CONCATENATE_AXIS)
        x = Conv2D(self.first_layer_filter_count * 2, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = keras.layers.advanced_activations.ELU()(x)
        x = Conv2D(self.first_layer_filter_count * 2, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = keras.layers.advanced_activations.ELU()(x)
        x = Conv2D(self.first_layer_filter_count, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = keras.layers.advanced_activations.ELU()(x)
        x = Conv2D(self.first_layer_filter_count, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = keras.layers.advanced_activations.ELU()(x)
        x = Conv2D(output_channel_count, 1, padding='same', kernel_initializer='he_uniform')(x)
        x = Activation(activation='softmax', name="last")(x)
        self.model = Model(inputs=[self.f0_x.input, self.f1_x.input, self.f2_x.input], outputs=[self.f0_x.output, self.f1_x.output, self.f2_x.output, x])

    def get_callbacks(self):
        monitor = "val_last_loss"
        mode = "min"
        patience = 5
        early_stopping = EarlyStopping(patience=patience, verbose=1, monitor=monitor, mode=mode)
        checkpoint = ModelCheckpoint(self.model_weight, monitor=monitor, verbose=1, save_best_only=True, save_weights_only=True, mode=mode)
        cp_Expert0 = AnotherModelCheckpointCallback(self.model_weight.replace(str(self.LEVEL), "[" + str(self.LEVEL[0]) + "]").replace(self.__class__.__name__, self.__class__.__name__ + "_Expert" + str(self.LEVEL)), self.f0_x, monitor=monitor, verbose=1, save_best_only=True, save_weights_only=True, mode=mode)
        cp_Expert1 = AnotherModelCheckpointCallback(self.model_weight.replace(str(self.LEVEL), "[" + str(self.LEVEL[1]) + "]").replace(self.__class__.__name__, self.__class__.__name__ + "_Expert" + str(self.LEVEL)), self.f1_x, monitor=monitor, verbose=1, save_best_only=True, save_weights_only=True, mode=mode)
        cp_Expert2 = AnotherModelCheckpointCallback(self.model_weight.replace(str(self.LEVEL), "[" + str(self.LEVEL[2]) + "]").replace(self.__class__.__name__, self.__class__.__name__ + "_Expert" + str(self.LEVEL)), self.f2_x, monitor=monitor, verbose=1, save_best_only=True, save_weights_only=True, mode=mode)

        return [early_stopping, checkpoint, cp_Expert0, cp_Expert1, cp_Expert2]

    def get_batch_size(self, size):
        return int(524288 / (size[0] * size[1]))

    def load_train_data(self, generator1, generator2, dir1, dir2, batch_size, img_height, img_width, num_classes, shuffle=True):
        genX1_1 = generator1.flow_from_directory(dir1 + "0/",
                                                 target_size=(img_height, img_width),
                                                 class_mode=None, batch_size=batch_size,
                                                 shuffle=shuffle, color_mode="rgb", seed=1)
        genX1_2 = generator1.flow_from_directory(dir1 + "1/",
                                                 target_size=(img_height, img_width),
                                                 class_mode=None, batch_size=batch_size,
                                                 shuffle=shuffle, color_mode="rgb", seed=1)
        genX1_3 = generator1.flow_from_directory(dir1 + "2/",
                                                 target_size=(img_height, img_width),
                                                 class_mode=None, batch_size=batch_size,
                                                 shuffle=shuffle, color_mode="rgb", seed=1)

        genX2 = generator2.flow_from_directory(dir2 + "0/",
                                               target_size=(img_height, img_width),
                                               class_mode=None, batch_size=batch_size,
                                               shuffle=shuffle, color_mode="grayscale", seed=1)
        while True:
            X1_1i = genX1_1.next()
            X1_2i = genX1_2.next()
            X1_3i = genX1_3.next()
            X2i = extended_to_categorical(genX2.next(), num_classes)
            yield [X1_1i, X1_2i, X1_3i], [X2i, X2i, X2i, X2i]

    def load_test_data(self, generator1, dir1, batch_size, img_height, img_width, num):
        genX1_1 = generator1.flow_from_directory(dir1 + "0/" + str(num) + "/",
                                                 target_size=(img_height, img_width),
                                                 class_mode=None, batch_size=batch_size,
                                                 shuffle=False, color_mode="rgb", seed=1)
        genX1_2 = generator1.flow_from_directory(dir1 + "1/" + str(num) + "/",
                                                 target_size=(img_height, img_width),
                                                 class_mode=None, batch_size=batch_size,
                                                 shuffle=False, color_mode="rgb", seed=1)
        genX1_3 = generator1.flow_from_directory(dir1 + "2/" + str(num) + "/",
                                                 target_size=(img_height, img_width),
                                                 class_mode=None, batch_size=batch_size,
                                                 shuffle=False, color_mode="rgb", seed=1)
        while True:
            X1_1i = genX1_1.next()
            X1_2i = genX1_2.next()
            X1_3i = genX1_3.next()
            yield [X1_1i, X1_2i, X1_3i]


class AdaptiveWeightingMultiFieldOfViewCNN(FCN):
    def __init__(self, input_channel_count, output_channel_count, LENGTH, LEVEL, weights_path, include_softmax=True):
        self.model_weight = weights_path
        self.nb_classes = output_channel_count
        self.LEVEL = LEVEL

        self.CONCATENATE_AXIS = -1
        self.CONV_FILTER_SIZE = 3
        self.first_layer_filter_count = 32

        #  Expert CNNs
        network0 = Unet(input_channel_count, output_channel_count, LENGTH, SCALE=0, weights_path=None, include_softmax=include_softmax)
        network1 = Unet(input_channel_count, output_channel_count, LENGTH, SCALE=LEVEL[1] - LEVEL[0], weights_path=None, include_softmax=include_softmax)
        network2 = Unet(input_channel_count, output_channel_count, LENGTH, SCALE=LEVEL[2] - LEVEL[0], weights_path=None, include_softmax=include_softmax)
        self.f0_x = network0.get_model()
        self.f1_x = network1.get_model()
        self.f2_x = network2.get_model()

        self.f012_x = Model(inputs=[self.f0_x.input, self.f1_x.input, self.f2_x.input], outputs=[self.f0_x.output, self.f1_x.output, self.f2_x.output])

        if weights_path != "plot":
            self.f0_x.load_weights(weights_path.replace(str(LEVEL), "[%s]" % LEVEL[0], 1).replace(self.__class__.__name__, "Unet"))
            self.f1_x.load_weights(weights_path.replace(str(LEVEL), "[%s]" % LEVEL[1], 1).replace(self.__class__.__name__, "Unet"))
            self.f2_x.load_weights(weights_path.replace(str(LEVEL), "[%s]" % LEVEL[2], 1).replace(self.__class__.__name__, "Unet"))

        x = concatenate([self.f0_x.output, self.f1_x.output, self.f2_x.output], axis=self.CONCATENATE_AXIS)
        multi_input_fcns = Model(inputs=[self.f0_x.input, self.f1_x.input, self.f2_x.input], outputs=[x])

        #  Weighting CNN
        ssc = Xception(include_top=False, weights="imagenet", input_tensor=self.f1_x.input, input_shape=(LENGTH, LENGTH, input_channel_count), pooling="avg")
        top_model = Dense(len(LEVEL), activation='sigmoid')(ssc.output)
        self.soft_switch_cnn = Model(inputs=ssc.input, outputs=top_model)

        # Aggregating CNN
        def resize_scalar_to_tensor(nb_classes):
            def custom_layer(x):
                x = K.expand_dims(x, axis=1)
                x = K.expand_dims(x, axis=1)
                x = K.resize_volumes(x, depth_factor=LENGTH, height_factor=LENGTH, width_factor=1, data_format='channels_last')  # depth,height,widthの順番がtensorflow版では間違ってる
                x1 = x[:, :, :, 0:1]
                x2 = x[:, :, :, 1:2]
                x3 = x[:, :, :, 2:3]
                x1_con = x1
                x2_con = x2
                x3_con = x3
                for i in range(nb_classes - 1):
                    x1_con = concatenate([x1_con, x1], axis=-1)
                    x2_con = concatenate([x2_con, x2], axis=-1)
                    x3_con = concatenate([x3_con, x3], axis=-1)
                x = concatenate([x1_con, x2_con, x3_con], axis=-1)
                return x
            return custom_layer
        resize_layer = Lambda(resize_scalar_to_tensor(output_channel_count))
        multiplied_model = Model(inputs=[self.f0_x.input, self.f1_x.input, self.f2_x.input], outputs=multiply([multi_input_fcns.output, resize_layer(self.soft_switch_cnn.output)]))
        output_fcn = Conv2D(self.first_layer_filter_count * 2, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(multiplied_model.output)
        output_fcn = BatchNormalization()(output_fcn)
        output_fcn = keras.layers.advanced_activations.ELU()(output_fcn)
        output_fcn = Conv2D(self.first_layer_filter_count * 2, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(output_fcn)
        output_fcn = BatchNormalization()(output_fcn)
        output_fcn = keras.layers.advanced_activations.ELU()(output_fcn)
        output_fcn = Conv2D(self.first_layer_filter_count, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(output_fcn)
        output_fcn = BatchNormalization()(output_fcn)
        output_fcn = keras.layers.advanced_activations.ELU()(output_fcn)
        output_fcn = Conv2D(self.first_layer_filter_count, self.CONV_FILTER_SIZE, padding='same', kernel_initializer='he_uniform')(output_fcn)
        output_fcn = BatchNormalization()(output_fcn)
        output_fcn = keras.layers.advanced_activations.ELU()(output_fcn)
        output_fcn = Conv2D(output_channel_count, 1, padding='same', kernel_initializer='he_uniform')(output_fcn)
        output_fcn = Activation(activation='softmax', name="last")(output_fcn)
        
        self.model = Model(inputs=multiplied_model.input, outputs=[output_fcn, self.f0_x.output, self.f1_x.output, self.f2_x.output])

    def plot_model(self, output_path="/dropbox/", network_mode="model"):
        plot_model(self.soft_switch_cnn, to_file=output_path + network_mode + "_Switch" + ".png", show_shapes=True, show_layer_names=False)
        plot_model(self.model, to_file=output_path + network_mode + "_Whole" + ".png", show_shapes=True, show_layer_names=False)

    def get_model(self, loss=None, opt=None, gpu_num=0, load_weight=False):
        if load_weight:
            self.model.load_weights(self.model_weight)

        if gpu_num >= 2:
            with tf.device("/cpu:0"):
                self.model_template = self.model
        else:
            self.model_template = self.model

        if gpu_num >= 2:
            self.model = multi_gpu_model(self.model_template, gpus=gpu_num)

        if loss is not None and opt is not None:
            self.f012_x.compile(optimizer=opt, loss=loss, metrics=[dice_coef_op])
            self.soft_switch_cnn.compile(optimizer=opt, loss="mse", metrics=None)
            for layer in self.soft_switch_cnn.layers:
                layer.trainable = False
            self.model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])

    def get_batch_size(self, size):
        return int(524288 / (size[0] * size[1]))

    def get_callbacks(self):
        return None

    # For generating training data for Weighting CNN
    def load_train_data1(self, generator1, generator2, dir1, dir2, img_height, img_width, num_classes, shuffle=False):
        genX1_1 = generator1.flow_from_directory(dir1 + "0/",
                                                 target_size=(img_height, img_width),
                                                 class_mode=None, batch_size=1,
                                                 shuffle=shuffle, color_mode="rgb", seed=1)
        genX1_2 = generator1.flow_from_directory(dir1 + "1/",
                                                 target_size=(img_height, img_width),
                                                 class_mode=None, batch_size=1,
                                                 shuffle=shuffle, color_mode="rgb", seed=1)
        genX1_3 = generator1.flow_from_directory(dir1 + "2/",
                                                 target_size=(img_height, img_width),
                                                 class_mode=None, batch_size=1,
                                                 shuffle=shuffle, color_mode="rgb", seed=1)

        genX2 = generator2.flow_from_directory(dir2 + "0/",
                                               target_size=(img_height, img_width),
                                               class_mode=None, batch_size=1,
                                               shuffle=shuffle, color_mode="grayscale", seed=1)
        while True:
            X1_1i = genX1_1.next()
            X1_2i = genX1_2.next()
            X1_3i = genX1_3.next()
            X2i = extended_to_categorical(genX2.next(), num_classes)
            yield [X1_1i, X1_2i, X1_3i], [X2i, X2i, X2i]

    # For training Weighting CNN
    def load_train_data2(self, generator1, dir1, batch_size, img_height, img_width, shuffle=False):

        genX1_2 = generator1.flow_from_directory(dir1 + "1/",
                                                 target_size=(img_height, img_width),
                                                 class_mode=None, batch_size=batch_size,
                                                 shuffle=shuffle, color_mode="rgb", seed=1)

        while True:
            X1_2i = genX1_2.next()
            yield X1_2i

    # For training the integrated network
    def load_train_data3(self, generator1, generator2, dir1, dir2, batch_size, img_height, img_width, num_classes, shuffle=True):
        genX1_1 = generator1.flow_from_directory(dir1 + "0/",
                                                 target_size=(img_height, img_width),
                                                 class_mode=None, batch_size=batch_size,
                                                 shuffle=shuffle, color_mode="rgb", seed=1)
        genX1_2 = generator1.flow_from_directory(dir1 + "1/",
                                                 target_size=(img_height, img_width),
                                                 class_mode=None, batch_size=batch_size,
                                                 shuffle=shuffle, color_mode="rgb", seed=1)
        genX1_3 = generator1.flow_from_directory(dir1 + "2/",
                                                 target_size=(img_height, img_width),
                                                 class_mode=None, batch_size=batch_size,
                                                 shuffle=shuffle, color_mode="rgb", seed=1)

        genX2 = generator2.flow_from_directory(dir2 + "0/",
                                               target_size=(img_height, img_width),
                                               class_mode=None, batch_size=batch_size,
                                               shuffle=shuffle, color_mode="grayscale", seed=1)
        while True:
            X1_1i = genX1_1.next()
            X1_2i = genX1_2.next()
            X1_3i = genX1_3.next()
            X2i = extended_to_categorical(genX2.next(), num_classes)
            yield [X1_1i, X1_2i, X1_3i], [X2i, X2i, X2i, X2i]

    def load_train_data(self, generator1, generator2, dir1, dir2, batch_size, img_height, img_width, num_classes, shuffle=True):
        gen1 = self.load_train_data1(generator1, generator2, dir1, dir2, img_height, img_width, num_classes, shuffle=False)
        gen2 = self.load_train_data2(generator1, dir1, batch_size, img_height, img_width, shuffle=False)
        gen3 = self.load_train_data3(generator1, generator2, dir1, dir2, batch_size, img_height, img_width, num_classes, shuffle=True)
        return [gen1, gen2, gen3]

    def train_model(self, train_generator, train_image_list, batch_size, G, valid_generator, valid_image_list, num_epoch, callbacks):
        def iterate_weightset(y_iter):
            while True:
                yi = y_iter.next()
                yield yi

        class History(object):
            def __init__(self):
                self.history = {"last_loss": [], "last_acc": [], "val_last_loss": [], "val_last_acc": []}

        history = History()
        early_stop_count = 0
        patience = 5
        for epoch in range(num_epoch):
            softswitch_weight = np.zeros((len(valid_image_list), 3), np.float32)
            for num in range(len(valid_image_list)):
                evaluation = self.f012_x.evaluate_generator(valid_generator[0], steps=1, max_queue_size=1)
                softswitch_weight[num, 0] = evaluation[-3]
                softswitch_weight[num, 1] = evaluation[-2]
                softswitch_weight[num, 2] = evaluation[-1]
            reshaped_weight = np.reshape(softswitch_weight[0:(len(valid_image_list) // batch_size) * batch_size, :], (-1, batch_size, 3))
            n = 1  # n = 5 - epoch if epoch < 5 else 1
            for _ in range(n):
                weightset_iterator = iterate_weightset(iter(reshaped_weight))

                self.soft_switch_cnn.fit_generator(iter(zip(valid_generator[1], weightset_iterator)),
                                                   steps_per_epoch=len(valid_image_list) // (batch_size * G),
                                                   epochs=1, verbose=2)

            history_one_epoch = self.model.fit_generator(train_generator[2],
                                                         steps_per_epoch=len(train_image_list) // (batch_size * G),
                                                         validation_data=valid_generator[2],
                                                         validation_steps=len(valid_image_list) // (batch_size * G),
                                                         epochs=1, verbose=2, callbacks=callbacks)

            for key in history.history.keys():
                history.history[key].append(history_one_epoch.history[key][0])

            # check point
            if min(history.history["val_last_loss"]) == history_one_epoch.history["val_last_loss"][0]:
                self.model.save_weights(filepath=self.model_weight, overwrite=True)
                self.f0_x.save_weights(filepath=self.model_weight.replace(str(self.LEVEL), "[" + str(self.LEVEL[0]) + "]").replace(self.__class__.__name__, self.__class__.__name__ + "_Expert" + str(self.LEVEL)), overwrite=True)
                self.f1_x.save_weights(filepath=self.model_weight.replace(str(self.LEVEL), "[" + str(self.LEVEL[1]) + "]").replace(self.__class__.__name__, self.__class__.__name__ + "_Expert" + str(self.LEVEL)), overwrite=True)
                self.f2_x.save_weights(filepath=self.model_weight.replace(str(self.LEVEL), "[" + str(self.LEVEL[2]) + "]").replace(self.__class__.__name__, self.__class__.__name__ + "_Expert" + str(self.LEVEL)), overwrite=True)
                self.soft_switch_cnn.save_weights(filepath=self.model_weight.replace(self.__class__.__name__, self.__class__.__name__ + "_WeightingCNN"), overwrite=True)

                early_stop_count = 0
                print('\nEpoch %05d: val_last_loss improved to %0.5f' % (epoch + 1, history_one_epoch.history["val_last_loss"][0]))
            else:
                early_stop_count = early_stop_count + 1
                print('\nEpoch %05d: val_last_loss did not improve' % (epoch + 1))

            # early stopping
            if early_stop_count >= patience:
                break
        return history

    def load_test_data(self, generator1, dir1, batch_size, img_height, img_width, num):
        genX1_1 = generator1.flow_from_directory(dir1 + "0/" + str(num) + "/",
                                                 target_size=(img_height, img_width),
                                                 class_mode=None, batch_size=batch_size,
                                                 shuffle=False, color_mode="rgb", seed=1)
        genX1_2 = generator1.flow_from_directory(dir1 + "1/" + str(num) + "/",
                                                 target_size=(img_height, img_width),
                                                 class_mode=None, batch_size=batch_size,
                                                 shuffle=False, color_mode="rgb", seed=1)
        genX1_3 = generator1.flow_from_directory(dir1 + "2/" + str(num) + "/",
                                                 target_size=(img_height, img_width),
                                                 class_mode=None, batch_size=batch_size,
                                                 shuffle=False, color_mode="rgb", seed=1)
        while True:
            X1_1i = genX1_1.next()
            X1_2i = genX1_2.next()
            X1_3i = genX1_3.next()
            yield [X1_1i, X1_2i, X1_3i]


if __name__ == "__main__":
    set_session_gpu(G=0)

    unet = Unet(3, 5, 256, 0, weights_path=None)
    unet.plot_model("/dropbox/", "Unet")
