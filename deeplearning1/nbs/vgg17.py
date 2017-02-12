import numpy as np
from keras.models import Sequential
from keras.layers import Lambda, ZeroPadding2D, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils.data_utils import get_file
from keras.optimizers import Adam


vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3, 1, 1))


def vgg_preprocess(x):
    x = x - vgg_mean
    return x[:, ::-1]  # reverse axis rgb->bgr


class VGG17(object):

    def __init__(self):
        """
        Creates the VGG16 model.
        """
        self.model = Sequential()
        self._create_model()

    def conv_block(self, num_layers, num_filters):
        model = self.model
        for i in num_layers:
            model.add(ZeroPadding2D((1, 1)))
            model.add(Convolution2D(num_filters, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    def _create_model(self):
        model = self.model
        # input layer that normalize the input
        model.add(Lambda(vgg_preprocess, input_shape=(3,224,224)))

        # hidden layers
        self.conv_block(2, 64)
        self.conv_block(2, 128)
        self.conv_block(3, 256)
        self.conv_block(3, 512)
        self.conv_block(3, 512)

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))

        # output layer
        model.add(Dense(1028, activation='softmax'))

    def load_weights(self, fname='http://www.platform.ai/models/vgg16.h5'):
        self.model.load_weights(get_file(fname, self.FILE_PATH + fname, cache_subdir='models'))

    def finetune(self, batches):
        model = self.model
        # remove last layer
        model.pop()
        # fix weights of layers
        for layer in model.layers: layer.trainable = False
        # add new last layer
        model.add(Dense(batches.nb_class, activation='softmax'))
        self.compile()

    def compile(self, lr=0.001):
        self.model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self, batches, val_batches, nb_epoch=1):
        self.model.fit_generator(batches,
                                 samples_per_epoch=batches.nb_sample,
                                 nb_epoch=nb_epoch,
                                 validation_data=val_batches,
                                 nb_val_samples=val_batches.nb_sample)
