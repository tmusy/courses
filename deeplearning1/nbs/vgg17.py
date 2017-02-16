import numpy as np
from keras.models import Sequential
from keras.layers import Lambda, ZeroPadding2D, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils.data_utils import get_file
from keras.optimizers import Adam
from keras.preprocessing import image
import h5py


vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1, 1, 3))


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

    def _conv_block(self, num_layers, num_filters):
        model = self.model
        for i in range(num_layers):
            model.add(ZeroPadding2D((1, 1)))
            model.add(Convolution2D(num_filters, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    def _create_model(self):
        model = self.model
        # input layer that normalize the input
        model.add(Lambda(vgg_preprocess, input_shape=(224, 224, 3)))

        # hidden layers
        self._conv_block(2, 64)
        self._conv_block(2, 128)
        self._conv_block(3, 256)
        self._conv_block(3, 512)
        self._conv_block(3, 512)

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))

        # output layer
        model.add(Dense(1028, activation='softmax'))

    def load_weights(self, source='http://www.platform.ai/models/vgg16.h5'):
        fname = source.split('/')[-1]
        weights = get_file(fname, source, cache_subdir='models')
        self.model.load_weights(weights)

    def finetune(self, n_classes):
        model = self.model
        # remove last layer
        model.pop()
        # fix weights of layers
        for layer in model.layers: layer.trainable = False
        # add new last layer
        model.add(Dense(n_classes, activation='softmax'))
        self.compile()

    def compile(self, lr=0.001):
        self.model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self, train_directory, val_directory, nb_epoch=1):
        train_batch_generator = self.get_batch_generator(train_directory)
        val_batch_generator = self.get_batch_generator(val_directory)
        self.model.fit_generator(generator=train_batch_generator,
                                 samples_per_epoch=train_batch_generator.nb_sample,
                                 nb_epoch=nb_epoch,
                                 validation_data=val_batch_generator,
                                 nb_val_samples=val_batch_generator.nb_sample)

    def get_batch_generator(self, directory, target_size=(224, 224), shuffle=True,
                            batch_size=32, class_mode='categorical'):
        gen = image.ImageDataGenerator()
        return gen.flow_from_directory(directory,
                                       target_size=target_size,
                                       shuffle=shuffle,
                                       batch_size=batch_size,
                                       class_mode=class_mode)


if __name__ == '__main__':
    path = "data/redux/"
    path = '/Users/musy/datasets/dogscats/sample/'
    vgg = VGG17()
    vgg.finetune(n_classes=2)
    # vgg.load_weights()
    # vgg.fit(train_directory=path+'train',
    #         val_directory=path+'valid',
    #         nb_epoch=1)
    model = vgg.model

    from os.path import expanduser
    home = expanduser("~")
    f = h5py.File(home+"/.keras/models/vgg16.h5", "r")
    f.keys()
