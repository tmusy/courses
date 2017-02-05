from keras.models import Sequential
from keras.layers import Dense, Activation


class VGG17(object):

    def __init__(self):
        """
        Creates the VGG16 model.
        """
        self._create_model()

    def _create_model(self):
        model = Sequential()
        model.add(Dense(32, input_shape=(784,)))

