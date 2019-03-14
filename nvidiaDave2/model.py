from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout
from tensorflow.keras.layers import Activation, Flatten, Lambda, Input, ELU
from tensorflow.keras.optimizers import Adam


def create_model():
    input_shape = (66, 200, 3)
    model = Sequential()
    # Input normalization layer

    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=input_shape, name='lambda_norm'))

    # 5x5 Convolutional layers with stride of 2x2
    model.add(Convolution2D(24, 5, 5, padding="same", name='conv1'))
    model.add(ELU(name='elu1'))
    model.add(Convolution2D(36, 5, 5, padding="same", name='conv2'))
    model.add(ELU(name='elu2'))
    model.add(Convolution2D(48, 5, 5, padding="same", name='conv3'))
    model.add(ELU(name='elu3'))

    # 3x3 Convolutional layers with stride of 1x1
    model.add(Convolution2D(64, 3, 3, padding="same", name='conv4'))
    model.add(ELU(name='elu4'))
    model.add(Convolution2D(64, 3, 3, padding="same", name='conv5'))
    model.add(ELU(name='elu5'))

    # Flatten before passing to the fully connected layers
    model.add(Flatten())
    # Three fully connected layers
    model.add(Dense(100, name='fc1'))
    model.add(Dropout(.5, name='do1'))
    model.add(ELU(name='elu6'))
    model.add(Dense(50, name='fc2'))
    model.add(Dropout(.5, name='do2'))
    model.add(ELU(name='elu7'))
    model.add(Dense(10, name='fc3'))
    model.add(Dropout(.5, name='do3'))
    model.add(ELU(name='elu8'))

    # Output layer with tanh activation
    model.add(Dense(1, activation='tanh', name='output'))
    return model
