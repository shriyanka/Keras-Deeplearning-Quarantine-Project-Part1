import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *


class Model:
    def __init__(self, input_dim, classes, lr=0.001, epochs=1):
        self.input_dim = input_dim
        self.num_classes = classes
        self.learning_rate = lr
        self.epochs = epochs

    def lr_schedule(self, epoch):
        if epoch % 15 == 0:
            self.learning_rate = self.learning_rate/10
        print('Learning rate: ', self.learning_rate)
        return self.learning_rate

    def model(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal', activation='relu', input_shape=self.input_dim))
        model.add(Conv2D(64, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal', activation='relu'))
        model.add(Conv2D(128, kernel_size=(5, 5), padding='same', kernel_initializer='he_normal', activation='relu'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(AveragePooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.5))

        model.add(Conv2D(512, kernel_size=(5, 5), padding='same', kernel_initializer='he_normal', activation='relu'))
        model.add(Conv2D(512, kernel_size=(5, 5), padding='same', kernel_initializer='he_normal', activation='relu'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(128, kernel_size=(5, 5), padding='same', kernel_initializer='he_normal', activation='relu'))
        model.add(Conv2D(64, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal', activation='relu'))
        model.add(Conv2D(32, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal', activation='relu'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(AveragePooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.5))

        model.add(Flatten())

        model.add(Dense(128, kernel_initializer='he_normal', activation='relu'))
        model.add(Dense(16, kernel_initializer='he_normal', activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(optimizer=Adam(lr=self.lr_schedule(1)),
                      loss=keras.losses.categorical_crossentropy,
                      metrics=['accuracy'])
        return model

