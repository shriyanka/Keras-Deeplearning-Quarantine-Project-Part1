import random
import keras
import numpy as np
from sklearn.utils import shuffle


class Data:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.input_shape = (112, 112, 3)

    def loadTrainData(self, trainFile, trainLabelsFile):
        trainData = np.load(trainFile)
        trainData = np.rollaxis(trainData, 1, 4)

        trainLabels = np.load(trainLabelsFile)
        trainLabels = keras.utils.to_categorical(trainLabels)

        trainData, trainLabels = shuffle(trainData, trainLabels)

        validationSize = int(0.8 * len(trainData))
        validationData = trainData[validationSize:]
        validationLabels = trainLabels[validationSize:]

        trainData = trainData[:validationSize]
        trainLabels = trainLabels[:validationSize]

        return trainData, trainLabels, validationData, validationLabels

    def loadTestData(self, testFile, testLabelsFile):
        testData = np.load(testFile)
        testData = np.rollaxis(testData, 1, 4)

        testLabels = np.load(testLabelsFile)
        testLabels = keras.utils.to_categorical(testLabels)

        return testData, testLabels

    def trainDataGenerator(self, train_data, train_labels):
        while True:
            batch_features = np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
            batch_labels = np.zeros((self.batch_size, train_labels.shape[1]))
            for i in range(self.batch_size):
                index = random.randrange(len(train_data))
                batch_features[i] = train_data[index]
                batch_labels[i] = train_labels[index]

            yield batch_features, batch_labels

    def valDataGenerator(self, val_data, val_labels):
        while True:
            batch_features = np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
            batch_labels = np.zeros((self.batch_size, val_labels.shape[1]))
            for i in range(self.batch_size):
                index = random.randrange(len(val_data))
                batch_features[i] = val_data[index]
                batch_labels[i] = val_labels[index]

            yield batch_features, batch_labels
