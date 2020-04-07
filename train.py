from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from model import Model
from data import Data
import sys

trainDataFile = sys.argv[1]
trainLabelsFile = sys.argv[2]
modelFile = sys.argv[3]

BATCH_SIZE = 32
LEARNING_RATE = 1e-2
EPOCHS = 50
NUM_CLASSES = 10

dataObj = Data(BATCH_SIZE)
trainData, trainLabels, valData, valLabels = dataObj.loadTrainData(trainDataFile, trainLabelsFile)

print("Train Data", trainData.shape)
print("Train Labels", trainLabels.shape)

print("Validation Data", valData.shape)
print("Validation Labels", valLabels.shape)
print(valLabels[0], trainLabels[0])

STEPS_PER_EPOCH_TRAIN = len(trainData) // BATCH_SIZE
STEPS_PER_EPOCH_VAL = len(valData) // BATCH_SIZE

trainDataGen = dataObj.trainDataGenerator(trainData, trainLabels)
valDataGen = dataObj.valDataGenerator(valData, valLabels)

model = Model(trainData[0].shape, NUM_CLASSES, LEARNING_RATE, EPOCHS)

lrScheduler = LearningRateScheduler(model.lr_schedule)
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
modelCheckpoint = ModelCheckpoint(modelFile, monitor='val_loss', verbose=1, save_best_only=True)

model = model.model()
# model.load_weights(modelFile)

model.fit_generator(
    trainDataGen,
    steps_per_epoch=STEPS_PER_EPOCH_TRAIN,
    validation_data=valDataGen,
    validation_steps=STEPS_PER_EPOCH_VAL,
    epochs=EPOCHS,
    callbacks=[lrScheduler, earlyStopping, modelCheckpoint],
    verbose=1
)