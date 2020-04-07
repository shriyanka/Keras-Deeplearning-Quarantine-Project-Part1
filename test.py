from keras.engine.saving import load_model
from data import Data
from model import Model
import numpy as np
import sys

testDataFile = sys.argv[1]
testLabelsFile = sys.argv[2]
modelFile = sys.argv[3]

BATCH_SIZE = 64
NUM_CLASSES = 10

dataObj = Data(BATCH_SIZE)
testData, testLabels = dataObj.loadTestData(testDataFile, testLabelsFile)

print("Test Data", testData.shape)
print("Test Labels", testLabels.shape)

STEPS_PER_EPOCH_TEST = len(testLabels) // BATCH_SIZE

testDataGen = dataObj.valDataGenerator(testData, testLabels)

model = Model(testData[0].shape, NUM_CLASSES).model()
model.load_weights(modelFile)
# predictions = model.predict_generator(
#     testDataGen,
#     steps=STEPS_PER_EPOCH_TEST,
#     verbose=1
# )
# way one - inconsistent somehow, need to debug
loss, acc = model.evaluate_generator(testDataGen, steps=STEPS_PER_EPOCH_TEST)
print("Accuracy- ", acc * 100)
print("Error- ", (1-acc) * 100)

results = model.predict(testData)

# accurate and consistent
count = 0
for i, result in enumerate(results):
    pred = np.argmax(result)
    true = np.argmax(testLabels[i])
    if pred == true:
        count += 1

print("Accuracy- ", count/len(testData) * 100)
print("Error- ", (1- (count/len(testData))) * 100)