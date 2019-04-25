from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import model_from_json
import pickle
import time

def add_convolutional_layers(numOfLayers, model, filters, kernelSize, poolSize):
    for i in range(numOfLayers):
        model.add(Conv2D(filters, (kernelSize, kernelSize)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(poolSize, poolSize)))

def add_dense_layers(numOfLayers, model, units):
    for i in range(numOfLayers):
        model.add(Dense(units))
        model.add(Activation('relu'))

def add_input_layer(X, model, filters, kernelSize, poolSize):
    model.add(Conv2D(filters, (kernelSize, kernelSize), input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(poolSize, poolSize)))

def add_output_layer(model):
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

def load_data(fileName, datadir):
    pickleIn = open(datadir + "pickles\\" + fileName, "rb")
    return pickle.load(pickleIn)

def save_model(model, datadir, modelName):
    modelJson = model.to_json()
    with open(datadir + "\\models\\" + modelName + ".json", 'w') as jsonFile:
        jsonFile.write(modelJson)
    model.save_weights(datadir + "\\models\\" + modelName + ".h5")

def load_model(datadir, modelName):
    jsonFile = open(datadir + "\\models\\" + modelName + ".json", 'r')
    modelJson = jsonFile.read()
    jsonFile.close()
    model = model_from_json(modelJson)
    model.load_weights(datadir + "\\models\\" + modelName + ".h5")
    return model

print("Enter the neural network data directory\n"
      "e.g. C:\\Users\singe\Desktop\\training")
dataDir = input()
XTrain = load_data("XTrain.pickle", dataDir + "\\generated\\")
yTrain = load_data("yTrain.pickle", dataDir + "\\generated\\")
XTest = load_data("XTest.pickle", dataDir + "\\test\\")
yTest = load_data("yTest.pickle", dataDir + "\\test\\")
testData = (XTest, yTest)
print("Do you want to load or enter your model?")
print("(enter \"load\" or \"enter\")")
enterType = input()
print("Enter neural network model name:")
modelName = input()
if (enterType == "enter"):

    print("Enter neural network parameters")
    print("(best results was on 3 conv layers with 64 filters, kernel size = 3, pool size = 2,\n"
          "0 dense layers, 8 batches and 20 epochs. ~93.48%)")
    print("Number on convolution layers:", end=' ')
    numOfConvLayers = int(input())
    print("Number of output filters in the convolution layers:", end=' ')
    filters = int(input())
    print("Kernel size (height and width of 2D convolution window):", end=' ')
    kernelSize = int(input())
    print("Number on dense layers:", end=' ')
    numOfDenseLayers = int(input())
    print("Number of output units in the dense layers:", end=' ')
    units = int(input())
    print("Size of max pooling windows:", end=' ')
    poolSize = int(input())
    print("Batch size:", end=' ')
    batchSize = int(input())
    print("Num of epochs:", end=' ')
    numOfEpochs = int(input())

    logName = "{}-conv-{}-filters-{}-batches-{}".format(numOfConvLayers, filters, batchSize, int(time.time()))
    tensorBoard = TensorBoard(log_dir='C:\\Users\\singe\\Desktop\\training\\logs\\{}'.format(logName))

    model = Sequential()
    add_input_layer(XTrain, model, filters, kernelSize, poolSize)
    add_convolutional_layers(numOfConvLayers, model, filters, kernelSize, poolSize)
    model.add(Flatten())
    add_dense_layers(numOfDenseLayers, model, units)
    add_output_layer(model)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(XTrain, yTrain,
                  batch_size=batchSize,
                  epochs=numOfEpochs,
                  validation_split=0.3,
                  callbacks=[tensorBoard],
                  validation_data=testData)

    save_model(model, dataDir, modelName)
else:

    model = load_model(dataDir, modelName)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    score = model.evaluate(XTest, yTest, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

########################################################################################################################
# research sets
# filters_set = [32, 64, 128]
# numOfConvLayers_set = [1, 2, 3, 4]
# batchSize_set = [8, 16, 32]
########################################################################################################################

########################################################################################################################
# research code
# for numOfConvLayers in numOfConvLayers_set:
#     for filters in filters_set:
#         for batchSize in batchSize_set:
#             logName = "{}-conv-{}-filters-{}-batches-{}".format(numOfConvLayers, filters, batchSize, int(time.time()))
#             tensorBoard = TensorBoard(log_dir='C:\\Users\\singe\\Desktop\\training\\logs\\{}'.format(logName))
#
#             model = Sequential()
#             add_input_layer(XTrain, model, filters, kernelSize, poolSize)
#             add_convolutional_layers(numOfConvLayers, model, filters, kernelSize, poolSize)
#             model.add(Flatten())
#             add_output_layer(model)
#
#             model.compile(loss='binary_crossentropy',
#                           optimizer='adam',
#                           metrics=['accuracy'])
#
#             model.fit(XTrain, yTrain,
#                       batch_size=batchSize,
#                       epochs=numOfEpochs,
#                       validation_split=0.3,
#                       callbacks=[tensorBoard],
#                       validation_data=testData)
########################################################################################################################