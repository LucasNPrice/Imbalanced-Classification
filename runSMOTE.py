from preprocess import Preprocess
from data_generator import DataGenerator
from model import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

"""
train with SMOTE data
validation and test with original data 
"""

dir_ = ...
data = dir_ + 'device_failure_data_scientist.csv'
remove_attributes = ['device','attribute8']

""" train, val, test split + remove unwanted attributes """
d = Preprocess(data)
d.drop_attributes(remove_attributes)
d.train_test_split((.8,.1,.1))

""" perform SMOTE - synthetically upsampled minority classes
    NOTE: we will only use SMOTE for training """
SMOTE = d.smote(ratio = 1, newFileName = dir_ + 'smoteData', redoProcessing = True)

""" select which attributes to transform
    fit train scaler on SMOTE data
    fit validation scaler on non-SMOTE data """
attributes = SMOTE.data.columns.tolist()[:-1]
train_scaler = SMOTE.normalize(attributes = attributes)
val_scaler = d.normalize(attributes = attributes, mode = 'validation')

""" create generator objects """
trainGen = DataGenerator(dataPath = SMOTE.dataPath, 
  data_indices = SMOTE.data_indices['train'], 
  scaler = train_scaler, 
  SMOTE = True)
valGen = DataGenerator(dataPath = d.dataPath, 
  data_indices = d.data_indices['validation'], 
  scaler = val_scaler)

""" compile and train model with binary cross-entropy"""
model = Model(lr = 0.0001, batchsize = 124, lossFun = 'binary_crossentropy')
model.compile_model()
model.train_model(epochs = 100, 
  trainGenerator = trainGen,
  valGenerator = valGen)

""" Testing - remember, we use orginal Non-smote data for testing """
test_indices = d.data_indices['test']
test_X = d.data.iloc[test_indices,:][attributes]
test_X = np.array(test_X)
test_y = np.array(d.data.iloc[test_indices,-1])

""" fit and apply new scaler on test data """
test_scaler = d.normalize(attributes = attributes, mode = 'test')
test_scaler.transform(test_X)

""" get predictions for test data and evaluate metrics """
model.predict(test_X = test_X)
model.evaluate(thresh = 0.5, y_true = test_y)

print(Counter([float(i) for i in model.predictions]))
plt.hist(model.predictions)
plt.show()



