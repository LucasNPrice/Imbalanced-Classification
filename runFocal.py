from preprocess import Preprocess
from data_generator import DataGenerator
from model import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from metrics import focalLoss

dir_ = ...
data = dir_ + 'device_failure_data_scientist.csv'
remove_attributes = ['device','attribute8']

""" train, val, test split + remove unwanted attributes + downsample majority classes """
d = Preprocess(data)
d.drop_attributes(remove_attributes)
d.train_test_split((.8,.1,.1))
d.downsample(n_samples = len(d.train_Neg)//2)

""" prepare training and validation scalers for regularization """
attributes = d.data.columns.tolist()[:-1]
train_scaler = d.normalize(attributes = attributes)
val_scaler = d.normalize(attributes = attributes, mode = 'validation')

""" create generator objects """
trainGen = DataGenerator(dataPath = d.dataPath, 
  data_indices = d.data_indices['train'], 
  scaler = train_scaler)
valGen = DataGenerator(dataPath = d.dataPath, 
  data_indices = d.data_indices['validation'], 
  scaler = val_scaler)

""" compile and train model with Focal Loss """
model = Model(lr = 0.0001, batchsize = 124, lossFun = focalLoss)
model.compile_model()
model.train_model(epochs = 100, 
  trainGenerator = trainGen,
  valGenerator = valGen)

""" prepare test data - we will not use a generator here """
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