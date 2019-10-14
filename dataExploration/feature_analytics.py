import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocess import Preprocess
from data_generator import DataGenerator

class Analyze():

  def __init__(self, data):
    self.data = data

  def density(self, attributes):
    print(attributes)
    for i, att in enumerate(attributes):
      ax = plt.subplot(4, 3, i + 1)
      ax.hist(self.data[i], color = 'blue', edgecolor = 'black')
      ax.set_title(att)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------
# below should be ran on separate script. pasted here to reduce file number 
# original dir and data path
dir_ = '/Users/lukeprice/Codes/BrainCorp/'
data = dir_ + 'device_failure_data_scientist.csv'

# train, val, test split
d = Preprocess(data)
d.train_test_split((.8,.1,.1))
test_indices = d.data_indices['test']
""" we create separate test indices var above to use later - dont use SMOTE test indices """

# perform SMOTE and get new object with new train, val, test indices
SMOTE = d.smote(ratio = 1, newFileName = dir_ + 'smoteData', redoProcessing = True)
""" we will not use SMOTE.data_indices['test'] """

# select which attributes to transform
# STANDARDIZE HERE< NOT NORALIZE ( MEAN 0 STD 1 )
attributes = SMOTE.data.columns.tolist()[:-1]
train_scaler = SMOTE.normalize(attributes = attributes)
val_scaler = SMOTE.normalize(attributes = attributes, mode = 'validation')

# analyze distributions 
from feature_analytics import Analyze
scale_attributes = SMOTE.data.columns.tolist()[:-1]
train_data = SMOTE.data.iloc[SMOTE.data_indices['train'],:]
train_trans = train_scaler.transform(train_data[scale_attributes])
Atrain = Analyze(data = train_trans)
Atrain.density(attributes = scale_attributes)

val_data = SMOTE.data.iloc[SMOTE.data_indices['validation'],:]
val_trans = val_scaler.transform(val_data[scale_attributes])
Aval = Analyze(data = val_trans)
Aval.density(attributes = scale_attributes)

test_scaler = d.normalize(attributes = scale_attributes, mode = 'test')
test_data = pd.read_csv(data)
test_X = test_data.iloc[test_indices,:][scale_attributes]
test_X = np.array(test_X)
test_trans = test_scaler.transform(test_X)

Atest = Analyze(data = test_trans)
Atest.density(attributes = scale_attributes)

