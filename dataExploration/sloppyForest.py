import pandas as pd
import numpy as np
from preprocess import Preprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


dir_ = '/Users/lukeprice/Codes/BrainCorp/'
dataPath = dir_ + 'device_failure_data_scientist.csv'

# train, val, test split
d = Preprocess(dataPath)
d.train_test_split((.8,.1,.1))

# sampling
# d.downsample(n_samples = len(d.train_Pos))
d.upsample(n_samples = len(d.train_Neg))

data = pd.read_csv(dataPath).drop(['device', 'attribute8'], axis = 1)
train = data.iloc[d.data_indices['train'],:]
val = data.iloc[d.data_indices['validation'],:]
input()

def forest(train, test, n_trees = 10, weights = None):

  X_train = np.array(train.drop(['failure'], axis = 1))
  y_train = np.array(train['failure'])
  X_val = np.array(val.drop(['failure'], axis = 1))
  y_val = np.array(val['failure'])

  n_trees = n_trees
  weights = weights
  rfc = RandomForestClassifier(n_estimators = n_trees, 
    class_weight = weights).fit(X_train, y_train)
  predictions = rfc.predict(X = X_val)

  # get F1
  f1 = F1(y_true = y_val, y_pred = predictions)
  print('Trees: {} - F1: {}'.format(n_trees, round(f1, 5)))
# weights = [{'0:1,1:1'}, {'0:1,1:2'}, {'0:1,1:3'}, {'0:1,1:4'}]

def F1(y_true, y_pred):
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
  precision = tp / (tp + fp)
  recall = tp / (tp + fn)
  f1_score = 2 * ((precision * recall) / (precision + recall))
  return f1_score

forest(train = train, test = val, n_trees = 50)
