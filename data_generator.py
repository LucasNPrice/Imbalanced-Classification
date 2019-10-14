import pandas as pd
import linecache
import numpy as np
# from keras.utils import np_utils 

class DataGenerator():

  def __init__(self, dataPath, data_indices, scaler, SMOTE = False):
    self.dataPath = dataPath
    self.data_indices = data_indices
    self.scaler = scaler
    self.test_targets = []
    self.test_index = []
    self.SMOTE = SMOTE

  def generate(self, batchsize = 32, mode = 'train'):

    while True:
      batch_data = []
      for i, index in enumerate(self.data_indices):
        if len(batch_data) == batchsize:
          batch_data = np.array(batch_data)
          targets = batch_data[:,-1]
          features = batch_data[:,0:-1]
          features = self.scaler.transform(features)
          yield (features, targets)

          batch_data = []
          continue

        # +2 because of reading/indexing mismatch
        # if mode == 'test':
        #   self.test_index.append(index)
        line = linecache.getline(self.dataPath, index+2).rstrip().split(',')
        if self.SMOTE:
          # if mode != 'train':
          #   line = np.delete(line, [1,9])
          pass
        else:
          line = np.delete(line, [1,9])
        batch_data.append(np.array(line))

 