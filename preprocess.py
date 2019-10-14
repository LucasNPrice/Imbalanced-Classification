import pandas as pd
import numpy as np
from collections import Counter
from random import shuffle
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

class Preprocess():
  """
  NOTE: sampling methods should be called AFTER train_test_split
  """
  def __init__(self, dataPath):
  
    self.data = pd.read_csv(dataPath)
    self.dataPath = dataPath
    self.data_indices = {}
    print(Counter(self.data['failure']))
    input()

  def drop_attributes(self, attributes):
    self.data = self.data.drop(attributes, axis = 1)

  def downsample(self, n_samples):
    """
    downsample majority/negative/non-rare classes
    """
    downsample_index = np.random.choice(self.train_Neg, n_samples, replace = False)
    train_indices = list(self.train_Pos) + list(downsample_index)
    self.data_indices['train'] = train_indices

  def upsample(self, n_samples):
    """
    upsample minority/positive/rare classes
    """
    upsample_index = np.random.choice(self.train_Pos, n_samples, replace = True)
    train_indices = list(self.train_Neg) + list(upsample_index)
    self.data_indices['train'] = train_indices

  def smote(self, ratio, newFileName, redoProcessing = False):
    """
    syntheic minority oversampling
    since we are training on batch, ...
    ... we will write to diskfile and load new smote data in when training 
    """
    sm = SMOTE(ratio = ratio)
    col_names = self.data.columns.tolist()
    X_train = self.data.iloc[self.data_indices['train'],0:-1]
    y_train = self.data.iloc[self.data_indices['train'],-1]

    X, y = sm.fit_sample(X_train, y_train)
    smote_data = pd.DataFrame(np.concatenate((X,y[:,None]),axis=1))
    smote_data.columns = col_names
    smote_data.to_csv(newFileName + '.csv', index = False)

    if redoProcessing:
      """ process smote data """
      smote_path = newFileName + '.csv'
      rp = Preprocess(smote_path, withPickel = False)
      smote_index = [i for i in range(len(rp.data))]
      shuffle(smote_index)
      rp.data_indices['train'] = smote_index
      return rp
    else: 
      """ else return smote data path """
      return smote_path

  def normalize(self, attributes, mode = 'train'):
    """ fit scaler for transformation """
    indices = self.data_indices[mode]
    fit_on_data = self.data.iloc[self.data_indices[mode]][attributes]
    scaler = MinMaxScaler() 
    return scaler.fit(fit_on_data)

  def train_test_split(self, split_ratios):
    """
    get train, val, test splits with equal proportions of hard positive classes
    """
    one_indices = self.data.index[self.data['failure'] == 1].tolist()
    zero_indices = self.data.index[self.data['failure'] == 0].tolist()

    self.train_Pos, val_Pos, test_Pos = np.split(one_indices, 
      [int(split_ratios[0] * len(one_indices)), 
      int((np.add(*split_ratios[0:2])) * len(one_indices))])

    self.train_Neg, val_Neg, test_Neg = np.split(zero_indices, 
      [int(split_ratios[0] * len(zero_indices)), 
      int((np.add(*split_ratios[0:2])) * len(zero_indices))])

    train_indices = list(self.train_Pos) + list(self.train_Neg)
    val_indices = list(val_Pos) + list(val_Neg)
    test_indices = list(test_Pos) + list(test_Neg)

    shuffle(train_indices)
    shuffle(val_indices)
    shuffle(test_indices)
    
    """ get index dictionary for training on batch """
    self.data_indices = {
      'train': train_indices,
      'validation': val_indices,
      'test': test_indices
    }


