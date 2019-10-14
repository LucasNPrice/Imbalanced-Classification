import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras import optimizers
import metrics
from sklearn.metrics import confusion_matrix
import numpy as np
import metrics


class Model():

  def __init__(self, lr, batchsize, lossFun):
    self.lr = lr
    self.batchsize = batchsize
    self.model = Sequential()
    self.optimizer = optimizers.Adam(lr = self.lr)
    self.lossFun = lossFun

  def compile_model(self):

    self.model.add(Dense(100, 
      input_dim = 9, activation='relu', name = 'Input_Layer'))
    self.model.add(BatchNormalization())
    self.model.add(Dropout(0.2))

    self.model.add(Dense(100, activation='relu'))
    self.model.add(BatchNormalization())
    self.model.add(Dropout(0.2))

    self.model.add(Dense(100, activation='relu'))
    self.model.add(BatchNormalization())
    self.model.add(Dropout(0.2))

    self.model.add(Dense(1000, activation='relu'))
    self.model.add(BatchNormalization())
    self.model.add(Dropout(0.2))

    self.model.add(Dense(1000, activation='relu'))
    self.model.add(BatchNormalization())
    self.model.add(Dropout(0.2))

    self.model.add(Dense(100, activation='relu'))
    self.model.add(BatchNormalization())
    self.model.add(Dropout(0.2))

    self.model.add(Dense(100, activation='relu'))
    self.model.add(Dense(1, activation = 'sigmoid', name = 'Output_Layer'))

    self.model.compile(optimizer = self.optimizer, 
      loss = self.lossFun, 
      metrics = ['accuracy'])

  def train_model(self, epochs, trainGenerator, valGenerator):

    self.model.summary()
    self.model.fit_generator(
      generator = trainGenerator.generate(self.batchsize),
      steps_per_epoch = len(trainGenerator.data_indices) // self.batchsize,
      validation_data = valGenerator.generate(batchsize = self.batchsize, mode = 'validation'),
      validation_steps = len(valGenerator.data_indices) // self.batchsize,
      epochs = epochs)

  def predict_on_batch(self, testGenerator):

    self.predictions = self.model.predict_generator(
      generator = testGenerator.generate(batchsize = self.batchsize, mode = 'test'), 
      steps = len(testGenerator.data_indices) // self.batchsize)

  def predict(self, test_X):  

    self.predictions = self.model.predict(x = np.array(test_X))

  def evaluate(self, thresh, y_true):

    for i, pred in enumerate(self.predictions):
      if pred >= thresh:
        self.predictions[i] = 1
      else:
        self.predictions[i] = 0

    conf_mat = confusion_matrix(y_true, self.predictions, labels = [0,1]).ravel()
    tn, fp, fn, tp = conf_mat.flatten()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)

    p = tp / (tp + fp)
    r = tp / (tp + fn)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))

    print('\ntp : ' + str(tp))
    print('fp : ' + str(fp))
    print('tn : ' + str(tn))
    print('fn : ' + str(fn))
    # print(' ----- Confusion Matrix -----{}  {} {}'.format('\n', conf_mat, '\n'))
    # print('True Positive Rate: {} {}False Positive Rate: {}'.format(round(tpr,3), '\n', round(fpr,3)))
    print('\nPrecision: {} {}Recall: {}'.format(round(p,3), '\n', round(r,3)))
    print('F1 Score: {}'.format(round(f1_score,3)))




