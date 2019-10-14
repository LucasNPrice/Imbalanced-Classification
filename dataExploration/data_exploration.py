import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

"""
Q = question
A = answer
O = observation
"""

dir_ = ...
dat = pd.read_csv(dir_ + 'device_failure_data_scientist.csv')
dat = dat.sort_values(by = ['device', 'date'])
print(dat.shape)


#----------------------------------------------------------------
# Feature exploration 
#----------------------------------------------------------------
dat[['failure']] = dat[['failure']].astype('category')
print(dat.dtypes)
input()

# Q: what is distrubition of fail/not fail? Is it disproportionate? 
outcomes = Counter(dat.loc[:,'failure'])
print(outcomes)
print('No-fail to fail ratio: {}'.format(outcomes[0] / outcomes[1]))
input()
# A: Yes. 106 fail, 124388 do not fail. 
# O: 1:1173 not-fail to fail ratio. Large data disparity. Take this into account when modeling. 


# Q: What are summary statistics?
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  
  print(dat.describe())
input()
# O: attribute 7 and 8 appear to be the same by summary statistics 
# O: odd ranges/quartiles. 


# Q: Is attribute 7 == attribute 8?
seven_equal_eight = all(dat.loc[:,'attribute7'] == dat.loc[:,'attribute8'])
print('Attribute 7 = Attribute 8: {}'.format(seven_equal_eight))
# dat = dat.drop(['attribute8'], axis = 1)
input()
# A: Yes. Remove attribute 7 or 8. 


# Q: How many unique values per attribute?
attributes = dat.columns[2:-1]
for att in attributes:
  print('{} unique values: {}'.format(att, len(set(dat[att]))))
input()
# A: variable


# Q: What do densities/histograms of each attribute X look like?
for i, att in enumerate(attributes):
  ax = plt.subplot(3, 3, i + 1)
  ax.hist(dat[att], color = 'blue', edgecolor = 'black')
  ax.set_title(att)
plt.tight_layout()
plt.show()
# O: attributes 2,3,4,7,8,9 have odd, heavilty skewed distributions


# Q: Is there a visual pattern between attribute and failure?
for i, att in enumerate(attributes):
  ax = plt.subplot(3, 3, i + 1)
  ax.scatter(dat[att], dat['failure'], c = dat['failure'])
  ax.set_title(att)
plt.tight_layout()
plt.show()
# A: No, nothing robust. 


# Q: Is there a visual pattern between day, attribute, and failure?
for i, att in enumerate(attributes):
  ax = plt.subplot(3, 3, i + 1)
  ax.scatter(dat['date'], dat[att], c = dat['failure'])
  ax.set_title(att)
plt.tight_layout()
plt.show()
# ugly, but gets the job done. 
# A: Again, no, nothing robust. 