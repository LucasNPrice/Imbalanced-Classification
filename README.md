# Imbalanced-Classification
Methods and models for classification with large class imbalances. 

Goal: classify failure and non-failures (0/1). 
Challenge: large class imbalance (124388:106 > 1000:1).

Two approaches:
  1. Focal Loss with downsampling of majority classes - runFocal.py
  2. SMOTE: synthetic minority over-sampling technique to oversample synthetic minority classes - runSMOTE.py
  
preprocess.py
- Preprocessing methods. 
data_generator.py
- Data generator for training on batch to avoid memory limitations.  
model.py
- Deep dense net model for training, testing, and evaluating.
metrics.py
  - Method for compting Focal Loss to over-emphasize minority classes. 
  
 Original data exploration and naive models in dataExploration folder. 
