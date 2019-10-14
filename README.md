# Imbalanced-Classification
## Methods and models for classification with large class imbalances. 

**Goal**: classify failure and non-failures (0/1). 

**Challenge**: large class imbalance (124388:106 > 1000:1).

Two approaches:
  1. Focal Loss with downsampling of majority classes - **runFocal.py**
  2. SMOTE: synthetic minority over-sampling technique to oversample synthetic minority classes - **runSMOTE.py**
  

**Preprocess.py**
  - Preprocessing methods. 
  
**data_generator.py**
  - Data generator for training on batch to avoid memory limitations.  
  
**model.py**
  - Deep dense net model for training, testing, and evaluating.
  
**metrics.py**
  - Method for compting Focal Loss to over-emphasize minority classes. 
  
Original data exploration and naive models in **dataExploration** folder. 
  - **data_assurance.py**: understanding the problem. 
  - **data_exploration.py**: summary statistics, visualizations, distributions. 
  - **sloppyForest.py**: random forest to ensure data complexity required Neural Nets.
  - **feature_analytics**: assess feature distributions after transformations to ensure train, val, test distributions were compatible. 
  
**NOTES**:
  - layman's idea: use data from day *t* to predict failure/non-failure on day *t*.  
  - it may be useful (possibly even better) to use an autoregressive time-series model (ex: use data on day *t-1* to predict failure/non-failure on day *t*).
  - more knowledge of the problem and specifics of the data aquisition process would aid in determining which approach to use. 
