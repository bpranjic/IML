import pandas as pd
import numpy as np
import scipy as sc
from scipy import linalg

def transform(x):
    return np.array([x[0],x[1],x[2],x[3],x[4],x[0]**2,x[1]**2,x[2]**2,x[3]**2,x[4]**2,np.e**x[0],np.e**x[1],np.e**x[2],np.e**x[3],np.e**x[4],np.cos(x[0]),np.cos(x[1]),np.cos(x[2]),np.cos(x[3]),np.cos(x[4]), 1.0])

#import file
train_file = pd.read_csv("train.csv")

#separate file
train_features = train_file.copy()
train_ids = train_features.pop('Id')
train_labels = train_features.pop('y')

#apply transformations
features = np.array(train_features)
xtrain = np.empty(shape=(700,21))
ytrain = np.empty(shape=(700,1))
for i in range (700):
    xtrain[i] = transform(features[i])
    ytrain[i][0] = train_labels[i]

#pseudoinverse
pinv = sc.linalg.pinv(xtrain)
weights = np.matmul(pinv,ytrain)

#create submission
submission = pd.DataFrame(weights)
submission = submission.to_csv("submission.csv", index=False, header=False)

