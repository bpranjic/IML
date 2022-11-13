import tensorflow as tf 
import pandas as pd
import numpy as np
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge
from keras import backend as K
import matplotlib.pyplot as plt
import scipy as sc


def transform(x1,x2,x3,x4,x5):
    return np.array([x1,x2,x3,x4,x5,x1**2,x2**2,x3**2,x4**2,x5**2,np.e**x1,np.e**x2,np.e**x3,np.e**x4,np.e**x5,np.cos(x1),np.cos(x2),np.cos(x3),np.cos(x4),np.cos(x5), 1])

#import file
train_file = pd.read_csv("train.csv")

#separate file
train_features = train_file.copy()
train_ids = train_features.pop('Id')
train_labels = train_features.pop('y')

#apply transformations
features = np.array(train_features)
ftrs = np.empty(shape=(700,21))
for i in range (700):
    ftrs[i] = transform(features[i][0],features[i][1],features[i][2],features[i][3],features[i][4])


#pseudoinverse
pinv = sc.linalg.pinv2(ftrs)

y_train = np.array(train_labels)
ytrain = np.empty(shape=(700,1))
for i in range(700):
    ytrain[i][0] = train_labels[i]

res = np.matmul(pinv,ytrain)

print(pinv.shape)
print(ftrs.shape)
print(ytrain.shape)
print(res.shape)
print(res)


print("-----------------------")

sub_best = pd.read_csv("submission.csv", header=None)
sub2 = np.array(sub_best)
res2 = np.matmul(ftrs, sub2)
res3 = np.abs(res2-ytrain)
res3 = res3**2
sum0 = 0.0
for i in range(700):
    sum0 += res3[i]
sum0 = sum0/700.0
sum0 = sum0**0.5
print(sum0)