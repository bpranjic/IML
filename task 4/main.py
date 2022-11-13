import numpy as np
import tensorflow as tf
import pandas as pd
import random
import os


from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Input, LeakyReLU



random.seed(420)
np.random.seed(420)
tf.random.set_seed(420)

pretrain_features_df = pd.read_csv("pretrain_features.csv")
pretrain_labels_df = pd.read_csv("pretrain_labels.csv")
test_features_df = pd.read_csv("test_features.csv")
train_features_df = pd.read_csv("train_features.csv")
train_labels_df = pd.read_csv("train_labels.csv")


pretrain_ids = pretrain_features_df.pop("Id")
pretrain_SMILE = pretrain_features_df.pop("smiles")
pretrain_labels = pretrain_labels_df.pop("lumo_energy")

train_ids = train_features_df.pop("Id")
train_SMILE = train_features_df.pop("smiles")
train_labels = train_labels_df.pop("homo_lumo_gap")

test_ids = test_features_df.pop("Id")
test_SMILE = test_features_df.pop("smiles")


pretrain_features = np.array(pretrain_features_df)
pretrain_labels = np.array(pretrain_labels)

train_features = np.array(train_features_df)
train_labels = np.array(train_labels)

test_features = np.array(test_features_df)



x = x_in = Input((1000,))
x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
x = Dropout(0.1)(x)
x = Dense(1152)(x)
x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
x = Dense(288)(x)
x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
x = Dense(72)(x)
x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
x = Dense(18)(x)
x = Dropout(0.1)(x)
x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
x = Dense(1)(x)

model = Model(inputs=x_in, outputs=x)

model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.001))

model.fit(x=pretrain_features, y=pretrain_labels, epochs=25)

lumo_prediction = model.predict(train_features)


lumo_prediction = np.array(lumo_prediction)
homo_predicted = np.subtract(lumo_prediction, train_labels)

homo_predicted = pd.DataFrame(homo_predicted, columns=["y"])

submission = pd.concat([train_ids, homo_predicted], axis=1)

submission.to_csv("submission.csv", index=False)
