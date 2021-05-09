import app
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score

dataset = pd.read_csv('admissions_data.csv')

print(dataset.head())
print(dataset.describe())

dataset = dataset.drop(['Serial No.'], axis = 1)
labels = dataset.iloc[:, -1]
features = dataset.iloc[:, 0:-1]

features_training_set, features_test_set, labels_training_set, labels_test_set = train_test_split(features, labels, test_size = 0.25, random_state = 10)

num_features = features.select_dtypes(include=['float64', 'int64'])
num_columns = num_features.columns
 
ct = ColumnTransformer([("only numeric", StandardScaler(), num_columns)], remainder='passthrough')

features_train_scaled = ct.fit_transform(features_training_set)
features_test_scaled = ct.transform(features_test_set)
features_training_set = pd.DataFrame()
features_test_set = pd.DataFrame()

num_features = features.select_dtypes(include=['float64', 'int64'])
num_columns = num_features.columns

adm_model = Sequential()

input = InputLayer(input_shape = (features.shape[1], ))

adm_model.add(input)
adm_model.add(Dense(64, activation = "relu"))
adm_model.add(Dense(1))

print(adm_model.summary())

opt = Adam(learning_rate = 0.1)

adm_model.compile(loss = 'mse', metrics = ['mae'], optimizer = opt)

print(adm_model.summary())

adm_model.compile(loss = 'mse', metrics = ['mae'], optimizer = opt)

adm_model.fit(features_train_scaled, labels_training_set, epochs=40, batch_size=1, verbose=1)

res_mse, res_mae = adm_model.evaluate(features_test_scaled, labels_test_set, verbose=0)

print(res_mse)
print(res_mae)

stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

history = adm_model.fit(features_train_scaled, labels_training_set, epochs=40, batch_size=1, verbose=1, validation_split=0.2, callbacks=[stop])


fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['mae'])
ax1.plot(history.history['val_mae'])
ax1.set_title('model mae')
ax1.set_xlabel('epoch')
ax1.set_ylabel('MAE')
ax1.legend(['train', 'validation'], loc='upper left')

ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('model loss')
ax2.set_xlabel('epoch')
ax2.set_ylabel('loss')
ax2.legend(['train', 'validation'], loc='upper left')

fig.tight_layout()
fig.savefig('static/images/my_plots.png')


