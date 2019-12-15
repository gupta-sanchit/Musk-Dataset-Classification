#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sanchit

"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from time import time
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import h5py


data = pd.read_csv('data.csv')

data.drop('molecule_name',axis = 1,inplace = True)
data.drop('ID', axis = 1, inplace = True)

"""  Splitting the conformation_name model into 3 parts
     1. Molecule Name
     2. ISO -- Sterioisomer Number
     3. CONF -- Conformation Number """

data['MOL'] = np.nan
data['ISO'] = np.nan
data['CONF'] = np.nan

for i in range(len(data['conformation_name'])):
    x = data['conformation_name'][i].replace('+','_').split('_')
    data['MOL'][i] = x[0]
    data['ISO'][i] = x[1]
    data['CONF'][i] = x[2]

# Deleting Unwanted Columns
data.drop('conformation_name', axis = 1, inplace = True)
data.drop('MOL', axis = 1, inplace = True)

# Shuffling the Dataset to generate randomness
df = shuffle(data).reset_index(drop = True)
df.head()

# Separating Input & Output
X = df.drop('class', axis = 1)
Y = df['class']

# Converting dataframe object to numpy array

X = X.to_numpy()
Y = Y.to_numpy()

# print(X.shape)
# print(Y.shape)


# Feature Scaling


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Principal Component Analysis

# Finding the optimal number of features after maintaining 95% variance
pca = PCA()
pca.fit(X_scaled)

current_variance= 0
total = sum(pca.explained_variance_)
optimal_components = 0
while current_variance/total < 0.95:
    current_variance+=pca.explained_variance_[optimal_components]
    optimal_components+=1
    
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)')
plt.title('Musk Dataset PCA Analysis')

plt.annotate('optimal-components(0.95 , 40)', xy=(40, 0.95), xytext=(70, 0.9),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.grid()
plt.show()


# Train-Test Split

#Splitting the Dataset for Training & Testing
# 80-20 Split Ratio is maintained

x_train, x_test, y_train,y_test = train_test_split(X_scaled,Y,test_size = 0.20,random_state = 0)

pca1 = PCA(n_components=optimal_components)
x_train_pca = pca1.fit_transform(x_train)
x_test_pca = pca1.transform(x_test)


# ## Multi Layer Perceptron

model = Sequential()

# Hidden Layer 1
model.add(Dense(20, activation = 'relu',
                kernel_initializer = 'random_normal', input_dim = 40))
# Hidden Layer 2
model.add(Dense(20, activation = 'relu',
               kernel_initializer = 'random_normal'))
# Output Layer
model.add(Dense(2, activation = 'sigmoid',
               kernel_initializer = 'random_normal'))
model.summary()

#Compiling Model
model.compile(loss = 'binary_crossentropy',optimer = 'adam', metrics = ['accuracy'])

tensorBoard = TensorBoard(log_dir = 'logs/{}'.format(time()))

history = model.fit(x_train_pca,to_categorical(y_train, num_classes = 2), epochs= 10 , batch_size = 20,
          validation_data = (x_test_pca,to_categorical(y_test, num_classes = 2)),
         callbacks = [tensorBoard])
# model.metrics_names

# model.save('model-updated.h5')

# Model Accuracy

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.grid()
plt.show()
# plt.savefig('Model-Accuracy-Updated.png')

# Model Loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epochs')
plt.title('Model Loss')
plt.legend(['train','test'])
plt.grid()
plt.show()
# plt.savefig('Model-Loss-Updated.png')


#Classification Report

y_predicted = model.predict(x_test_pca, batch_size = 20)
y_pred_bool = np.argmax(y_predicted, axis=1)

print(classification_report(y_test,y_pred_bool))

