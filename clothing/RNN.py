# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 10:02:31 2020

@author: Kaja Amalie
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.executing_eagerly()

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model



X =  np.load('train_images.npy', allow_pickle=True)
y =  np.load('train_labels.npy', allow_pickle=True)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# show image
data_idx = 1
plt.imshow(X[data_idx,:,:]/255, cmap='binary')
class_number = y[data_idx]
class_text = class_names[class_number]
print(f'This is a {class_text}')


# data prep
X = X/255
X = X.reshape(-1, 28,28)

y = y.reshape(-1, 1)
y = y.astype('float64')

#Hot Encoder for y
from sklearn.preprocessing import OneHotEncoder
clothing_ohe = OneHotEncoder(sparse=False)
clothing_ohe.fit(y)
y = clothing_ohe.transform(y)
  

#Split the data: 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 420)
# create a model and train it:

# prep validation data
X_val =  np.load('val_images.npy', allow_pickle=True)
X_val = X_val/255
X_val = X_val.reshape(-1, 784)
X_val = X_val.reshape(-1, 28,28)
#############################################################################################################
#%% training the model
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPool2D, LSTM, Embedding
from tensorflow.keras.models import Model
tf.keras.layers.BatchNormalization
​
#1 test: 85%
input_layer = Input(shape=(28,28))
lstm_layer_1 = LSTM(100, return_sequences=True)(input_layer)
flatten_layer = tf.keras.layers.Flatten()(lstm_layer_2)
first_hidden_layer = Dense (15, activation='relu')(flatten_layer)
output_layer = Dense(10, activation='softmax')(first_hidden_layer)

model_pic = Model(inputs = input_layer, outputs=output_layer)
model_pic.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

history_pic = model_pic.fit(X_train, y_train, batch_size=150, epochs=10, validation_data = (X_test, y_test))


#2 test: train: 0.8945, train 0.8917
input_layer = Input(shape=(28,28))
lstm_layer_1 = LSTM(50, return_sequences=True)(input_layer)
lstm_layer_2 = LSTM(20, return_sequences=True)(lstm_layer_1)
flatten_layer = tf.keras.layers.Flatten()(lstm_layer_2)
first_hidden_layer = Dense (100, activation='relu')(flatten_layer)
output_layer = Dense(10, activation='softmax')(first_hidden_layer)

model_pic = Model(inputs = input_layer, outputs=output_layer)
model_pic.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

history_pic = model_pic.fit(X_train, y_train, batch_size=150, epochs=10, validation_data = (X_test, y_test))


#3 test


from sklearn.metrics import accuracy_score

y_train_pred = model_pic.predict(X_train) #92,9
accuracy_score(y_train, y_train_pred>0.5) 

y_test_pred = model_pic.predict(X_test)
accuracy_score(y_test, y_test_pred>0.5) 



y_val_pred = model_pic.predict(X_val)
y_val_pred =y_train_pred.astype('float64')

pred1 = np.argmax(y_val_pred, axis=1)

#%%


import matplotlib.pyplot as plt
# make each plot seperatly 
plt.plot(history_pic.history['loss'], label='train loss')
plt.plot(history_pic.history['val_loss'], label='test loss')
plt.legend(loc='upper right')
plt.show()

   



y_val_pred = model_pic.predict(X_val)
y_val_pred_argmax = np.argmax(y_val_pred, axis=1)



# predic validation data
my_prediction = np.array([0,1,2])

# save predictions
my_name = 'Kaja'
np.save(f'{my_name}_predictions_RNN.npy', my_prediction)