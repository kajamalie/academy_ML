# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 12:42:49 2020

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
X = X.reshape(-1, 28,28, 1)

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
X_val = X_val.reshape(-1, 28,28, 1)
#############################################################################################################
#%% training the model
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPool2D
from tensorflow.keras.models import Model
tf.keras.layers.BatchNormalization
​
input_layer = Input(shape=(28,28,1))
first_conv = Conv2D(30, (3,3), activation='relu')(input_layer)
pooling_layer = MaxPool2D (pool_size = (3,3))(first_conv)
#second_conv  = Conv2D(25, (3,3), activation='relu')(pooling_layer)
#pooling_layer_2 = MaxPool2D (pool_size = (3,3))(second_conv)
flatten_layer = tf.keras.layers.Flatten()(pooling_layer)
first_hidden_layer = Dense(20, activation='relu')(flatten_layer)
second_hidden_layer = Dense(15, activation='relu')(first_hidden_layer)
output_layer = Dense(10, activation='softmax')(second_hidden_layer)




model_pic = Model(inputs = input_layer, outputs=output_layer)
model_pic.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])


history_pic = model_pic.fit(X_train, y_train, batch_size=150, epochs=8, validation_data = (X_test, y_test))



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

   






### Med denne koden kunne du unngått å bruke OneHotEncoder... (ADD SPARSE to loss)
model_pic = Model(inputs = image_input, outputs=output_layer)
model_pic.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])



#Gjør om til riktig format 
X = X/255
X = X.reshape(-1, 28, 28, 1)


#Denne skal mellom siste layer og output
flatten_layer = tf.keras.layers.Flatten()(first_layer)