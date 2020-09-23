import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model


import numpy as np
from sklearn.datasets import fetch_openml

#Get the data (pict)
mnist = fetch_openml('mnist_784', version=1)

#Extract features and labels
X = mnist['data']
y = mnist['target']

# investigate data
print(X.shape) # 70000 images, with 784 pixels
print(y.shape) # 70000 labels of which number the picture is of

# extract single digit
instance_index = 210
single_digit = X[instance_index,:]
single_digit_image = single_digit.reshape(28,28)

# plot the digit
import matplotlib.pyplot as plt
plt.imshow(single_digit_image, cmap='binary')
plt.axis('off')
plt.show()
# each instance is a and drawn digit between 0 and 1 (28X28 pixels equals 784 flatten pixels)
print(y[instance_index])
print(type(y[instance_index])) # every label is a string

# change labels to number
y=y.astype(np.uint8)
print(type(y[instance_index])) # every label is now numeric
########################################################################
# Data prep
X=X/255
y_is5 = (y==5)
y_is5 = y_is5.astype('float64')

# check that X and y has same type
assert y_is5.dtype==X.dtype

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y_is5, test_size=0.1, random_state=420)

# Create model
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model


image_input = Input(shape=(784,))
first_hidden_layer = Dense(512, activation='relu')(image_input)
first_hidden_layer_d = Dropout(0.2)(first_hidden_layer)
second_hidden_layer = Dense(256, activation='relu')(first_hidden_layer)
second_hidden_layer_d = Dropout(0.2)(second_hidden_layer )
third_hidden_layer = Dense(32, activation='relu')(second_hidden_layer)
output_layer = Dense(1, activation='sigmoid')(third_hidden_layer)


model_is5 = Model(inputs = image_input, outputs=output_layer)
model_is5.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])



history_pic = model_is5.fit(X_train, y_train, batch_size=64, epochs=10, validation_data = (X_test, y_test))



from sklearn.metrics import accuracy_score

y_train_is5_pred = model_is5.predict(X_train)
accuracy_score(y_train, y_train_is5_pred>0.5)


y_test_is5_pred = model_is5.predict(X_test)
accuracy_score(y_test, y_test_is5_pred>0.5)


model_is5.predict(single_digit.reshape(1,784))




import matplotlib.pyplot as plt
# make each plot seperatly 
plt.plot(history_pic.history['loss'], label='train loss')
plt.plot(history_pic.history['val_loss'], label='test loss')
plt.legend(loc='upper right')
plt.show()