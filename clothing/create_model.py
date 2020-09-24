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


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model



X_train =  np.load('train_images.npy', allow_pickle=True)
y_train =  np.load('train_labels.npy', allow_pickle=True)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# show image
data_idx = 1
plt.imshow(X_train[data_idx,:,:]/255, cmap='binary')
class_number = y_train[data_idx]
class_text = class_names[class_number]
print(f'This is a {class_text}')


# data prep
X_train = X_train/255
X_train = X_train.reshape(-1, 784)

y_train = y_train.reshape(-1, 1)
y_train = y_train.astype('float64')

#Hot Encoder for y
from sklearn.preprocessing import OneHotEncoder
clothing_ohe = OneHotEncoder(sparse=False)
clothing_ohe.fit(y_train)
y_train = clothing_ohe.transform(y_train)


#Split the data: 
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = 420)
# create a model and train it:

# prep validation data
X_val =  np.load('val_images.npy', allow_pickle=True)
X_val = X_val/255
X_val = X_val.reshape(-1, 784)

#############################################################################################################

image_input = Input(shape=(784,))
first_hidden_layer = Dense(784, activation='relu')(image_input)
first_hidden_layer_d = Dropout(0.2)(first_hidden_layer)
second_hidden_layer = Dense(600, activation='relu')(first_hidden_layer)
second_hidden_layer_d = Dropout(0.2)(second_hidden_layer )
third_hidden_layer = Dense(200, activation='relu')(second_hidden_layer_d)
output_layer = Dense(10, activation='softmax')(third_hidden_layer)


model_pic = Model(inputs = image_input, outputs=output_layer)
model_pic.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])


history_pic = model_pic.fit(X_train, y_train, batch_size=64, epochs=10, validation_data = (X_test, y_test))



from sklearn.metrics import accuracy_score

y_train_pred = model_pic.predict(X_train)
accuracy_score(y_train, y_train_pred>0.5) # 0.8782352941176471


y_test_pred = model_pic.predict(X_test)
accuracy_score(y_test, y_test_pred>0.5) #0.8782352941176471


y_train_pred = model_pic.predict(X_val)
y_train_pred =y_train_pred.astype('float64')

prediction_1 = np.argmax(y_train_pred[:,])






import matplotlib.pyplot as plt
# make each plot seperatly 
plt.plot(history_pic.history['loss'], label='train loss')
plt.plot(history_pic.history['val_loss'], label='test loss')
plt.legend(loc='upper right')
plt.show()











#Lage np array fil som kan legges inn i filen som skal leveres: 
y_val_pred = model_clothes.predict(X_val)
y_val_pred_argmax = np.argmax(y_val_pred, axis=1)








# predic validation data
my_prediction = np.array([0,1,2])

# save predictions
my_name = 'Kristoffer'
np.save(f'{my_name}_predictions.npy', my_prediction)