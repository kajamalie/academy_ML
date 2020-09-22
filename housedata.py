
#%%

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model


#Choose the data:
os.chdir(r'C:\Users\Kaja Amalie\Documents\Kaja\Forelesninger\uke_7\Housing_data')

df = pd.read_csv('kc_house_data.csv')

df_selected_column = df[[
    'price', 
    'sqft_living', 
    'sqft_lot', 
    'bedrooms', 
    'bathrooms', 
    'lat',
    'long',
    'view',
    'waterfront',
    'grade']]

#%%
#Create X(input) and y(output) data 
y = df_selected_column['price']
X = df_selected_column [['bedrooms',
                         'bathrooms', 
                         'sqft_living', 
                         'sqft_lot',
                         'lat',
                         'long',
                         'view',
                         'waterfront',
                         'grade']]
 #%%       
#Fill inn values containing null values (create imuter then fill in)
house_imputer = SimpleImputer(strategy='median')
house_imputer.fit(X)

np_train_features = house_imputer.transform(X)
features_column_names = list(X)

X_filled = pd.DataFrame(np_train_features, columns=features_column_names)
#Take this one further if you have missing values - I'm not using this one further 
#now because I know that I have no missing values in X. 
#%%
#Look at the current X data:


#Prep the data
# log transformation
X['sqft_living_log'] = np.log(X['sqft_living'])
del X['sqft_living']
X['sqft_lot_log'] = np.log(X['sqft_lot'])
del X['sqft_lot']

# split dataframe into scaling types
X_no_scl = X[['view', 'waterfront']]
X_minmax_scl = X[['bedrooms', 'grade', 'lat', 'long']]
X__std_scl = X[['bathrooms', 'sqft_living_log', 'sqft_lot_log']]


X_no_scl = X_no_scl.values
house_minmax = MinMaxScaler()
house_minmax.fit(X_minmax_scl )
X_minmax = house_minmax.transform(X_minmax_scl)

house_std = StandardScaler()
house_std.fit(X__std_scl)
X_std = house_std.transform(X__std_scl)


#%%

#concatinate into finished dataset
X_prep = np.concatenate([X_no_scl, X_minmax, X_std], axis=1)
y_prep = np.c_[y]
preped_data_np = np.concatenate([X_no_scl, X_minmax, X_std, y_prep], axis=1)

df_prep_data = pd.DataFrame(data =preped_data_np, columns=['view', 'waterfront','bedrooms', 'grade', 'lat', 'long', 'bathrooms', 'sqft_living_log', 'sqft_lot_log', 'price'])

############################################################################################################################################

df_train, df_test = train_test_split(df_prep_data, test_size = 0.2, random_state = 420)

#create X and y (test and train)
train_X = df_train[['view', 'waterfront','bedrooms', 'grade', 'lat', 'long', 'bathrooms', 'sqft_living_log', 'sqft_lot_log']]
train_y = df_train['price']
test_X = df_test[['view', 'waterfront','bedrooms', 'grade', 'lat', 'long', 'bathrooms', 'sqft_living_log', 'sqft_lot_log']]
test_y = df_test['price']

X_train_arr = np.c_[train_X]
y_train_arr = np.c_[train_y]

X_test_arr = np.c_[test_X]
y_test_arr = np.c_[test_y]

input_layer = Input(shape=(9,))
first_hidden_layer = Dense(9, activation = 'relu')(input_layer)
second_hidden_layer = Dense(7, activation = 'relu')(first_hidden_layer)
third_hidden_layer = Dense(6, activation = 'relu')(second_hidden_layer)
output_layer = Dense(1, activation =None)(third_hidden_layer)

#Combine the input and output into a model: 

model_house = Model(inputs=input_layer, outputs = output_layer)
model_house.compile(optimizer='adam', loss='mse', metrics=['mae'])
model_house.fit(X_train_arr, y_train_arr, batch_size=64, epochs=50)



y_train_pred = model_house.predict(X_train_arr )
y_train_pred = y_train_pred.astype('float64')

mean_absolute_error(y_train_arr, y_train_pred) #144189.56301318953 with 3p:50  143525.60292076346
np.sqrt(mean_squared_error(y_train_arr, y_train_pred)) #236213.42797360278


y_test_pred = model_house.predict(X_test_arr )
y_test_pred = y_test_pred.astype('float64')

mean_absolute_error(y_test_arr, y_test_pred) #with 10 ep : 147736.17041298, with 50 ep:146763.61711159785
np.sqrt(mean_squared_error(y_test_arr, y_test_pred)) #with 10 ep: 240723.12362202845

