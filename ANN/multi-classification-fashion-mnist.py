from tensorflow.keras.datasets import fashion_mnist
(X_train,y_train),(X_test,y_test)= fashion_mnist.load_data()

X_train.shape
# len(X_train)
# len(X_test)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

plt.imshow(X_train[11]);

#Normalize
X_train= (X_train/255.0)
X_test=  (X_test/255.0)

np.unique(y_train)

from tensorflow.keras.utils import to_categorical
number_of_classes= len(np.unique(y_train))
y_train= to_categorical(y_train,number_of_classes)
y_test= to_categorical(y_test,number_of_classes)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model= Sequential()

X_train= X_train.reshape(60000,784)
X_test= X_test.reshape(10000,784)

X_test.shape

model.add(Dense(units=256,activation='relu',input_dim=784))
model.add(Dense(units=256,activation='relu'))
model.add(Dense(units=256,activation='relu'))
model.add(Dense(units=256,activation='relu'))
model.add(Dense(units=256,activation='relu'))
model.add(Dense(units=10,activation='softmax'))
model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=(['accuracy']))

history= model.fit(X_train,y_train,epochs=50)

X_train[[21]]

model.predict(X_train[[221]])
y_train[221]
