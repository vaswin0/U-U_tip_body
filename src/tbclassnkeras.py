#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('gdown --id 1CLqld838ql2yNPVahJYTSdP-XHpYB7JB')
get_ipython().system('gdown --id 1qJkIEp1U3T9oNqhvKcuz_q2ZHgrzOxSG')


# In[2]:


import numpy as np

from keras import layers
from keras.models import Model
import keras

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from keras import layers, models


# In[3]:


bb = np.load('/kaggle/working/b_phipt.npy').reshape((-1, 64, 64, 1))
tt = np.load('/kaggle/working/t_phipt.npy').reshape((-1, 64, 64, 1))


# In[4]:


X = np.append(bb, tt, axis=0)
X_log  = np.log1p(X)
X_sqrt = np.sqrt(X)
X_norm =  X / (np.sum(X, axis=(1, 2), keepdims=True) + 1e-8)  # +epsilon to avoid div-by-zero for empty events


Y = np.append(np.ones(len(bb)), np.zeros(len(tt))).reshape((-1, 1))


# In[5]:


X_inputs = {
    'Raw':  X,
    'Sqrt': X_sqrt,
    'Log':  X_log,
    'L1':   X_norm
}


# In[5]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)


# In[6]:


#  Build and Compile the CNN

input_img = layers.Input(shape=(64, 64, 1))

x = layers.Conv2D(8, (3,3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2,2), padding='same')(x)

x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((3,3), strides=(2,2), padding='valid')(x)

x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((3,3), strides=(1,1), padding='valid')(x)

x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2,2), strides=(2,2), padding='valid')(x)

x = layers.Conv2D(128, (3,3), strides=(2,2), activation='relu', padding='valid')(x)

x = layers.Flatten()(x)
x = layers.Dense(2048, activation='relu')(x)
x = layers.Dense(512, activation='relu')(x)
output = layers.Dense(2, activation='softmax')(x)

model = models.Model(inputs=input_img, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


# In[7]:


for name, X in X_inputs.items():
    
    print(f"\nTraining with {name}")

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)

    model = models.Model(inputs=input_img, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    #Train the Model
    history = model.fit(X_train, Y_train,epochs=30, 
                    batch_size=32, validation_data=(X_test, Y_test),verbose=0)

    test_loss, test_acc = model.evaluate(X_test, Y_test)
    print(f"{name}:Test Accuracy: {test_acc:.4f}")
    

