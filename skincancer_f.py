#!/usr/bin/env python
# coding: utf-8

# In[2]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, sys
from glob import glob
import seaborn as sns
from PIL import Image


# In[3]:


path = "/Users/Toshita Sharma/Desktop/data/"
dirs = os.listdir(path)
for file in dirs:
   print (file)


# In[4]:


folder_benign_train = 'C:/Users/Toshita Sharma/Desktop/data/test/benign'
folder_malignant_train = 'C:/Users/Toshita Sharma/Desktop/data/train/malignant'

folder_benign_test = 'C:/Users/Toshita Sharma/Desktop/data/test/benign'
folder_malignant_test = 'C:/Users/Toshita Sharma/Desktop/data/test/malignant'

read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))

# Load in training pictures 
ims_benign = [read(os.path.join(folder_benign_train, filename)) for filename in os.listdir(folder_benign_train)]
X_benign = np.array(ims_benign, dtype='uint8')
ims_malignant = [read(os.path.join(folder_malignant_train, filename)) for filename in os.listdir(folder_malignant_train)]
X_malignant = np.array(ims_malignant, dtype='uint8')

# Load in testing pictures
ims_benign = [read(os.path.join(folder_benign_test, filename)) for filename in os.listdir(folder_benign_test)]
X_benign_test = np.array(ims_benign, dtype='uint8')
ims_malignant = [read(os.path.join(folder_malignant_test, filename)) for filename in os.listdir(folder_malignant_test)]
X_malignant_test = np.array(ims_malignant, dtype='uint8')

# Create labels
y_benign = np.zeros(X_benign.shape[0])
y_malignant = np.ones(X_malignant.shape[0])

y_benign_test = np.zeros(X_benign_test.shape[0])
y_malignant_test = np.ones(X_malignant_test.shape[0])


# Merge data 
X_train = np.concatenate((X_benign, X_malignant), axis = 0)
y_train = np.concatenate((y_benign, y_malignant), axis = 0)

X_test = np.concatenate((X_benign_test, X_malignant_test), axis = 0)
y_test = np.concatenate((y_benign_test, y_malignant_test), axis = 0)

s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s]
y_train = y_train[s]

s = np.arange(X_test.shape[0])
np.random.shuffle(s)
X_test = X_test[s]
y_test = y_test[s]


# In[5]:


w=40
h=30
fig=plt.figure(figsize=(12, 8))
columns = 5
rows = 3

for i in range(1, columns*rows +1):
    ax = fig.add_subplot(rows, columns, i)
    if y_train[i] == 0:
        ax.title.set_text('Benign')
    else:
        ax.title.set_text('Malignant')
    plt.imshow(X_train[i], interpolation='nearest')
plt.show()


# In[6]:


plt.bar(0, y_train[np.where(y_train == 0)].shape[0], label = 'benign')
plt.bar(1, y_train[np.where(y_train == 1)].shape[0], label = 'malignant')
plt.legend()
plt.title("Training Data")
plt.show()

plt.bar(0, y_test[np.where(y_test == 0)].shape[0], label = 'benign')
plt.bar(1, y_test[np.where(y_test == 1)].shape[0], label = 'malignant')
plt.legend()
plt.title("Test Data")
plt.show()


# In[7]:


X_train = X_train/255.
X_test = X_test/255.


# In[8]:


from sklearn.svm import SVC

model = SVC()

model.fit(X_train.reshape(X_train.shape[0],-1), y_train)


# In[9]:


from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test.reshape(X_test.shape[0],-1))

print(accuracy_score(y_test, y_pred))


# In[14]:


from sklearn.metrics import confusion_matrix
 
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[15]:


plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Negative','Positive']
plt.title('Versicolor or Not Versicolor Confusion Matrix - Test Data')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()


# In[26]:


from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
import itertools

# Helper libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import Sequence
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import PIL
print(tf.__version__)
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[27]:


train = 'C:/Users/Toshita Sharma/Desktop/train'
test = 'C:/Users/Toshita Sharma/Desktop/test'
train_dir = os.path.join(train)
validation_dir = os.path.join(test)

train_ben_dir = os.path.join(train_dir, 'benign')  # directory with our training benign pictures
train_mal_dir = os.path.join(train_dir, 'malignant')  # directory with our training malignant pictures
validation_ben_dir = os.path.join(validation_dir, 'benign')  # directory with our validation benign pictures
validation_mal_dir = os.path.join(validation_dir, 'malignant')  # directory with our validation malignant pictures


# In[30]:


num_ben_tr = len(os.listdir(train_ben_dir))
num_mal_tr = len(os.listdir(train_mal_dir))

num_ben_val = len(os.listdir(validation_ben_dir))
num_mal_val = len(os.listdir(validation_mal_dir))

total_train = num_ben_tr + num_mal_tr
total_val = num_ben_val + num_mal_val


# In[31]:


epochs = 100
batch_size = 58
IMG_HEIGHT=112
IMG_WIDTH=112

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )
train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='binary')
#display augmented images
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)
image_gen_val = ImageDataGenerator(rescale=1./255)
val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                 directory=validation_dir,
                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                 class_mode='binary')


# In[32]:


model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', 
           input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])

#loss function shows how accurate our model is
#optimizer updates the model based on loss function and data it sees
#metrics - monitors training and testing steps (this example uses accuracy metric)
model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
model.summary()
#fit the model to the training data
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)
model.summary()


# In[33]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = epochs
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[34]:


model.evaluate(val_data_gen)


# In[57]:

def prediction(input_msg):
    filename = os.path.join(validation_dir, 'input_msg')
    
    img = load_img(filename)
    
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), resample=PIL.Image.BICUBIC)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr /= 255
    prediction = (model.predict(arr) > 0.5).astype("int32")
    pred = prediction[0]
    print(('This skin mole is malignant' if pred >0 else 'This skin mole is benign'))
    return pred


# In[72]:
from keras.models import save_model
model = save_model(model,'C:/Users/Toshita Sharma/Desktop/model.h5')
  

# In[ ]:
