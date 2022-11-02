#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install imutils')


# In[2]:


get_ipython().system('pip install opencv-python')


# In[5]:


import numpy as np
import urllib
import pandas as pd
import seaborn as sns
import random ,os ,glob
import matplotlib.pyplot as plt
import cv2
from imutils import paths
import itertools
from sklearn.utils import shuffle
from urllib.request import urlopen
import warnings
from sklearn.metrics import confusion_matrix,classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense,Dropout,SpatialDropout2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img,array_to_img


# In[6]:


import os


# In[7]:


working_directory=os.getcwd()


# In[8]:


print(working_directory)


# In[9]:


dir_path=working_directory+'\garbage\Garbage classification\Garbage classification'


# In[10]:


target_size=(224,224)
waste_labels={'cardboard':0,'glass':1,'metal':2,'paper':3,'plastic':4,'trash':5}


# In[11]:


def load_datasets(path):
    x=[]
    labels=[]
    image_paths=sorted(list(paths.list_images(path)))
    for image_path in image_paths:
        img=cv2.imread(image_path)
        img=cv2.resize(img,target_size)
        x.append(img)
        label=image_path.split(os.path.sep)[-2]
        labels.append(waste_labels[label])
    x,labels=shuffle(x,labels,random_state=42)
    print(f"X Botuyu: {np.array(x).shape}")
    print(f"Label Sınıf Sayısı{len(np.unique(labels))} Gözlem Sayısı:{len(labels)}")
    return x , labels


# In[12]:


x, labels=load_datasets(dir_path)


# In[13]:


input_shape=(np.array(x[0]).shape[1],np.array(x[0]).shape[1],3)
print(input_shape)


# In[14]:


def visualize_img(image_batch,label_batch):
    plt.figure(figsize=(10,10))
    for n in range(10):
        ax=plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        plt.title(np.array(list(waste_labels.keys()))[to_categorical(labels,num_classes=6)[n]==1][0].title())
        plt.axis("off")


# In[15]:


visualize_img(x,labels)


# In[16]:


train= ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.1
)

test = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1
)


# In[17]:


train_generator = train.flow_from_directory(
    directory=dir_path,
    target_size=(target_size),
    class_mode='categorical',
    subset='training')

validation_generator = test.flow_from_directory(
    directory=dir_path,
    target_size=(target_size),
    class_mode='categorical',
    subset='validation')


# In[18]:


model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3),padding='same', activation='relu', input_shape=(input_shape)))
model.add(MaxPooling2D(pool_size=2,strides=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2,strides=(2,2)))
    
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2,strides=(2,2)))
    
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2,strides=(2,2)))

model.add(Flatten()),

model.add(Dense(units=64, activation='relu'))
model.add(Dropout(rate=0.2))

model.add(Dense(units=32, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(units=6, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),'acc'])

model.summary()


# In[19]:


callbacks=[EarlyStopping(monitor='val_loss',patience=50,verbose=1, mode='min'),
        ModelCheckpoint(filepath='mymodel.h5',monitor='val_loss',mode='min',save_best_only=True,save_weights_only=False,verbose=1)]


# In[20]:


history=model.fit_generator(train_generator, epochs=100, callbacks=callbacks,validation_data=validation_generator,workers=4,steps_per_epoch=2276//32,validation_steps=251//32)


# In[21]:


plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
plt.plot(history.history["acc"],color="b",label="training accuracy")
plt.plot(history.history["val_acc"],color="b",label="validation accuracy")
plt.legend(loc="lower right")
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Accuracy",fontsize=16)
plt.ylim([min(plt.ylim()),1])

plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
plt.plot(history.history["loss"],color="b",label="training loss")
plt.plot(history.history["val_loss"],color="b",label="validation loss")
plt.legend(loc="upper right")
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Loss",fontsize=16)
plt.ylim([0,max(plt.ylim())])

plt.show()


# In[28]:


loss,precision,recall,acc=model.evaluate(train_generator,batch_size=32)
print(100.0*acc)
print(100.0*loss)
print(100.0*precision)
print(100.0*recall)


# In[29]:


loss,precision,recall,acc=model.evaluate(validation_generator,batch_size=32)
print(100.0*acc)
print(100.0*loss)
print(100.0*precision)
print(100.0*recall)


# In[31]:


x_test,y_test=validation_generator.next()
y_pred=model.predict(x_test)
y_pred=np.argmax(y_pred,axis=1)
y_test=np.argmax(y_test,axis=1)


# In[32]:


target_names=list(waste_labels.keys())


# In[33]:


print(classification_report(y_test,y_pred,target_names=target_names))


# In[42]:


cm=confusion_matrix(y_test,y_pred)

def plot_confusion_matrix(cm,classes,normalize=False,title="confusion matrix",cmap=plt.cm.Blues):
    if normalize:
        cm=cm.astype("float")/cm.sum(axis=1)[:,np.newaxis]
    plt.figure(figsize=(8,6))
    plt.imshow(cm,interpolatioon="nearest",cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)
    fmt='.2f' if normalize else 'd'
    thresh=cm.max() / 2.
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j],fmt),
                horizontalalignment="center",
                color="white" if cm [i,j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel("true label" , fontweight="bold")
    plt.xlabel("predicted label" , fontweight="bold")


# In[ ]:


plot_confusion_matrix(cm,waste_labels.keys(),title="confusion matrix",cmap=plt.cm.OrRd)

