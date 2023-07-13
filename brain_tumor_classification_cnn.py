#This program was originally written in Kaggle because of the dataset size and computation power required to run the program.
#This python file is a copy of the original file in Kaggle.

import numpy as np 
import pandas as pd 
# Input data files are available in the read-only "pre./input/" directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import keras
import tensorflow as tf
import cv2 
import tqdm
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,MaxPooling2D,Dropout
from sklearn.metrics import accuracy_score
import ipywidgets as widgets #required for classification
import io #for input n output 
from PIL import Image #Public Image Library
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle #for spliting train and test data
tf.random.set_seed(393) #set our seed for replicable results

"""We will put all images in the training and testing sets into x and y train and split it ourselves"""

X_train = []
Y_train = []
image_size = 150
labels = ['glioma_tumor','no_tumor','meningioma_tumor','pituitary_tumor']
#index 0: glioma, index 1: no_tumor, index 2 = meningioma_tumor, index 3: pituitary_tumor
for i in labels:
    folderPath = os.path.join('../input/brain-tumor-classification-mri/Training',i)
    for j in os.listdir(folderPath):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(image_size,image_size))
        X_train.append(img)
        Y_train.append(i)
        
for i in labels:
    folderPath = os.path.join('../input/brain-tumor-classification-mri/Testing',i)
    for j in os.listdir(folderPath):
        img = cv2.imread(os.path.join(folderPath,j))
        img = cv2.resize(img,(image_size,image_size))
        X_train.append(img)
        Y_train.append(i)
#convert x and y train to arrays of images
X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_train, Y_train = shuffle(X_train, Y_train, random_state=101)

#3264 images in X_train and the images are 150x150 with 3 color channels

X_train,X_test,y_train,y_test = train_test_split(X_train,Y_train,test_size=0.2,random_state=393)

print(X_train.shape)
print(y_test.shape)

y_train_new = []
for i in y_train:
    y_train_new.append(labels.index(i))
y_train=y_train_new
y_train = tf.keras.utils.to_categorical(y_train) #convert label indexes to categorical variables
y_test_new = []
for i in y_test:
    y_test_new.append(labels.index(i))
y_test=y_test_new
y_test = tf.keras.utils.to_categorical(y_test)

"""Convolutional Neural Network"""

#model 1 9 convolutional layers with maxpooling and dropout. kernel/filter size 3,3
model = Sequential()
model.add(Conv2D(64,(3,3),activation = 'relu',input_shape=(150,150,3))) #specify image shape from training set
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2)) #add may pooling/padding 2x2
model.add(Dropout(0.3)) #0.3 dropout

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3)) #0.3 dropout

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3)) #0.3 dropout

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3)) #0.3 dropout

model.add(Flatten()) #flatten before dense layer ouput. converts to 1d vector
model.add(Dense(512,activation = 'relu'))
model.add(Dense(512,activation = 'relu'))
model.add(Dropout(0.3)) #0.3 dropout
model.add(Dense(4,activation='softmax')) #softmax for output layer for multiclass classification
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])

history = model.fit(X_train,y_train,epochs=20,validation_split=0.2)

import matplotlib.pyplot as plt
import seaborn as sns

#plot training and validation accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs=range(len(acc))
fig = plt.figure(figsize=(16, 8))
plt.plot(epochs,acc,'r',label="Training Accuracy")
plt.plot(epochs,val_acc,'b',label="Validation Accuracy")
plt.legend(loc="upper left")
plt.show()

#plots training and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs=range(len(loss))
fig = plt.figure(figsize=(16, 8))
plt.plot(epochs,loss,'r',label="Training loss")
plt.plot(epochs,val_loss,'b',label="Validation loss")
plt.legend(loc="upper left")
plt.show()

from sklearn.metrics import classification_report, confusion_matrix
#make predictions on the test set
pred_test = model.predict(X_test)
pred_test = np.argmax(pred_test,axis=1)
y_test_new = np.argmax(y_test,axis=1)

print(classification_report(y_test_new,pred_test, target_names=['Glioma Tumor', 'No Tumor', 'Meningioma Tumor', 'Pituitary Tumor']))
#index 0: glioma, index 1: no_tumor, index 2 = meningioma_tumor, index 3: pituitary_tumor

#train preds
pred_train = model.predict(X_train)
pred_train = np.argmax(pred_train,axis=1)
y_train_new = np.argmax(y_train,axis=1)

print(classification_report(y_train_new,pred_train, target_names=['Glioma Tumor', 'No Tumor', 'Meningioma Tumor', 'Pituitary Tumor']))
#index 0: glioma, index 1: no_tumor, index 2 = meningioma_tumor, index 3: pituitary_tumor

fig,ax=plt.subplots(1,1,figsize=(14,7))
sns.heatmap(confusion_matrix(y_test_new,pred_test),ax=ax,xticklabels=labels,yticklabels=labels,annot=True,alpha=0.7,linewidths=2, fmt='g')
fig.text(s='Heatmap of the Confusion Matrix',size=18,fontweight='bold',y=0.92,x=0.28,alpha=0.8)

plt.show()
#the results of the confusion matrix show that the model is performing well(most values are in the main diagonal)
# Most of the misclassifications the model is making is classifying it as a glioma_tumor when it is a mningoma tumor. 
#The Result that we most care about however is a false negative and in the confusion matrix, we see that there are xxx 

#false negatives. 
# We define a false negative to be when the network classifies no_tumor when the true value is that there is a tumor(glioma, meningioma, pituary)

#There are xxx false positive in total

img = cv2.imread('/kaggle/input/brain-tumor-classification-mri/Testing/pituitary_tumor/p (56).jpg')
img = cv2.resize(img,(150,150)) #resize image to 150,150
img_array = np.array(img)
img_array.shape

from google.colab import drive
drive.mount('/content/drive')

img_array = img_array.reshape(1, 150, 150, 3)
img_array.shape

from tensorflow.keras.preprocessing import image
img = image.load_img('/kaggle/input/brain-tumor-classification-mri/Testing/pituitary_tumor/image(56).jpg')
plt.imshow(img,interpolation='nearest')
print("This is the image we are attempting to predict: it is in the testing set and it is a Pituitary tumor")
plt.show()

opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
img = cv2.resize(opencvImage,(150,150))
img = img.reshape(1,150,150,3) #change it (150x150) with 3 color channels
p = model.predict(img)
p = np.argmax(p,axis=1)[0] #get index of prediction
#index 0: glioma, index 1: no_tumor, index 2 = meningioma_tumor, index 3: pituitary_tumor
if p==0:
    p='Glioma Tumor'
elif p==1:
    print('The model predicts that there is no tumor')
elif p==2:
    p='Meningioma Tumor'
else:
    p='Pituitary Tumor'
print("The model predicts it is a",p)

img = image.load_img('/kaggle/input/brain-tumor-classification-mri/Testing/no_tumor/image(2).jpg')
plt.imshow(img,interpolation='nearest')
print("This is the image we are attempting to predict: it is in the testing set and it is NOT A TUMOR")
plt.show()

opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
img = cv2.resize(opencvImage,(150,150))
img = img.reshape(1,150,150,3) #change it (150x150) with 3 color channels
p = model.predict(img)
p = np.argmax(p,axis=1)[0] #get index of prediction
#index 0: glioma, index 1: no_tumor, index 2 = meningioma_tumor, index 3: pituitary_tumor

if p==0:
    p='Glioma Tumor'
elif p==1:
    print('The model predicts that there is no tumor')
elif p==2:
    p='Meningioma Tumor'
else:
    p='Pituitary Tumor'
if p!=1:
        print(f'The Model predicts that it is a {p}')
#misclassification(false positive)

img = image.load_img('/kaggle/input/brain-tumor-classification-mri/Testing/meningioma_tumor/image(100).jpg')
plt.imshow(img,interpolation='nearest')
print("This is the image we are attempting to predict: it is in the testing set and it is a Meningioma tumor")
plt.show()

opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
img = cv2.resize(opencvImage,(150,150))
img = img.reshape(1,150,150,3) #change it (150x150) with 3 color channels
p = model.predict(img)
p = np.argmax(p,axis=1)[0] #get index of prediction
#index 0: glioma, index 1: no_tumor, index 2 = meningioma_tumor, index 3: pituitary_tumor
if p==0:
    p='Glioma Tumor'
elif p==1:
    print('The model predicts that there is no tumor')
elif p==2:
    p='Meningioma Tumor'
else:
    p='Pituitary Tumor'
if p!=1:
        print(f'The Model predicts that it is a {p}') #prints result if it is a tumor
#the model correctly predicts the output

img = image.load_img('/kaggle/input/brain-tumor-classification-mri/Testing/glioma_tumor/image(42).jpg')
plt.imshow(img,interpolation='nearest')
print("This is the image we are attempting to predict: it is in the testing set and it is a Glioma tumor")
plt.show()

opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
img = cv2.resize(opencvImage,(150,150))
img = img.reshape(1,150,150,3) #change it (150x150) with 3 color channels
p = model.predict(img)
p = np.argmax(p,axis=1)[0] #get index of prediction
#index 0: glioma, index 1: no_tumor, index 2 = meningioma_tumor, index 3: pituitary_tumor

if p==0:
    p='Glioma Tumor'
elif p==1:
    print('The model predicts that there is no tumor')
elif p==2:
    p='Meningioma Tumor'
else:
    p='Pituitary Tumor'
    
    
if p!=1:
        print(f'The Model predicts that it is a {p}') #prints result if it is a tumor
#the model correctly predicts the output

#model 2: 4 convolutional layers, 2 dense layers, same # of hidden units and same parameters from model 1
#instead of 20 epochs we will use 50 since there are less convolutional layers
model2 = Sequential()
model2.add(Conv2D(64,(3,3),activation = 'relu',input_shape=(150,150,3))) #specify image shape from training set
model2.add(Conv2D(128,(3,3),activation='relu'))
model2.add(MaxPooling2D(2,2)) #add may pooling/padding 2x2
model2.add(Dropout(0.3)) #0.3 dropout

model2.add(Conv2D(128,(3,3),activation='relu'))
model2.add(Conv2D(256,(3,3),activation='relu'))
model2.add(MaxPooling2D(2,2))
model2.add(Dropout(0.3)) #0.3 dropout

model2.add(Flatten()) #flatten before dense layer ouput
model2.add(Dense(512,activation = 'relu'))
model2.add(Dense(512,activation = 'relu'))
model2.add(Dropout(0.3)) #0.3 dropout
model2.add(Dense(4,activation='softmax')) #softmax for output layer
model2.summary()

model2.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"]) 
history = model2.fit(X_train,y_train,epochs=50,validation_split=0.2)

#model 2 plot 1
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs=range(len(acc))
fig = plt.figure(figsize=(16, 8))
plt.plot(epochs,acc,'r',label="Training Accuracy")
plt.plot(epochs,val_acc,'b',label="Validation Accuracy")
plt.legend(loc="upper left")
plt.show()

#model 2 plot 2
#plots training and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs=range(len(loss))
fig = plt.figure(figsize=(16, 8))
plt.plot(epochs,loss,'r',label="Training loss")
plt.plot(epochs,val_loss,'b',label="Validation loss")
plt.legend(loc="upper left")
plt.show()

from sklearn.metrics import classification_report, confusion_matrix
#make predictions on the test set
pred = model2.predict(X_test)
pred = np.argmax(pred,axis=1)
y_test_new = np.argmax(y_test,axis=1)

print(classification_report(y_test_new,pred, target_names=['Glioma Tumor', 'No Tumor', 'Meningioma Tumor', 'Pituitary Tumor']))

fig,ax=plt.subplots(1,1,figsize=(14,7))
sns.heatmap(confusion_matrix(y_test_new,pred),ax=ax,xticklabels=labels,yticklabels=labels,annot=True,alpha=0.7,linewidths=2, fmt='g')
fig.text(s='Heatmap of the Confusion Matrix',size=18,fontweight='bold',y=0.92,x=0.28,alpha=0.8)

plt.show()

#train predictions
pred_train = model2.predict(X_train)
pred = np.argmax(pred_train,axis=1)
y_train_new = np.argmax(y_train,axis=1)
print(classification_report(y_train_new,pred, target_names=['Glioma Tumor', 'No Tumor', 'Meningioma Tumor', 'Pituitary Tumor']))

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from google.colab.patches import cv2_imshow

model = InceptionV3(weights='imagenet')

model.summary()

!wget /kaggle/input/brain-tumor-classification-mri/Testing/pituitary_tumor/image(56).jpg
!wget /kaggle/input/brain-tumor-classification-mri/Testing/no_tumor/image(2).jpg
!wget /kaggle/input/brain-tumor-classification-mri/Testing/meningioma_tumor/image(100).jpg
!wget /kaggle/input/brain-tumor-classification-mri/Testing/glioma_tumor/image(42).jpg

ORIGINAL = '/kaggle/input/brain-tumor-classification-mri/Testing/pituitary_tumor/image(56).jpg'

DIM = 299

img = image.load_img(ORIGINAL, target_size=(DIM, DIM))

cv2_imshow(cv2.imread(ORIGINAL)) # Visualize image

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print(decode_predictions(preds))

with tf.GradientTape() as tape:
  last_conv_layer = model.get_layer('conv2d_93')
  iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
  model_out, last_conv_layer = iterate(x)
  class_out = model_out[:, np.argmax(model_out[0])]
  grads = tape.gradient(class_out, last_conv_layer)
  pooled_grads = K.mean(grads, axis=(0, 1, 2))
  
heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
heatmap = heatmap.reshape((8,8))
plt.matshow(heatmap)
plt.show()

img = cv2.imread(ORIGINAL)

INTENSITY = 0.5

heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

img = heatmap * INTENSITY + img

cv2_imshow(cv2.imread(ORIGINAL))
cv2_imshow(img)

def gradCAM(orig, intensity=0.5, res=250):
  img = image.load_img(orig, target_size=(DIM, DIM))

  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)

  preds = model.predict(x)
  print(decode_predictions(preds)[0][0][1]) # prints the class of image

  with tf.GradientTape() as tape:
    last_conv_layer = model.get_layer('conv2d_93')
    iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
    model_out, last_conv_layer = iterate(x)
    class_out = model_out[:, np.argmax(model_out[0])]
    grads = tape.gradient(class_out, last_conv_layer)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    
  heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
  heatmap = np.maximum(heatmap, 0)
  heatmap /= np.max(heatmap)
  heatmap = heatmap.reshape((8,8))

  img = cv2.imread(orig)

  heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

  heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

  img = heatmap * intensity + img

  cv2_imshow(cv2.resize(cv2.imread(orig), (res, res)))
  cv2_imshow(cv2.resize(img, (res, res)))

gradCAM("/kaggle/input/brain-tumor-classification-mri/Testing/pituitary_tumor/image(56).jpg")
gradCAM("/kaggle/input/brain-tumor-classification-mri/Testing/no_tumor/image(2).jpg")
gradCAM("/kaggle/input/brain-tumor-classification-mri/Testing/meningioma_tumor/image(100).jpg")
gradCAM("/kaggle/input/brain-tumor-classification-mri/Testing/glioma_tumor/image(42).jpg")

#model 3: 2 convolutional layers, 1 dense layers, same # of hidden units and same parameters from model 1
#instead of 50 epochs we will use 100 since there are less convolutional layers
model3 = Sequential()
model3.add(Conv2D(64,(3,3),activation = 'relu',input_shape=(150,150,3))) #specify image shape from training set
model3.add(Conv2D(128,(3,3),activation='relu'))
model3.add(MaxPooling2D(2,2)) #add may pooling/padding 2x2
model3.add(Dropout(0.3)) #0.3 dropout

model3.add(MaxPooling2D(2,2))
model3.add(Dropout(0.3)) #0.3 dropout

model3.add(Flatten()) #flatten before dense layer ouput
model3.add(Dense(512,activation = 'relu'))
model3.add(Dropout(0.3)) #0.3 dropout
model3.add(Dense(4,activation='softmax')) #softmax for output layer
model3.summary()

model3.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"]) 
history = model3.fit(X_train,y_train,epochs=100,validation_split=0.2)

#model 3 plot 1
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs=range(len(acc))
fig = plt.figure(figsize=(16, 8))
plt.plot(epochs,acc,'r',label="Training Accuracy")
plt.plot(epochs,val_acc,'b',label="Validation Accuracy")
plt.legend(loc="upper left")
plt.show()

#model 3 plot 2
#plots training and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs=range(len(loss))
fig = plt.figure(figsize=(16, 8))
plt.plot(epochs,loss,'r',label="Training loss")
plt.plot(epochs,val_loss,'b',label="Validation loss")
plt.legend(loc="upper left")
plt.show()

from sklearn.metrics import classification_report, confusion_matrix
#make predictions on the test set
pred = model3.predict(X_test)
pred = np.argmax(pred,axis=1)
y_test_new = np.argmax(y_test,axis=1)

print(classification_report(y_test_new,pred, target_names=['Glioma Tumor', 'No Tumor', 'Meningioma Tumor', 'Pituitary Tumor']))

fig,ax=plt.subplots(1,1,figsize=(14,7))
sns.heatmap(confusion_matrix(y_test_new,pred),ax=ax,xticklabels=labels,yticklabels=labels,annot=True,alpha=0.7,linewidths=2, fmt='g')
fig.text(s='Heatmap of the Confusion Matrix',size=18,fontweight='bold',y=0.92,x=0.28,alpha=0.8)

plt.show()

#train predictions
pred_train = model3.predict(X_train)
pred = np.argmax(pred_train,axis=1)
y_train_new = np.argmax(y_train,axis=1)
print(classification_report(y_train_new,pred, target_names=['Glioma Tumor', 'No Tumor', 'Meningioma Tumor', 'Pituitary Tumor']))

#model 4: 9 convolutional layers, 2 dense layers, same # of hidden units and same parameters from model 1
#using 50 epochs
model4 = Sequential()
model4.add(Conv2D(64,(3,3),activation = 'relu',input_shape=(150,150,3))) #specify image shape from training set
model4.add(Conv2D(128,(3,3),activation='relu'))
model4.add(MaxPooling2D(2,2)) #add may pooling/padding 2x2
model4.add(Dropout(0.3)) #0.3 dropout

model4.add(Conv2D(64,(3,3),activation='relu'))
model4.add(Conv2D(128,(3,3),activation='relu'))
model4.add(Dropout(0.3))
model4.add(MaxPooling2D(2,2))
model4.add(Dropout(0.3)) #0.3 dropout

model4.add(Conv2D(64,(3,3),activation='relu'))
model4.add(Conv2D(128,(3,3),activation='relu'))
model4.add(Conv2D(128,(3,3),activation='relu'))
model4.add(MaxPooling2D(2,2))
model4.add(Dropout(0.3)) #0.3 dropout

model4.add(Conv2D(128,(3,3),activation='relu'))
model4.add(Conv2D(256,(3,3),activation='relu'))
model4.add(MaxPooling2D(2,2))
model4.add(Dropout(0.3)) #0.3 dropout

model4.add(Flatten()) #flatten before dense layer ouput. converts to 1d vector
model4.add(Dense(512,activation = 'relu'))
model4.add(Dense(512,activation = 'relu'))
model4.add(Dropout(0.3)) #0.3 dropout
model4.add(Dense(4,activation='softmax')) #softmax for output layer for multiclass classification
model4.summary()

model4.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"]) 
history = model4.fit(X_train,y_train,epochs=50,validation_split=0.2)

#model 4 plot 1
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs=range(len(acc))
fig = plt.figure(figsize=(16, 8))
plt.plot(epochs,acc,'r',label="Training Accuracy")
plt.plot(epochs,val_acc,'b',label="Validation Accuracy")
plt.legend(loc="upper left")
plt.show()

#model 4 plot 2
#plots training and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs=range(len(loss))
fig = plt.figure(figsize=(16, 8))
plt.plot(epochs,loss,'r',label="Training loss")
plt.plot(epochs,val_loss,'b',label="Validation loss")
plt.legend(loc="upper left")
plt.show()

from sklearn.metrics import classification_report, confusion_matrix
#make predictions on the test set
pred = model4.predict(X_test)
pred = np.argmax(pred,axis=1)
y_test_new = np.argmax(y_test,axis=1)

print(classification_report(y_test_new,pred, target_names=['Glioma Tumor', 'No Tumor', 'Meningioma Tumor', 'Pituitary Tumor']))

fig,ax=plt.subplots(1,1,figsize=(14,7))
sns.heatmap(confusion_matrix(y_test_new,pred),ax=ax,xticklabels=labels,yticklabels=labels,annot=True,alpha=0.7,linewidths=2, fmt='g')
fig.text(s='Heatmap of the Confusion Matrix',size=18,fontweight='bold',y=0.92,x=0.28,alpha=0.8)

plt.show()

#train predictions
pred_train = model4.predict(X_train)
pred = np.argmax(pred_train,axis=1)
y_train_new = np.argmax(y_train,axis=1)
print(classification_report(y_train_new,pred, target_names=['Glioma Tumor', 'No Tumor', 'Meningioma Tumor', 'Pituitary Tumor']))