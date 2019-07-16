# -*- coding: utf-8 -*-
#
# Last modification: 4 July. 2019
# Author: Rayanne Souza

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import itertools    

from glob import glob
from PIL import Image

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet121
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# sklearn libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from itertools import cycle


np.random.seed(10)


# Normalization of the input image between -1 to 1
def normalize_data(img_list):
 
  x = np.asarray(img_list)
  x =  (x - 127.5) / 127.5

  return x


def build_custom_model(model_name, n_classes):

  base_model = None
  height = 90
  width = 120
  
  if model_name == "vgg":
    base_model = VGG16(weights='imagenet',include_top=False, input_shape=(height, width, 3))  
  
  elif model_name == "inception": 
    base_model = InceptionV3(include_top=False, input_shape=(height, width, 3), weights = 'imagenet')
    
  elif model_name == "densenet":
    base_model = DenseNet121(include_top=False, input_shape=(height, width, 3), weights = 'imagenet')
  
  for layer in base_model.layers:
    layer.trainable = False
    
  for layer in base_model.layers:
    print(layer, layer.trainable)
    
  x = base_model.output
  x = Flatten()(x)
  x = Dropout(.5)(x)
  x = Dense(512, activation='relu')(x)
  predictions = Dense(n_classes, activation='softmax')(x)
  model = Model(base_model.input, predictions)

  return model


def show_result(acc, val_acc, loss, val_loss):

  plt.figure(1)
  plt.plot(loss, label='training loss')
  plt.plot(val_loss, label='validation loss')
  plt.xlabel('epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.savefig('loss.jpg')
 
  plt.figure(2)
  plt.plot(acc, label='training accuracy')
  plt.plot(val_acc, label='validation accuracy')
  plt.xlabel('epoch')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.savefig("acc.jpg")
  

def train(model, x_train, y_train, x_val, y_val, batch_size, data_augmentation = False):
  
  early_stop = EarlyStopping(monitor = 'val_acc', min_delta = 0.001, mode = 'max', patience = 10)
  learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
  Epochs = 50
  
  
  if data_augmentation :
      print("-------------Using Data augmentation------------")
      # This will do realtime data augmentation:
      datagen = ImageDataGenerator(
          fill_mode = "nearest",
          zoom_range = 0.20,
          rotation_range=30,                    
          width_shift_range=0.1,               
          height_shift_range=0.1,              
          horizontal_flip=False,                
          vertical_flip=False,
          zca_whitening=False,
          featurewise_std_normalization=False,
          samplewise_std_normalization=False,
          samplewise_center=False,
          featurewise_center=False
          
          )                 

      datagen.fit(x_train)
  
      history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                          steps_per_epoch=x_train.shape[0]//batch_size,
                          epochs=Epochs,
                          verbose=1,
                          validation_data=[x_val,y_val],
                          callbacks=[early_stop, learning_rate_reduction])
      
  else:
    print("-----Not Using Data augmentation---------------")
    
   
    history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=Epochs,
                    verbose=1,
                    validation_data = [x_val,y_val],
                    #class_weight=d_class_weights, 
                    callbacks=[early_stop])

  
 
    show_result(history.history['acc'], 
                history.history['val_acc'],
                history.history['loss'],
                history.history['val_loss'])


if __name__=='__main__':

 # Reads data from csv file
 path = "data_csv"
 train_df = pd.read_csv(path + '/train_df.csv')
 validation_df = pd.read_csv(path + '/validation_df.csv')
 test_df = pd.read_csv(os.path.join(path + '/test_df.csv'))

 all_image_path = glob(os.path.join('images/', '*', '*.jpg'))
 imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}

 train_df['npath'] = train_df['new_image_id'].map(imageid_path_dict.get)
 validation_df['npath'] = validation_df['new_image_id'].map(imageid_path_dict.get)
 test_df['npath'] = test_df['new_image_id'].map(imageid_path_dict.get)
 

 # Loading image
 train_df['image'] = train_df['npath'].map(lambda x: np.asarray(Image.open(x)))
 validation_df['image'] = validation_df['npath'].map(lambda x: np.asarray(Image.open(x)))
 test_df['image'] = test_df['npath'].map(lambda x: np.asarray(Image.open(x)))
    
 x_train = normalize_data(train_df['image'].tolist())
 x_validate = normalize_data(validation_df['image'].tolist())
 x_test =  normalize_data(test_df['image'].tolist())
 
 # One hot encode 
 y_train = train_df['categorical_label'] 
 y_train = np_utils.to_categorical(y_train, num_classes = 7)
  
 y_validate = validation_df['categorical_label'] 
 y_validate = np_utils.to_categorical(y_validate, num_classes = 7)

 y_test = test_df['categorical_label'] 
 y_test = np_utils.to_categorical(y_test, num_classes = 7)
      
 # Creating custom vgg  
 model = build_custom_model("vgg", 7)
 model.summary()
  
 opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
 model.compile(loss = 'categorical_crossentropy', optimizer=opt , metrics=['accuracy'])
  

 bs = 64 #batch size  
 train(model, x_train, y_train, x_validate, y_validate, bs, False)
 
 # Model Evaluation 
 test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
 val_loss, val_acc = model.evaluate(x_validate, y_validate, verbose=1)
 print("Test_accuracy = %f  ;  Test_loss = %f" % (test_acc, test_loss))
 print("Val_accuracy = %f  ;  Val_loss = %f" % (val_acc, val_loss))

 # Saving the custom-trained VGG16
 model.save('my_model.h5')
 
 # Saving testing set
 np.save(open("x_test.npy" ,"wb"), x_test)
 np.save(open("y_test.npy","wb"), y_test)
 
