#-*- coding: utf-8 -*-
import random

import numpy as np
from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K

from loadData import load_dataset, resize_image, IMAGE_SIZE

class Dataset:
    def __init__(self, path_name):
        self.train_images = None	#train data set
        self.train_labels = None
        self.valid_images = None	#valid data set
        self.valid_labels = None	
        self.test_images  = None	#test data set            
        self.test_labels  = None
        self.path_name    = path_name
        self.input_shape = None

    def load(self, img_rows = IMAGE_SIZE, img_cols = IMAGE_SIZE, img_channels = 3, nb_classes = 5):    #load data and preprocessing 
        images, labels = load_dataset(self.path_name)        
        
        train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size = 0.3, random_state = random.randint(0, 100))        
        _, test_images, _, test_labels = train_test_split(images, labels, test_size = 0.5, random_state = random.randint(0, 100))                

        if K.image_dim_ordering() == 'th':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)            
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)   
                     
            print(train_images.shape[0], 'train samples')
            print(valid_images.shape[0], 'valid samples')
            print(test_images.shape[0], 'test samples')
            
            #use the one hot coding method to vectorlize the labels
            train_labels = np_utils.to_categorical(train_labels, nb_classes)                        
            valid_labels = np_utils.to_categorical(valid_labels, nb_classes)            
            test_labels = np_utils.to_categorical(test_labels, nb_classes)                        
        
            #normlize the image as float format
            train_images = train_images.astype('float32')            
            valid_images = valid_images.astype('float32')
            test_images = test_images.astype('float32')
            
            #normlize the image pixes into (0~1)
            train_images /= 255
            valid_images /= 255
            test_images /= 255            
        
            self.train_images = train_images
            self.valid_images = valid_images
            self.test_images  = test_images
            self.train_labels = train_labels
            self.valid_labels = valid_labels
            self.test_labels  = test_labels

#CNN network model          
class Model:
    def __init__(self):
        self.model = None 
        
    #build model
    def build_model(self, dataset, nb_classes = 5):
        self.model = Sequential() 
        self.model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape = dataset.input_shape))
        self.model.add(Activation('relu'))                                 
        self.model.add(Convolution2D(32, 3, 3))                                                      
        self.model.add(Activation('relu'))                                
        self.model.add(MaxPooling2D(pool_size=(2, 2)))                     
        self.model.add(Dropout(0.25))                                     
        self.model.add(Convolution2D(64, 3, 3, border_mode='same'))         
        self.model.add(Activation('relu'))                                
        self.model.add(Convolution2D(64, 3, 3))                            
        self.model.add(Activation('relu'))                               
        self.model.add(MaxPooling2D(pool_size=(2, 2)))                    
        self.model.add(Dropout(0.25))                                       
        self.model.add(Flatten())                                         
        self.model.add(Dense(512))                                        
        self.model.add(Activation('relu'))                                 
        self.model.add(Dropout(0.5))                                     
        self.model.add(Dense(nb_classes))                            
        self.model.add(Activation('softmax'))                             
        
        #output the model infomation
        self.model.summary()  
    #train model
    def train(self, dataset, batch_size = 16, nb_epoch = 16, data_augmentation = True):        
        sgd = SGD(lr = 0.01, decay = 1e-6, 
                  momentum = 0.9, nesterov = True) 
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])   
                           
        #increse the data set ahd rotate, flip, add noise to create new images 
        if not data_augmentation:            
            self.model.fit(dataset.train_images,
                           dataset.train_labels,
                           batch_size = batch_size,
                           nb_epoch = nb_epoch,
                           validation_data = (dataset.valid_images, dataset.valid_labels),
                           shuffle = True)
        else:            
            datagen = ImageDataGenerator(
                featurewise_center = False,             #set the data center as false(mean = 0)
                samplewise_center  = False,             #set sample center as false(mean = 0)
                featurewise_std_normalization = False,  #normlize the data
                samplewise_std_normalization  = False,  
                zca_whitening = False,                  #set the ZCA whitening as false
                rotation_range = 20,                    #rotate image randomly with 20 degree each time
                width_shift_range  = 0.2,              
                height_shift_range = 0.2,               
                horizontal_flip = True,                 #randomly flip the image horizontally
                vertical_flip = False)                  

            datagen.fit(dataset.train_images)                        

            #train the new generated data
            self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels,
                                                   batch_size = batch_size),
                                     samples_per_epoch = dataset.train_images.shape[0],
                                     nb_epoch = nb_epoch,
                                     validation_data = (dataset.valid_images, dataset.valid_labels))
    #save and load models
    MODEL_PATH = './model.h5'
    def save_model(self, file_path = MODEL_PATH):
        self.model.save(file_path)
        
    def load_model(self, file_path = MODEL_PATH):
        self.model = load_model(file_path)
    
    #evaluate the models   
    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose = 1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))
        
    #predit face 
    def face_predict(self, image):    
        if K.image_dim_ordering() == 'th' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = resize_image(image)                             
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))    
        elif K.image_dim_ordering() == 'tf' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_image(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))                    
        image = image.astype('float32')
        image /= 255
        result = self.model.predict_proba(image)
        print('result:', result)
        result = self.model.predict_classes(image)        
        return result[0]
         
           
if __name__ == '__main__':

    dataset = Dataset('./flower_photos/train')    
    dataset.load()
    model = Model()
    model.build_model(dataset) 
    model.train(dataset)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.save_model(file_path = 'model.h5')       
    model.load_model(file_path = 'model.h5')
    model.evaluate(dataset)                                 
  
