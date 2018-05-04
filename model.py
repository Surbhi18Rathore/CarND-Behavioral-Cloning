
# coding: utf-8

# In[1]:


import csv
import cv2
import os
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers import Flatten,Dense, Lambda
from keras.models import Sequential
from keras.layers.core import  Activation, Dropout
from keras.layers.convolutional import Convolution2D
from sklearn.model_selection import train_test_split
from keras.layers.pooling import MaxPooling2D


# In[2]:


lines=[]
with open('./data/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)

train_data, validation_data = train_test_split(lines, test_size=0.2)


# In[3]:


def preprocessing(image):
    cropped=image[70:140,:,:]
    resize_img=cv2.resize(cropped,(64,32))
    return resize_img
    
    


# In[4]:


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            steering = []
            for batch_sample in batch_samples:
                center_name = './data/IMG/'+batch_sample[0].split('/')[-1]
                left_name= './data/IMG/'+batch_sample[1].split('/')[-1]
                rgt_name= './data/IMG/'+batch_sample[2].split('/')[-1]
                
                center_img = preprocessing(cv2.imread(center_name))
                left_img= preprocessing(cv2.imread(left_name))
                rgt_img=preprocessing(cv2.imread(rgt_name))
                
                flip_center_img=np.fliplr(center_img)
                flip_left_img=np.fliplr(left_img)
                flip_rgt_img=np.fliplr(rgt_img)
                
                center_steering = float(batch_sample[3])
                left_steering = float(batch_sample[3])+0.3
                right_steering = float(batch_sample[3])-0.3
                
                flip_center_steering =center_steering*(-1.0)
                flip_left_steering = left_steering*(-1.0)
                flip_right_steering = right_steering*(-1.0)
                
                
                
                images.extend([center_img,left_img,rgt_img,flip_center_img,flip_left_img,flip_rgt_img])
                
                steering.extend([center_steering,left_steering,right_steering,flip_center_steering ,flip_left_steering,flip_right_steering])

           
            X_train = np.array(images)
            y_train = np.array(steering)
            yield (X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_data, batch_size=32)
validation_generator = generator(validation_data, batch_size=32)


# In[5]:




#Sequential Model
model=Sequential()

#Normalization of Data
model.add(Lambda(lambda x:x/255.0-0.5, input_shape=(32,64,3)))

#convolution Layer 1
model.add(Convolution2D(16, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

#convolution Layer 2
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))


#flatten Layer
model.add(Flatten())

model.add(Dense(128))

model.add(Dense(64))

model.add(Dense(1))


model.compile(loss='mse',optimizer='adam')

history_object = model.fit_generator(train_generator, samples_per_epoch= 6*len(train_data), validation_data=validation_generator,nb_val_samples=len(validation_data), nb_epoch=2)

model.save('model.h5')
print('done')


# In[6]:


import matplotlib.pyplot as plt
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


