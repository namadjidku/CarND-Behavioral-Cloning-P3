import os
import csv
from scipy import ndimage
import matplotlib.pyplot as plt
from keras.layers import Input, Lambda, Cropping2D, Dropout, Flatten
from keras.models import Sequential
import tensorflow as tf
import cv2
import numpy as np
import random
import sklearn
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D, Convolution2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Model
from math import ceil
#####################################

#tf.get_variable_scope().reuse_variables()

samples = []

with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


train_samples, validation_samples = train_test_split(samples, test_size=0.3)

# randomly select one image from train samples
index = random.randint(0, len(train_samples))
image = ndimage.imread('/opt/carnd_p3/data/IMG/'+train_samples[index][0].split('/')[-1])
cv2.imwrite('examples/tr_image.jpg', image) 
cv2.imwrite('examples/fltr_image.jpg', np.fliplr(image))


# Set our params
batch_size=64
width = 320
height = 160
cropy_top = 50
cropy_bottom = 20
ch = 3
input_size = 139 # 299
freeze_flag = False  # `True` to freeze layers, `False` for full training
weights_flag = 'imagenet' # 'imagenet' or None
#preprocess_flag= True # Should be true for ImageNet pre-trained typically

'''
# Using Incept ion with ImageNet pre-trained weights
inception = InceptionV3(weights=weights_flag, include_top=False, input_shape=(input_size,input_size,ch))

if freeze_flag == True:
    for layer in inception.layers:
        layer.trainable = False
        
        
inception.summary()

n_input = Input(shape=(height, width, ch))

# Preprocess incoming data, centered around zero with small standard deviation 
normalized_input = Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(height, width, ch)) (n_input)

# set up cropping2D layer
cropped_input = Cropping2D(cropping=((cropy_top,cropy_bottom), (0,0)), input_shape=(height, width, ch)) (normalized_input)

# Re-sizes the input with Kera's Lambda layer & attach to cifar_input
resized_input = Lambda(lambda x: tf.image.resize_images(x, (input_size, input_size))) (cropped_input)

# Feeds the re-sized input into Inception model
inp = inception(resized_input)

#  Global Average Pooling
x = GlobalAveragePooling2D()(inp)
x = Dropout(0.5)(x)
x = Dense(256)(x)
x = Dense(50)(x)
x = Dense(10)(x)
predictions = Dense(1)(x)

model = Model(inputs=n_input, outputs=predictions)
'''

model = Sequential()

# cropping
model.add(Cropping2D(cropping=((cropy_top, cropy_bottom), (0,0)), input_shape=(height,width,ch)))

# normalizing
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

# conv 5x5 with stride (2,2)
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Dropout(0.2))

# conv 3x3 with stride (1,1)
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dropout(0.3))

# fully connected layers
model.add(Dense(100))
model.add(Dense(50))
#model.add(Dropout(0.1))
model.add(Dense(10))
model.add(Dense(1))
          
model.compile(loss='mse', optimizer='adam')

# Check the summary of this new model to confirm the architecture
model.summary()


def generator(samples, batch_size=32):
    num_samples = len(samples)
    path = '/opt/carnd_p3/data'
    
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            correction = 0.2 # this is a parameter to tune

            images = []
            angles = []
            for batch_sample in batch_samples:
                try:
                    center_image = ndimage.imread(path + '/IMG/'+batch_sample[0].split('/')[-1])
                    left_image = ndimage.imread(path + '/IMG/'+batch_sample[1].split('/')[-1])
                    right_image = ndimage.imread(path + '/IMG/'+batch_sample[2].split('/')[-1])

                    center_angle = float(batch_sample[3])
                    left_angle = center_angle + correction
                    right_angle = center_angle - correction

                    images.extend([center_image, left_image, right_image])
                    angles.extend([center_angle, left_angle, right_angle])

                    # flipping
                    images.extend( [np.fliplr(center_image), 
                                   np.fliplr(left_image), 
                                   np.fliplr(right_image)])
                    angles.extend([-center_angle, -left_angle, -right_angle])
                    
                except FileNotFoundError:
                    continue

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

           
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

history_object = model.fit_generator(train_generator, 
            steps_per_epoch=ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=ceil(len(validation_samples)/batch_size), 
            epochs=5, verbose=1)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig("loss.jpg")