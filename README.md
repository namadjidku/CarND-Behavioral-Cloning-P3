# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[image1]: examples/nvidia-model.png 
[image2]: examples/loss.jpg
[image3]: examples/tr_image.jpg
[image4]: examples/fltr_image.jpg


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

First I tried to complete the project using the inception module with ImageNet pre-trained weights. 

```python
InceptionV3(weights=weights_flag, include_top=False, input_shape=(139,139,3)) 
```
Two Lambda and one cropping layers were added to the network to preprocess the images. The first Lambda layer was added to normalize the images, the cropping layer - to remove distracting backgroung from the images and the second Lambda layer to resize the images to 139x139. 

The network was loaded without the last fully connected layer and the average pooling layer. A pooling layer was added and connected to the end of the Inception module. After the added pooling layer, dropout layer was included with a keeping probability of 0.5 followed by three fully connected layers with a linear activation function:

```python
x = GlobalAveragePooling2D()(inp)
x = Dropout(0.5)(x)
x = Dense(256)(x)
x = Dense(50)(x)
x = Dense(10)(x)
predictions = Dense(1)(x)
```

Training the model from the first layer for 5 epochs resulted in a validation accuracy of 0.02. It was not possible to test the network's perfomance on the first track, as I was getting errors with loading the model while running driver.py. The next solution was to go with a network proposed by a Nvidia team. Following is the architecture of the network: ![alt text][image1].


#### 2. Attempts to reduce overfitting in the model

To reduce overfitting two dropout layers were added. The first dropout layer was added with a keeping probability of 0.2 after the last 5x5 convolutional layer, the second - with the probabilty of 0.3 before the first fully-connected layer.  

#### 3. Model parameter tuning

Since we used an adam optimizer, the learning rate was not tuned manually. The training was done with a batch size of 64 for 5 epochs. These parameters were optimal - training and validation losses were decreasing monotonically and the vehicle was performing well on the first track in the autonomous mode.

#### 4. Appropriate training data

I used the default training data, which was avaliable in the workspace. Following are the examples of a randomly selected image from the training with its augmentation (flipped image):

Image from the central camera            |  Flipped image                              |
:---------------------------------------:|:-------------------------------------------:|
![alt text][image3]                      |  ![alt text][image4]


#### 5. Solution Design Approach

The overall strategy for deriving a model architecture was to try the transfer learning concept first with the inception module and then try the architecture developed by the Nvidia team. 

In order to gauge how well the model was working, images and steering angle data were split into a training and validation set. Both models I tried had a low mean squared error on the training set but a higher mean squared error on the validation set. This implied that the models were overfitting. So to overcome the overfitting, dropout layers were added to the second model. After adding dropouts, the loss between training and validation sets was decreasing monotonically and quite close for both sets.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, so to improve the driving behavior in these cases, the keeping probability for the dropout layers was increased from 0.4 to 0.5.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 6. Final Model Architecture

The final model architecture:

| Layer         		      |     Output Shape 	|  Params            |
|:---------------------------:|:-------------------:|:------------------:|
| cropping2d_1 (Cropping2D)   | (None, 90, 320, 3)  |  0    			 | 
| lambda_1 (Lambda)      	  | (None, 90, 320, 3)  |  0                 |
| conv2d_1 (Conv2D) 		  |	(None, 43, 158, 24) | 1824               |
| conv2d_2 (Conv2D) 	      | (None, 20, 77, 36)  | 21636              |
| conv2d_3 (Conv2D)  	      | (None, 8, 37, 48)   | 43248              |
| dropout_1 (Dropout) 	      | (None, 8, 37, 48)   |  0                 |
| conv2d_4 (Conv2D) 	      | (None, 6, 35, 64)   | 27712  			 |
| conv2d_5 (Conv2D)           | (None, 4, 33, 64)   | 36928              |
| flatten_1 (Flatten)         | (None, 8448)        |  0                 |  
| dropout_2 (Dropout)		  |	(None, 8448)		|  0				 |
| dense_1 (Dense)             | (None, 100)         | 844900             |
| dense_2 (Dense)             | (None, 50)          | 5050               |
| dense_3 (Dense) 			  |	(None, 10)			| 510			     |


Total params: 981,819
Trainable params: 981,819
Non-trainable params: 0


#### 7. Creation of the Training Set & Training Process

The data was plit in the ration of 70/30 for training and validation. Images from all three cameras were used and augmented via flipping, so for each row in a csv file we got 6 images.   

Finally both training and validation sets were randomly shuffled.

Training and validation loss are plotted on the graphic below:

![alt text][image2]

Epoch 1/5
88/88 [==============================] - 86s 975ms/step - loss: 0.0232 - val_loss: 0.0179

Epoch 2/5
88/88 [==============================] - 78s 881ms/step - loss: 0.0175 - val_loss: 0.0158

Epoch 3/5
88/88 [==============================] - 78s 885ms/step - loss: 0.0161 - val_loss: 0.0155

Epoch 4/5
88/88 [==============================] - 78s 886ms/step - loss: 0.0153 - val_loss: 0.0147

Epoch 5/5
88/88 [==============================] - 78s 885ms/step - loss: 0.0145 - val_loss: 0.0143
