# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

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


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

First I tried to complete the project using the inception module with ImageNet pre-trained weights. 

```python
InceptionV3(weights=weights_flag, include_top=False, input_shape=(139,139,3)) 
```
Two Lambda and one cropping layers were added to the network to preprocess the images. The first Lambda layer was added to normalize the images, the cropping layer - to remove distracting backgroung from the images and the second Lambda layer to resize the images to 139x139. 

The network was loaded without the last fully connected layer and the average pooling layer. I added a pooling layer and connected it to the end of the Inception module. After the added pooling layer, dropout layer was included with a keeping probability of 0.5 followed by three fully connected layers with a linear activation function:

```python
x = GlobalAveragePooling2D()(inp)
x = Dropout(0.5)(x)
x = Dense(256)(x)
x = Dense(50)(x)
x = Dense(10)(x)
predictions = Dense(1)(x)
```

I tried this model with not frozen weights (the model was retrained from the first layer) for 5 epochs. I got a validation accuracy of 0.02. Unfortunately, I was not able to test the network's perfomance on the first track, as I was getting errors with loading the model while running driver.py. 

I also had an intution that for using such a powerful pretrained network I need more data, so I decided to go with a network suggested in the lectures (proposed by Nvidia team). Following is the architecture of the network: ![alt text][image1].


#### 2. Attempts to reduce overfitting in the model

To reduce overfitting I added two dropout layers. The first dropout layer was added with a keeping probability of 0.2 after the last 5x5 convolutional layer, the second - with the probabilty of 0.3 before the first fully-connected layer.  

#### 3. Model parameter tuning

Since we use an adam optimizer, the learning rate was not tuned manually. I decided to go with a batch size of 64, although in the lectures' material the batch size was 32, I doubled it to reduce overfitting. Number of epochs was set to 5. I tried to increase the number of epochs, but it resulted in overfitting, so the optimal value, when training and validation losses decrease monotonically and the vehicle performs well on the first track in the autonomous mode, was 5.

#### 4. Appropriate training data

I used the default training data, which is avaliable in the workspace. Following are the examples of a randomly selected image from the training set within the augmentation (flipped image):

Image from the central camera            |  Flipped image                              |
:---------------------------------------:|:-------------------------------------------:|
![alt text][image3]                      |  ![alt text][image4]


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try the transfer learning concept first with the inception module and then try the architecture developed by the Nvidia team. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that both models I tried had a low mean squared error on the training set but a higher mean squared error on the validation set. This implied that the model was overfitting. So to combat the overfitting, I added dropout layers to the model. After adding dropouts, the loss between training and validation sets was decreasing monotonically and quite close for both sets.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, so to improve the driving behavior in these cases, I increased the keeping probability for the dropout layers from 0.4 to 0.5.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

My final model consisted of the following layers:

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


#### 3. Creation of the Training Set & Training Process

I did not collect data, for the training and validation sets I used default data avaliable in the workspace. I split the data in the ration of 70/30 for training and validation. I used images from all three cameras and augmented data via flipping, so for each row in a csv file I got 6 images.   

I finally randomly shuffled training and validation sets. 

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