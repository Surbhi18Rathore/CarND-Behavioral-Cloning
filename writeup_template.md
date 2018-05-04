
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


[//]: # (Image References)

[image1]: ./examples/model_visulization.png "Model Visualization"
[image2]: ./examples/center.png "center"
[image3]: ./examples/model.png "Validation and Training loss curve"
[image4]: ./examples/2.jpg "left"
[image5]: ./examples/1.jpg "center"
[image6]: ./examples/3.jpg "right"

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

My model consists of a 2 convolution neural network layers with 3x3 filter sizes and depths between  16 and 32 (model.py code cell 5, lines 9-17) 

The model includes RELU layers to introduce nonlinearity (code line 11,16), and the data is normalized in the model using a Keras lambda layer (code cell 5 line 7). 

The model includes MaxPooling layer to downsize the images with filter size of 2*2 (code sell 5 line 12 and 17).

One flatten and 3 dense layer were included in model.

#### 2. Attempts to reduce overfitting in the model
 
The model was trained and validated on different data sets to ensure that the model was not overfitting (code cell 2 line 7). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually ( model.py cell 5 line 35).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road .

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to

My first step was to use a convolution neural network model similar to the LeNet architecture. I thought this model might be appropriate because its good in feature detection and good for lane line detection also.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that model can be generalized for any kind of data ...

Then I augmented my data set by using center, left and right camera images and flipped of these images to avoid the biasing of model towards a perticular direction... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track towards left side. to improve the driving behavior in these cases, I augment the training and validation dataset and added more fine convolution layers....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py code cell 5 lines 8-23) consisted of a convolution neural network with the following layers and layer sizes .
 1. Convolution layer 1
    filter size= 3*3 
    Depth = 16
 
 2. Convolution layer 2
    filter size= 3*3 
    Depth = 32
 
  
Here is a visualization of the architecture 

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

i used Udacity provided data for training and validation:
![alt text][image4]
![alt text][image5]
![alt text][image6]

To augment the data sat, I also flipped images and angles thinking that this would help model to be unbiased.

After the collection process, I had 38568 data points. I then preprocessed this data by cropping and resizing the images data.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3.I used an adam optimizer so that manually training the learning rate wasn't necessary .

![alt text][image3]