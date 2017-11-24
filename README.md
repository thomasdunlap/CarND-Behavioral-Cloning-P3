# **Behavioral Cloning**
![Normal image][normal]
---

### **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[Nvidia_model]: ./examples/Nvidia_model.png "Model Visualization"
[curve_1]: ./examples/stupid_curve1.jpg "Difficult Curve 1"
[curve_2]: ./examples/stupid_curve2.jpg "Difficult Curve 2"
[curve_3]: ./examples/stupid_curve3.jpg "Difficult Curve 3"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[normal]: ./examples/normal.png "Normal Image"
[flipped]: ./examples/flipped.png "Flipped Image"

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
Using the Udacity provided simulator and my [drive.py](https://github.com/thomasdunlap/CarND-Behavioral-Cloning-P3/blob/master/drive.py) file, the car can be driven autonomously around the track by executing

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The [model.py](https://github.com/thomasdunlap/CarND-Behavioral-Cloning-P3/blob/master/model.py) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the suggested Nvidia architecture.

![Nvidia model][Nvidia_model]

My model consists convolutional neural network with 5x5 and 3x3 filter sizes, and depths between 24 and 64 (model.py lines 86-117)

The model includes RELU activations to introduce nonlinearity in each of its convolutions (lines 92-100), and the data is normalized in the model using a Keras Lambda layer (code line 86).

The images are cropped, with 70 pixels removed from the top, and 25 from the bottom, with a Keras Cropping2D layer (line 90).

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 117). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (model.py line 115).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used exclusively images based on trying to stay in the center of the lane, although I was imperfect at staying in the center, which I kept in to help the model learn to correct itself.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to follow the steps given in the Udacity lessons.  I put together what they suggested, and it worked without even including the left or right images, or actively adding correction data.

I had to adjust the speed in the drive.py file (line 47), but other than that I mostly just followed directions.

My first step was to use a convolutional neural network model similar to the Nvidia model presented in the online lectures. I thought this model might be appropriate because it had already been used on in the real world, and that should hopefully translate to more simplified, simulated conditions.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation sets. At first the model was taking a LONG time to run.  This was because I was not using the generator function, which yields (returns) an array in batches (smaller groups) instead of constructing and iterating through the entire data set. I was also initially using the left and right camera images, as well as flipping them, which I will eventually put back into an improved model, but I have to minimum-viable-product things for now, with Term 1 closing soon.  

My model has a low mean squared error on both the training and validation sets, implying it was doing a good job predicting the angles. Again, I was surprised by this, because the Nvidia model is strong enough to work without drop out layers, or the left and right camera angles as long as you augment the center images with their flipped counter-parts.

The final step was to run the simulator to see how well the car was driving around track one. The model worked the first time I tried it at a constant speed of 9 mph.  I then really hit the simulated nitro at by setting my drive.py speed to 30 mph (line 47), which caused the car to veer wildly into a stony ravine.  I then adjusted it to 18 mph, and it almost worked, eventually settling at 15 mph.  Here's the only curve the vehicle had trouble with at 18 mph:

![Curve my model couldn't handle part 1][curve_1] ![Curve my model couldn't handle part 2][curve_2] ![Curve my model couldn't handle part 3][curve_3]

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

| Layer (type)         | Output Shape        | Parameters  | Patch Size      |
| ---------------------|:-------------------:| --------:| ----------------: |
| Lambda         | (None, 160, 320, 3) | 0        |     |
| Cropping2D     | (None, 65, 320, 3)  | 0        |     |
| Convolution2D  | (None, 31, 158, 24) | 1824     | 5x5 |
| Convolution2D  | (None, 14, 77, 36)  | 21636    | 5x5 |
| Convolution2D  | (None, 5, 37, 48)   | 43248    | 5x5 |
| Convolution2D  | (None, 3, 35, 64)   | 27712    | 3x3 |
| Convolution2D  | (None, 1, 33, 64)   | 36928    | 3x3 |
| Flatten        | (None, 2112)        | 0        |     |
| Dense          | (None, 100)         | 211300   |     |
| Dense          | (None, 50)          | 5050     |     |
| Dense          | (None, 10)          | 510      |     |
| Dense          | (None, 1)           | 11       | None |

Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![Center lane image][normal]

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![Center lane image][normal]
![Flipped image][flipped]

After the collection process, I had X number of data points. I then preprocessed this data by ...

I also converted the BGR to RGB for Keras, and used a Lambda layer to normalize the data around an average value of zero.  I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

There are many ways I hope to improve this model in the future.  I'd like to incorporate the left and right images, and data from the second, more difficult track.  It would also be helpful to give the drive.py file a range of speeds to drive at, as opposed to a constant speed.

Also, sometimes the road has shadows that the car seems to avoid slightly, so shadows could be included in some of the augmented images.

Epoch 1/10

6400/6428 [============================>.] - ETA: 0s - loss: 0.0076/home/carnd/anaconda3/envs/carnd-term1/lib/python3.5/site-packages/keras/engine/training.py:1569: UserWarning: Epoch comprised more than `samples_per_epoch` samples, which might affect learning results. Set `samples_per_epoch` correctly to avoid this warning.warnings.warn('Epoch comprised more than '

6464/6428 [==============================] - 54s - loss: 0.0076 - val_loss: 0.0060

Epoch 2/10

6456/6428 [==============================] - 13s - loss: 0.0067 - val_loss: 0.0067

Epoch 3/10

6464/6428 [==============================] - 12s - loss: 0.0054 - val_loss: 0.0060

Epoch 4/10

6456/6428 [==============================] - 12s - loss: 0.0060 - val_loss: 0.0067

Epoch 5/10

6464/6428 [==============================] - 12s - loss: 0.0049 - val_loss: 0.0065

Epoch 6/10

6456/6428 [==============================] - 12s - loss: 0.0055 - val_loss: 0.0055

Epoch 7/10

6464/6428 [==============================] - 12s - loss: 0.0046 - val_loss: 0.0056

Epoch 8/10

6456/6428 [==============================] - 12s - loss: 0.0052 - val_loss: 0.0047

Epoch 9/10

6464/6428 [==============================] - 12s - loss: 0.0044 - val_loss: 0.0048

Epoch 10/10
6456/6428 [==============================] - 12s - loss: 0.0051 - val_loss: 0.0058
