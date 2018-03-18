
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives car around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/steering_angle_b4.png "Steering angles distribution - Before drop"
[image2]: ./images/steering_angle_after.png "Steering angles distribution - After drop"
[image3]: ./images/model_stats.png "Model Statistics"
[image4]: ./images/model_viz.png "Model Visualization"
[image5]: ./images/center_image.png "Center Image"
[image6]: ./images/recovery_images.png "Recovery Image"
[image7]: ./images/flip_augmented.png "Flipped Image"
[image8]: ./images/brightness_augmented.png "Brightness augmented Image"
[an AirSim tutorial]: <https://github.com/Microsoft/AutonomousDrivingCookbook/tree/02561865cb7bacedc461204bb6fde55b1c607cb9/AirSimE2EDeepLearning>
[Blog]: <https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9>

#### [Rubric Points](https://review.udacity.com/#!/rubrics/432/view)  

---
### Files

The project includes the following files:
* `model.py` for creating and training the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` saved trained convolution neural network
* `video_track1.mp4` the video recording for track one
* `video_track2.mp4` the video recording for track two

Using the Udacity provided simulator and the modified drive.py file, the car can drive autonomously around both tracks by executing
```sh
python drive.py model.h5
```

### Model and Training Data

#### 1. Model architecture

The model is a convolution neural network of depths 16, 16, 32 and filter sizes of 3x3 (`model.py` lines 197-210). Data is normalized within the model using a Keras Lambda layer (line 196).
To reduce overfitting, dropout layers were applied after each of the fully-connected layers (lines 210, 217, 219).

#### 2. Parameter tuning

The model used an Nadam optimizer, therefore the learning rate was not tuned manually (model.py line 222, 224). Per Keras doc (https://keras.io/optimizers/), "Nadam is Adam RMSprop with Nesterov momentum."

#### 3. Training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of:
* center lane driving
* recovering from the left and right sides of the road
* opposite direction lane driving, and
* curves emphasizing driving

For track two, driving within right lane was strived for as much as I could.

The next section details how I created the training data.


### Solution Design Approach

#### 1. Model and training data

The overall strategy for deriving a model architecture was to be simple yet effective, so car driving in autonomous mode is always on track and the driving is as smooth as possible.

My first step was to use a convolution neural network model of two convolution layers and one fully-connected layer. I thought this model might be appropriate because the input images are of similar patterns, and furthermore, the provided sample dataset is not enough for training a very deep network.

I also split the log data into a training and a validation sets and make sure the training loss and validation loss have a declined trend. In fact, the losses were decreasing, and validation loss had been always less than training loss because of the dropout layers and also of the training loss calculation is batch accumulated - high losses in earlier batches had contributed to the overall higher loss.

There were a few things to consider:
* Include images from left and right camera? If yes, how much steering angle adjustment to make for each camera? will a constant be appropriate?
* Drop zero-steering samples? If yes, at what percentage?
* Augmented data for training? how?
* How many training epochs? It seemed that more epochs does not guarantee a better performance.

After reading through some posts in the discussion forum and in the slack channel, I chose to include the images from left and right cameras with an angle correction 0.2 and dropped half of the zero-angle images. In addition, adding augmented data via changing brightness of the original images and flipping original images with 50% probability.

The first model trained on the provided sample data didn't work out well. The autonomous car easily fell off the track, especially at the sharp left curve after the bridge. Examining the video frames, I found the curve was not in training set, no wonder the car had a tendency to derail since the right border is different from all other borders in the sample data. I need to record more data, especially the data for this curve.

At least two laps of data on both directions are recorded, as well as data from recovery and data from curves.
I also grid-searched the hyperparameters of zero-angle drop rate, steering angle correct amount, and number of training epochs. The results had settled the zero-angle drop rate to 0.9 and steering angle correction 0.2. As for number of epochs, 20 seemed to be good at this stage. Later on, when incorporating Keras EarlyStopping callback, I was able to set a large number and let the callback decides when to stop. However, the lowest validation loss does not always yields a better result. I needed to spot check on models saved in other epochs and run the simulator to actually see how well the model perform. To me, this is the most tedious part.

At the end of the process, the vehicle is able to drive autonomously around the track one without leaving the road.

To validate the chosen value (0.9) for zero-angle drop rate, the plots below shows that the aggressive removing zero-angle samples with rate 0.9 gives a more balanced dataset. __Note__ that the dataset also include the data recorded from track two.

Original data distribution
![alt text][image1]

Zero-angle drop rate = 0.9
![alt text][image2]

The data collection process was repeated on track two. Two laps of data were collected, as well as recovery data and data from curves. After several trial and error , I decided to increase the network depth. So now the network includes three convolution layers and two fully-connected layers.  

I also added one Input layer of the previous state (by 'state' it means steering angle, throttle and speed) to be combined with the output of the last convolution layer. The combined feature maps and new Input layer are then fed into the fully-connected layers.

There are a few findings with the changes:
* By adding data from track two, the quality of driving on track one is improved.
* By adding data from track one and increasing network depth, the successful driving on track two becomes possible. Missing one of them, the driving on track two was easily fell off the track or bumping into objects.
* By adding previous state as input, the quality of driving is improved substantially. Even the speed of the recorded data fluctuates from time to time and is very low (e.g., less than 5 mph) at some sharp curves and steep slopes, autonomous car is able to smoothing those discomforts out and drive at the speed limit of 15 mph.
* The handling of the sharp left turn after the bridge in track one becomes a little worse at some point during the tuning process for track two. Perhaps the large portion of track two data has make this spot (which has very different borders) less learned. This could possibly be fixed by adding more data of this curve.

__Note__
1. `drive.py` was modified due to the adding of the extra Input layer.
2. New csv files including the previous state column have to be created before running `model.py` (using `preprocess_logs.py`).


#### 2. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps of both directions (clockwise and counterclockwise) on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image5]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct itself once it drifts toward the sides. These images show what a recovery looks like starting from left to right and top to down:

![alt text][image6]

To augment the data sat, I also flipped images and/or applied random brightness. For example, here is an image that has then been flipped and an image that has been applied random brightness:

Flip
![alt text][image7]

Brightness
![alt text][image8]


After the collection process, I had 31,754 number of data points (i.e., logs). I then preprocessed this data by cropping 30 pixels from the top and 20 pixels form the bottom, and normalized to have a zero mean and a very small variance.


I finally randomly shuffled the data set and put aside 20% of it to be a validation set.
The validation set helped determine if the model was over or under fitting. The ideal number of epochs returned by EarlyStopping callback is around 34.

#### 3. Final Model Architecture

The final model architecture (model.py lines 176-226) consisted of a convolution neural network with the following layers and layer sizes

![alt text][image3]

Here is a visualization of the architecture (inputs on the left, outputs on the right)

![alt text][image4]


Between Adam and Nadam, Nadam seems to perform better in this case. However, this conclusion was obtained when I was running the simulator on a less powerful MacBook where the simulation results were not consistent; same model yielded different driving results. Usually the first run after the machine's back to work from some rest would succeed, but the subsequent runs on the same model would mostly fail. I spent a lot of time trying to make the results consistent by tuning the hyperparameters, optimizers, epochs, and also did minor changes in the network architecture until I found an old MacBook Pro to try on, which finally confirmed that my model was actually consistently working. So, maybe Adam is just as good as Nadam, and I have yet to re-train the model using Adam.

Adding a BatchNormalization layer to every convolution layer seems not to make difference and it would increase the number of parameters thus slow down the training. I decided not to apply BatchNormalization to the convolution layers, but only to the merged layer of the last convolution layer and the second Input layer. (i.e., before the tensor was fed into the fully-connected layers)

### Helpful Resources
* Udacity discussion forum and slack channel
* Autonomous Driving using End-to-End Deep Learning: [an AirSim tutorial]
* [Blog] An augmentation based deep neural network approach to learn human driving behavior
