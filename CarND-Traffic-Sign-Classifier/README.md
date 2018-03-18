# **Traffic Sign Recognition**

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

Here is the [notebook] for the code and details.


[//]: # (Image References)

[image1]: ./examples/class_images.png "Class Images"
[image2]: ./examples/data_distribution.png "Data Distribution"
[image3]: ./examples/original_images.png "Original Images"
[image4]: ./examples/datagen_images.png "DataGen Images"
[image5]: ./examples/five_new_images.png "Selected Five New Images"
[image6]: ./examples/new_images_pred.png "New Images Top-1 Predictions"
[image7]: ./examples/new_images_top_5_probs.png "New Images Top 5 Probabilities Sign 4"
[notebook]: ./Traffic_Sign_Classifier.ipynb
[VisualBackProp]: <https://arxiv.org/pdf/1611.05418.pdf>


### Data Set Summary & Exploration

#### 1. A basic summary of the data set.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. An exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.

One sample image per class
![alt text][image1]

Class distribution
![alt text][image2]

Apparently the class distribution is imbalanced for all training, validation, and test sets; however, the three distributions share a similar pattern. We can also see that the distributions of training and test sets are almost identical.

### Design and Test a Model Architecture

#### 1. Image processing and data augmentation.
What techniques were chosen and why did you choose these techniques?

As a first step, I normalize the images as suggested: (pixel - 128.)/ 128., to make the data to have zero mean and approximately equal variance.
I also converted the images to grayscale using OpenCV's API and had a little performance gain (in terms of per epoch speed and validation accuracy). When data augmentation was considered later, I decided to use RGB images (normalized) instead of grayscaled images. Based on the result of test set, I conjecture the color channels could possibly provide useful information.

I decided to generate additional data to see if the additional data would improve the performance, especially for those classes with less samples.  
For data augmentation, __OpenCV's API__ seems to be the first choice; however, the API processes only one image per call; it was too slow when processing a large set. Besides, additional memory is required to store the augmented data (the size could be large). __TensorFlow's Images module__ also provides image processing APIs, but I found them not easy nor flexible to use (perhaps I'm still new to TF) and lack of some useful functionality like shifting and zooming. In the end, I decided to use __Keras's ImageDataGenerator__. The single API lets me specify/tune various parameters that meets most of the needs. Further more, it's a generator, the augmented data will be generated in real time; no additional memory required. Nevertheless, `ImageDataGenerator` does not solve the imbalanced-distribution problem; it generates data of a similar class distribution as the input set. I will need to produce the new data separately for certain classes. (not in this work though)


Here is an example of a traffic sign image before and after ImageDataGenerator.

```python
datagen = ImageDataGenerator(
                            rotation_range=10,
                            width_shift_range=0.1,
                            height_shift_range=0.05,
                            zoom_range=0.1,
                            horizontal_flip=False,
                            fill_mode='nearest')
```

Before
![alt text][image3]

After
![alt text][image4]




#### 2. The model architecture

I use a LeNet-5 model with dropout layer for fully-connected layers. It consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| ReLU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| ReLU                  |                                               |
| Max pooling	      	| 2x2 stride,  outputs 5x5x16     				|
| Fully connected		| 400 -> 120   									|
| Dropout(keep_prob=0.5)|                                               |
| Fully connected		| 120 -> 84   									|
| Dropout(keep_prob=0.5)|                                               |
| Softmax				| 84 -> 43     									|




#### 3. Describe how you trained your model.

To train the model, I use the following configurations:
* learning_rate = 0.001
* optimizer = Adam optimizer
* batch_size = 128 (w/o data augmentation), 128 + 32 (w/ data augmentation)
* epochs = max value is 201, but training would stop early if the validation loss is not getting improvement for 20 epochs

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated.

My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of 97.1%
* test set accuracy of 96.1%

In an iterative approach:
* What was the first architecture that was tried and why was it chosen?

  The first architecture tried was plain LeNet-5, since it's the simplest ConvNet and we've just implemented in the lectures.  

  Before training the whole training set, I trained with a dataset consisting of 100 samples per class, to make sure the network can overfit this small dataset.

* What were some problems with the initial architecture?

  With the plain LeNet-5, the validation accuracy progressed slowly. The training accuracy and validation accuracy were getting farther apart , generally by more than 10%; the model seemed to be overfitting the training set.

* How was the architecture adjusted and why was it adjusted?

  By adding a dropout layer after every activation layer, the accuracy gap decreased and the validation accuracy reached 93% much sooner (50+, 100- epochs) than the previous result.  

  Nevertheless, it makes more sense to me that dropout is better applied to only  fully-connected layers but not convolution layers. The reasoning is that the model has only two convolution layers, and the number of parameters for each convolution layer is small, thus it's unlikely the overfitting would be caused by convolution layers. By removing dropout from convolution layers, the net achieved 93+% accuracy well within 50 epochs.

  I've also tried converting images to grayscale and applying early stopping when the validation loss becomes stale. The performance for validation accuracy and test accuracy were 96-97% and 94-95% respectively. Later, data augmentation was considered, and I decided to use RGB instead of grayscale. The performance improves a tiny bit but now the validation accuracy can reach a steady 96% at around 25 epochs and the test accuracy 96%.

* Which parameters were tuned? How were they adjusted and why?

  I don't really fine tune the parameters, except the dropout rate. keep_prob value of 1.0 (no dropout), 0.7, 0.6, and 0.5 were tried, and I use 0.5 in the end. Learning rate is pre-set to be 0.001 and it's not changed since the performance has been good.

  I also use a batch size of 128 because of the larger dataset (compared to MNIST data). When data augmentation technique is used, 32 newly generated data are added to the original batch, which results a batch size of 160.

  As for epochs, I set a maximum value of 201 but the network will stop training when the validation loss is not improving for 20 epochs. The current network stops at around 60 epochs.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

  __ConvNet__ was chosen because of its revolutionary performance achieved in solving computer vision problems.

  Contrast to fully-connected layers, a convolution layer uses a shared kernel (set of parameters) convolving its inputs to extract features, that mechanism preserves the locally correlated information of the inputs (as opposed to global information obtained via a fully-connected layer) and empowers the extracted features to be scale/location invariant. These two major characteristics of ConvNet have made ConvNet perform so well for image tasks.

  The shared kernels in convolution layers also greatly reduce the number of trainable parameters. That in turn speeds up the training cycles and accelerates the process of model development.

  The vast parameters that a deep neural network optimizes has always been a concern of overfitting. A __dropout layer__'s turning off the randomly selected activations of the previous layer has equivalently created many neural networks, in that each network has different neurons being activated, thus has various capability. Like Bootstrap Aggregation in RandomForest, the effect of adding dropout layers reduces variance of the original network, alleviate overfitting, and thus improve the network performance.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5]

The third and forth images might be difficult to classify because they have been shifted and sheared.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image6]


The model was able to correctly guess all the 5  traffic signs.  
I actually use 25 new images (please see Traffic_Sign_Classifier_5.ipynb), which give Top-1 accuracy of 80% and Top-5 of 84%.

It doesn't make much sense to compare the performance on the new images to the performance on the test set. Since the number of new images are so small and we are free to choose whatever images on the Internet, I can easily make it 100% or less than 50%. (It's probably not easy to make it 0%, as the network seems to be quite powerful.)

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 37th and 38th cells of the Ipython notebook.

And here is the softmax probabilities for the five images:

![alt text][image7]

Except for the first image, the model is relatively sure on what it is predicting.


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Visualizing the 2nd convolution layer's activations, we can see that slanted lines and arcs are activated for triangular signs and circular signs respectively. However, the other activation patterns are still quite obscure to me.

It would be really cool if we can project the activations of the final convolution layer back to the input image space, and then visualize what exact characteristics of the images that the network uses to make classifications (the paper: [VisualBackProp]).
