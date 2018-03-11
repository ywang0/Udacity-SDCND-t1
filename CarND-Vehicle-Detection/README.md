
### Writeup / README

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color.  
(Note: Need to normalize the feature set and randomize a selection for training and testing.)
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (project_video.mp4): create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected using the above heat map.

[//]: # (Image References)
[image1]: ./examples/car_notcar_hog.png
[image2]: ./examples/slide_windows_1520.png
[image3]: ./examples/slide_windows_15.png
[image4]: ./examples/slide_windows_ex1.png
[image5]: ./examples/slide_windows_ex2.png
[image6]: ./examples/series_frames_heatmap_1.png
[image7]: ./examples/series_frames_heatmap_2.png
[image8]: ./examples/integrated_heatmap.png
[image9]: ./examples/output_bboxes.png
[video result]: ./project_video.mp4
[notebook]: ./CarND-Vehicle-Detection.ipynb

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

(Here is a companion [notebook])

---

### Histogram of Oriented Gradients (HOG)

#### Explain how (and identify where in your code) you extracted HOG features from the training images.

HOG (Histograms of Oriented Gradients) is a global feature descriptor based on image pixels' gradeints, to describes an object with a single feature vector/array.

In this project, the classifier uses the extracted HOG features to identify whether an input image is a car or not.  
Below images show how HOGs are different in a car image and in a not-car image.
- First row shows car/not-car images and their representations in YCrCb color space
- The rest shows per channel HOG (gradeints of cells)

![alt text][image1]

The `skimage.feature.hog()` (in `vehicle_detection_utils.py` `get_hog_features_sk()`) takes an image in YCrCb color space with parameters of orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), and returns a HOG feature vector and a gradient graph for visualization.

To achieve higher classification accuracy, a finer granularity of the parameters is chosen, i.e., orientations=9 not 8, pixels_per_cell=(8, 8) not (16, 16) and cells_per_block=(2, 2) not (3, 3).


#### 2. Train a SVM classifier with extracted features

HOG features, SVM, and sliding-windows are a popular combination for image detection tasks in the field of classical image processing. The combination is introduced in the lessons and used here.  

Note: Replacing SVM with a DNN (i.e., LeNet 5) slightly improves the accuracy, but the inference time is much longer and the false positive rate is high when applying to video implementation.

** Data sets **  

The training images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.

Due to frequent false positives shown in video impelmentation, I've added ~1700 (hard negative mining + augmented) images (`./neg_mining.tar.gz`) to reduce the false positve rate.

** Features Extraction **  

In addition to HOG features, I also added spatial features and color-histograms features with `spatial_size=32` and `hist_bins=48` to achieve higher classification accuracy. Compared to RGB, HSV and HSL, YCrCb color space yields a higher accuracy. YUV color space achieves a comparable accuracy, therefore YUV can also be used.  

These are functions used to extract features ( in `vehicle_detection_utils.py`):
- `get_hog_features()`: extract HOG features
- `get_spatial_features()`: extract spatial features
- `get_colorhist_features()`: extract color-histograms features

** The Classifier **  

`LinearSVC` is chosen as the classifier.  

`LinearSVC` and `svm.SVC` with linear kernel are both linear support vector machine, but the former run much faster. `svm.SVC` has "fit time complexity is more than quadratic with the number of samples" per [sklearn documentation](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html), that makes training with `svm.SVC` way much slower.  

(See [notebook] cell 10)  
As the accuracy is 99.2+%, I did not tune the hyperparameters C and gamma but use the default values.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

To systematically survey regions of an image, I use window(s) to slide through x- and y- directions. The collection of slided regions (image patches) are fed to the trained SVM classifier to be determined whether a region is a car or not.  

The parameters are:
- window size (x and y)
- overlap per step (x and y)
- region to search for  

In the actual implementation, instead of using different window size, we have a fixed `window=64` and use `scale` to scale the image, which equivalently changes the window size. Also, instead of specifying overlap percentage per slide, number of cells per step is used. This is necessary for one-shot HOG feature extraction because 'cell' is the smallest unit in HOG descriptor, we won't be able to access the gradients that does not start at some cell's boundary.  

Here is an example shows windows returned from `sliding_windows()` (in `vehicle_detection_utils.py`), with scales = 1.5 and 2, number of cell per step = 2. We also limit our search region below the horizon and exclude the bottom portion of the image where no car would appear (y = 380 ~ 680).

![alt text][image2]

I decided to use single scale = 1.5 and step per step = one cell.  

Adding scale=1 results more accurate bounding box but also dramatically increases per frame processing time. Adding scale=2 stabilize the bounding boxes slightly but again slow down the video processing (though it's still acceptable). On the other hand, using single scale=1.5 seems to be good enough to approximately cover the two nearest cars in all distances, near and far.  

And the number of cells per step has to be one, in order to create reasonably tight and stable bounding boxes.  
Below is the windows returned from `sliding_windows()` with `scale=[1.5]` and `cells_per_step=1`.

![alt text][image3]


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

To make the feature extraction efficient, I used one-shot HOG extraction on the whole image and then get the sub-images HOG features by slicing from the HOG features of the whole image.  

Ultimately I searched on one scale(=1.5) using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  

Here are some example images:

![alt text][image4]
![alt text][image5]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a link to my [video result](./project_video.mp4)  
The result also shows the lane found in project 4, "Advanced Lane Lines".

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heat map and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heat map.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heat map from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

#### Here are six frames and their corresponding heatmaps:

![alt text][image6]
![alt text][image7]

#### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image8]

#### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image9]

---

### Discussion

#### Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

** Performance of HOG API **  
In the lessons, we used `skimage.feature.hog()` to extract HOG of an image. Though the function is convenient in terms of passing parameters and getting returned values as vector/non-vector, it is quite slow and becomes an issue when applying to videos; rendering of the video `project_video.mp4` would take hours.

   A quick comparison of run time on `skimage.feature.hog()` and `cv2.HOGDescriptor()` shows `cv2.HOGDescriptor()` is usually 15+ time faster than `skimage.feature.hog()`, therefore `cv2.HOGDescriptor()` is actually used in the implementation.  
   (see [notebook] cell 6)

** The number of windows for searching is too big **   
I used single scale=1.5 and step=one-cell for sliding_windows search. The number of windows for searching is more than 1700 and that makes the video processing slow. To improve the performance, the different region proposal techniques used in RPN family may be applied here. If the feature extraction time can be significantly reduced, we can add more scales to fine tuning the bounding boxes and to be able to detect more cars of various sizes.

** Frames per Second Improvement **  
Currently the frame processing is too slow (~ 1.69s per frame). To get real time performance much improved, using CNN like SSD or YOLO could be a better approach for this task.

### Helpful Resources
* Uacity discussion forum and slack channel
* A very good explanation of HOG can be found [here](http://mccormickml.com/2013/05/09/hog-person-detector-tutorial/).
* Reshape 1-D cv2.HOGDescriptor() to 5-dimensional array. [stackoverflow](https://stackoverflow.com/questions/6090399/get-hog-image-features-from-opencv-python?noredirect=1&lq=1) answered by user 'pixelou'.
