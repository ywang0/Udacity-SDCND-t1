#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
from scipy.ndimage.measurements import label

#%%
def convert_cspace(img, cvt_from='RGB', cvt_to='BGR'):
    """Return an image that is converted from the color space cvt_from to the
    color space cvt_to."""
    cspaces = ('RGB', 'BGR', 'HLS', 'HSV', 'YUV', 'YCrCb', 'LUV', 'LAB')
    if cvt_from not in cspaces:
        raise ValueError("cvt_from '{}' is not one of {}".format(
                cvt_from, cspaces))
    if cvt_to not in cspaces:
        raise ValueError("cvt_to '{}' is not one of {}".format(
                cvt_to, cspaces))

    cvt_img = np.copy(img)
    if cvt_from != cvt_to:
        cvt_format = 'COLOR_{}2{}'.format(cvt_from, cvt_to)
        cvt_img = cv2.cvtColor(cvt_img, getattr(cv2, cvt_format))

    return cvt_img


def get_spatial_features(img, spatial_size):
    """Return the spatial fetaures of the input image when resized to the given
    spatial_size"""
    return cv2.resize(img, spatial_size).ravel()


def get_colorhist_features(img, bins):
    """Return the concatenated per channel binned histogram."""
    bins_range = (img.min(), img.max()+1)
    ch1_hist = np.histogram(img[:,:,0], bins=bins, range=bins_range)[0]
    ch2_hist = np.histogram(img[:,:,1], bins=bins, range=bins_range)[0]
    ch3_hist = np.histogram(img[:,:,2], bins=bins, range=bins_range)[0]

    return np.concatenate((ch1_hist, ch2_hist, ch3_hist))


# skimage.feature.hog()
def get_hog_features_sk(img, orient=9, pix_per_cell=8, cell_per_block=2,
                        vis=False, feature_vec=True):
    """Return HOG features using skimage.feature.hog()"""
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), block_norm= 'L2-Hys',
                                  transform_sqrt=True, visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), block_norm= 'L2-Hys',
                       transform_sqrt=True, visualise=vis, feature_vector=feature_vec)
        return features


# cv2.HOGDescriptor()
def get_hog_features(img, orient=9, pix_per_cell=8, cell_per_block=2,
                     feature_vec=True):
    """Return HOG features using cv2.HOGDescriptor(). If feature_vec is False,
    the 1-D result is reshaped back to a 5-dimensional array."""
    window_size = (img.shape[1] // pix_per_cell * pix_per_cell,
                   img.shape[0] // pix_per_cell * pix_per_cell)
    block_size = (cell_per_block * pix_per_cell, cell_per_block * pix_per_cell)
    cell_size = (pix_per_cell, pix_per_cell)
    block_stride = cell_size

    hog = cv2.HOGDescriptor(_winSize=window_size,
                            _blockSize=block_size,
                            _blockStride=block_stride,
                            _cellSize=cell_size,
                            _nbins=orient)

    n_cells = (img.shape[0] // pix_per_cell, img.shape[1] // pix_per_cell)
    hog_features = hog.compute(img).reshape(n_cells[1] - cell_per_block + 1,
                                            n_cells[0] - cell_per_block + 1,
                                            cell_per_block, cell_per_block,
                                            orient).transpose(1, 0, 2, 3, 4)
    if feature_vec:
        hog_features = np.ravel(hog_features)

    return hog_features


def extract_features(img,
                     spatial_size=(32,32),
                     hist_bins=32,
                     orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
                     spatial_feat=True,
                     hist_feat=True,
                     hog_feat=True,
                     feature_vec=True):
    """
    Extract feature vector of the given image.

    Args:
        img (array of int or float): the input image
        spatial_size (int, int): spatial binning dimensions (x, y)
        hist_bins (int): number of histogram bins of per color channel
        orient (int): HOG orientations
        pix_per_cell (int): HOG pixels per cell
        cell_per_block (int): HOG cells per block
        hog_channel (int or str): color channel to be extracted HOG feature
                    from, can be 0, 1, 2, or 'ALL'
        spatial_feat (bool): whether to extract spatial features
        hist_feat (bool): whether to extract color histogram feature
        hog_feat (bool): whether to extract HOG feature

    Return:
        A 1-D feature vector
    """
    features_list = []
    if spatial_feat:
        features_list.append(get_spatial_features(img, spatial_size))
    if hist_feat:
        features_list.append(get_colorhist_features(img, hist_bins))
    if hog_feat:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(img.shape[2]):
                hog_features.append(get_hog_features(img[:,:,channel], orient,
                                                     pix_per_cell, cell_per_block,
                                                     feature_vec=feature_vec))
            features_list.append(np.ravel(hog_features))
        else:
            hog_feature = get_hog_features(img[:,:,hog_channel], orient,
                                           pix_per_cell, cell_per_block,
                                           feature_vec=feature_vec)
            features_list.append(hog_feature)

    return np.concatenate(features_list)

#%%
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """Return a copy of the input image with the given bboxes drawn."""
    img_ = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(img_, bbox[0], bbox[1], color, thick)

    return img_


def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0

    return heatmap


def get_heatmap(hot_windows, image_size, threshold):
    """Return a heatmap given the positive windows (hot_windows)."""
    heat = np.zeros(image_size).astype(np.float)
    heat = add_heat(heat,hot_windows)
    heat = apply_threshold(heat, threshold)
    heatmap = np.clip(heat, 0, 255)

    return heatmap


def get_bounding_boxes(heatmap):
    """Return a list of (top-left, bottom-right) pairs of bounding boxes."""
    bboxes = []
    labeled_array, n_cars = label(heatmap)

    for car in range(1, n_cars + 1):
        nonzeroy, nonzerox = (labeled_array == car).nonzero()
        bboxes.append(((np.min(nonzerox), np.min(nonzeroy)),
                       (np.max(nonzerox), np.max(nonzeroy))))

    return bboxes

#%%
def sliding_windows(image, scales=[1.5, 2.0],
                    y_start_stop=[None, None],
                    pix_per_cell=8, cell_per_block=2,
                    window=64):
    """
    Return all coordinate pairs of various window sizes depending on the given
    scales. The effective window size is equal to window * scale.
    """
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = image.shape[0]

    windows = []
    for scale in scales:
        img = image[y_start_stop[0]:y_start_stop[1], :]
        y_size = int(img.shape[0] / scale)
        x_size = int(img.shape[1] / scale)
        if scale != 1.0:
            img = cv2.resize(img, (x_size, y_size))

        # Define blocks and steps as above
        x_nblocks = (img.shape[1] // pix_per_cell) - cell_per_block + 1
        y_nblocks = (img.shape[0] // pix_per_cell) - cell_per_block + 1

        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 1  # Instead of overlap, define how many cells to step
        x_nsteps = (x_nblocks - nblocks_per_window) // cells_per_step + 1
        y_nsteps = (y_nblocks - nblocks_per_window) // cells_per_step + 1

        for xb in range(x_nsteps):
            for yb in range(y_nsteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                top_left = (xbox_left, ytop_draw+y_start_stop[0])
                bottom_right = (xbox_left+win_draw,ytop_draw+win_draw+y_start_stop[0])

                windows.append((top_left, bottom_right))

    return windows


def search_windows(image, svc, scaler, scales=[1.5],
                   cvt_from='RGB',cvt_to='YCrCb',
                   y_start_stop=[None, None],
                   spatial_size=(32, 32),
                   hist_bins=32,
                   orient=9, pix_per_cell=8, cell_per_block=2,
                   window=64):
    """
    Search all positive windows in the given image using sliding window.

    Args:
        image (array of int or float): a 3-channel image
        svc (object): a trained SVM model
        scaler (object): a fitted instance of sklearn.preprocessing.StandardScaler
        scales (list of int): a list of scales for the sliding window
        window (int): sliding window size (assuming the window is of the shape of
                    square)

    Return:
        A list of coordinates pair of all positive windows.
    """
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = image.shape[0]

    hot_windows = []

    for scale in scales:
        img = convert_cspace(image, cvt_from=cvt_from, cvt_to=cvt_to)
        img = img[y_start_stop[0]:y_start_stop[1], :]
        y_size = int(img.shape[0] / scale)
        x_size = int(img.shape[1] / scale)
        if scale != 1.0:
            img = cv2.resize(img, (x_size, y_size))

        ch1 = img[:, :, 0]
        ch2 = img[:, :, 1]
        ch3 = img[:, :, 2]

        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block,
                                feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block,
                                feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block,
                                feature_vec=False)

        # Define blocks and steps as above
        x_nblocks = (img.shape[1] // pix_per_cell) - cell_per_block + 1
        y_nblocks = (img.shape[0] // pix_per_cell) - cell_per_block + 1

        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 1  # Instead of overlap, define how many cells to step
        x_nsteps = (x_nblocks - nblocks_per_window) // cells_per_step + 1
        y_nsteps = (y_nblocks - nblocks_per_window) // cells_per_step + 1

        for xb in range(x_nsteps):
            for yb in range(y_nsteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat = np.concatenate((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(img[ytop:ytop+window, xleft:xleft+window], (64,64))

                # Get color features
                spatial_feat = get_spatial_features(subimg,
                                                    spatial_size=spatial_size)
                hist_feat = get_colorhist_features(subimg, bins=hist_bins)

                # Scale features and make a prediction
                features = np.concatenate([spatial_feat, hist_feat, hog_feat]).reshape(1, -1)
                features = scaler.transform(features)
                prediction = svc.predict(features)

                if prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    top_left = (xbox_left, ytop_draw+y_start_stop[0])
                    bottom_right = (xbox_left+win_draw,ytop_draw+win_draw+y_start_stop[0])
                    hot_windows.append((top_left, bottom_right))

    return hot_windows

#%%
### helper functions used by writeup notebook

def plot_images(img_list, figsize=(20, 8), gray=False):
    n_imgs = len(img_list)
    fig, axes = plt.subplots(1, n_imgs, figsize=figsize)
    for i, (name, img), ax in zip(range(n_imgs), img_list, axes.ravel()):
        ax.set_title(name, fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
        if gray:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)


def demo_car_notcar(cars, notcars, vis=True, feature_vec=False):
    car = mpimg.imread(cars[7297])
    car_ycrcb = cv2.cvtColor(car, cv2.COLOR_RGB2YCrCb)
    notcar = mpimg.imread(notcars[2854])
    notcar_ycrcb = cv2.cvtColor(notcar, cv2.COLOR_RGB2YCrCb)

    imgs = [('Car', car), ('Car YCrCb', car_ycrcb),
            ('NotCar', notcar), ('NotCar YCrCb', notcar_ycrcb)]
    plot_images(imgs)

    for i in range(car.shape[2]):
        _, car_hog_img = get_hog_features_sk(car_ycrcb[:,:,i],
                                             vis=vis, feature_vec=feature_vec)
        _, notcar_hog_img = get_hog_features_sk(notcar_ycrcb[:,:,i],
                                                vis=vis, feature_vec=feature_vec)
        imgs_ch = [('Car YCrCb Ch{}'.format(str(i+1)), car_ycrcb[:,:,i]),
                   ('Car YCrCb Ch{} HOG'.format(str(i+1)), car_hog_img),
                   ('NotCar YCrCb Ch{}'.format(str(i+1)), notcar_ycrcb[:,:,1]),
                   ('NotCar YCrCb Ch{} HOG'.format(str(i+1)), notcar_hog_img)]
        plot_images(imgs_ch, gray=True)


def demo_search_windows(image, lsvc, scaler):
    hot_windows= search_windows(image, lsvc, scaler, hist_bins=48,
                                scales = [1.5],
                                y_start_stop=[380, 680],
                                window=64)
    hot_windows_img = draw_boxes(image, hot_windows)
    img_list = [('Original image', image),
                ('Image with searched windows', hot_windows_img)]
    plot_images(img_list)


def demo_series_heatmaps(image_series, lsvc, scaler):
    img_list = []
    for f in image_series:
        fname = os.path.split(f)[1]
        img = mpimg.imread(f)
        hot_windows= search_windows(img, lsvc, scaler, hist_bins=48,
                                    scales = [1.5],
                                    y_start_stop=[380, 680],
                                    window=64)
        hot_windows_img = draw_boxes(image, hot_windows)
        heatmap = get_heatmap()

#%%
