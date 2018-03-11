#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

from classes import Line

#%%
# define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

# number of sliding windows for finding lines
nwindows = 9

#%%
def color_thresholded(img, cvt_from='BGR', cvt_to='HLS', channel='S',
                      thresh_min=0, thresh_max=255):
    """Return a binary image thresholded of the given color channel."""
    color_spaces = ('RGB', 'BGR', 'HLS', 'HSV', 'YUV', 'YCrCb', 'LUV', 'LAB')
    # cvt_from, cvt_to, channel = [s.upper() for s in (cvt_from, cvt_to, channel)]
    if cvt_from not in color_spaces:
        raise ValueError("cvt_from '{}' is not one of {}".format(
                cvt_from, color_spaces))
    if cvt_to not in color_spaces:
        raise ValueError("cvt_to '{}' is not one of {}".format(
                cvt_to, color_spaces))
    if cvt_to == 'YCrCb' and channel not in ('Y', 'Cr', 'Cb'):
        raise ValueError("channel '{}' is invalid for color space '{}'".format(
                        channel, cvt_to))
    if channel not in cvt_to:
        raise ValueError("channel '{}' is invalid for color space '{}'".format(
                        channel, cvt_to))

    cvt_img = np.copy(img)
    if cvt_from != cvt_to:
        cvt_format = 'COLOR_{}2{}'.format(cvt_from, cvt_to)
        cvt_img = cv2.cvtColor(cvt_img, getattr(cv2, cvt_format))
    if cvt_to == 'YCrCb':
        channel_i = 0 if channel == 'Y' else 1 if channel == 'Cr' else 2
    else:
        channel_i = cvt_to.find(channel)
    one_channel_img = cvt_img[:,:,channel_i]
    binary_output = np.zeros_like(one_channel_img)
    binary_output[(one_channel_img >= thresh_min) & \
                  (one_channel_img <= thresh_max)] = 1

    return binary_output

#%%
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    """Return a binary sobel thresholded image"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    elif orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    else:
        raise ValueError("Invalid orient input {}". format(orient))

    scaled_sobel = np.uint8(255 * (abs_sobel / np.max(abs_sobel)))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max) ] = 1

    return binary_output

#%%
def get_lines(binary_warped):
    """Return left and right lines of the binary warped image."""

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

    # Create an output image to draw on and  visualize the result
#    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    window_height = np.int(binary_warped.shape[0] // nwindows)
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_line = Line(binary_warped, leftx, lefty, 2)
    right_line = Line(binary_warped, rightx, righty, 2)

    return left_line, right_line

#%%
def pipeline(camera, image):
    """
    1. Undistort image
    2. Create binary image thresholded by color channels and sobel x-gradient
    3. Perspective transform the binary thresholded image
    4. Find lines from the warped image
    5. Calculate average curvature and distance w.r.t. the lane center
    7. Fill the lane bounded by found lines with color
    8. Inverse perspective transform the colored lane to world space and
       combine the result with the undistorted image
    9. Return images from above steps
    """
    undist = camera.undistort(image)

    r_thresh_binary = color_thresholded(undist, cvt_to='RGB', channel='R',
                                        thresh_min=200, thresh_max=255)
    g_thresh_binary = color_thresholded(undist, cvt_to='RGB', channel='G',
                                        thresh_min=170, thresh_max=255)
    rgb_thresh_binary = np.zeros_like(r_thresh_binary)
    rgb_thresh_binary[(r_thresh_binary == 1) & (g_thresh_binary == 1)] = 1
    s_thresh_binary = color_thresholded(undist, cvt_to='HLS', channel='S',
                                        thresh_min=120, thresh_max=255)
    c_thresh_binary = np.zeros_like(s_thresh_binary)
    c_thresh_binary[(rgb_thresh_binary == 1) & (s_thresh_binary == 1)] = 1

    sx_thresh_binary = abs_sobel_thresh(undist, orient='x',
                                        thresh_min=40, thresh_max=100)
    thresh_binary = np.dstack((np.zeros_like(c_thresh_binary),
                               sx_thresh_binary,
                               c_thresh_binary)) * 255
    combined_binary = np.zeros_like(sx_thresh_binary)
    combined_binary[(sx_thresh_binary == 1) | (c_thresh_binary == 1)] = 1
    warped = camera.perspective_transform(combined_binary)

    left_line, right_line = get_lines(warped)

    ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
    left_fitx = left_line.fit[0]*(ploty**2) + left_line.fit[1]*ploty + \
                left_line.fit[2]
    right_fitx = right_line.fit[0]*(ploty**2) + right_line.fit[1]*ploty + \
                 right_line.fit[2]
    # create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,50, 0))
    # plot all fitted points
    color_warp[left_line.all_ys, left_line.all_xs] = [255, 0, 0]
    color_warp[right_line.all_ys, right_line.all_xs] = [0, 0, 255]

    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx = left_line.fit[0]*ploty**2 + left_line.fit[1]*ploty + \
                left_line.fit[2]
    right_fitx = right_line.fit[0]*ploty**2 + right_line.fit[1]*ploty + \
                 right_line.fit[2]
    color_warp[np.int_(ploty), np.int_(left_fitx)] = [0, 255, 255]
    color_warp[np.int_(ploty), np.int_(right_fitx)] = [0, 255, 255]

    # warp the blank back to original image space
    newwarp = cv2.warpPerspective(color_warp, camera.Minv,
                                  (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    final = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    # calculate the radii of curvature (in meter)
    avg_curverad = (left_line.curverad + right_line.curverad) / 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(final,'Radius of Curvature = {:.2f} (m)'.format(avg_curverad),
                      (50,50), font, 1, (255,255,255), 2, cv2.LINE_AA)

    # calculate off-center distance (in meter)
    dist = (left_line.dist2center + right_line.dist2center) / 2
    if not np.isclose(dist, 0.):
        side = 'right' if dist < 0 else 'left'
    cv2.putText(final,'Vehicle is {:.2f}m {} of Center'.format(abs(dist), side),
                      (50,100), font, 1, (255,255,255), 2, cv2.LINE_AA)

    return OrderedDict([('original', image),
                        ('undist', undist),
                        ('c_thresh_binary', c_thresh_binary),
                        ('sx_thresh_binary', sx_thresh_binary),
                        ('thresh_binary', thresh_binary),
                        ('combined_binary', combined_binary),
                        ('color_warp', color_warp),
                        ('final', final)])

#%%
def plot_stages(camera, image):
    stg_images = pipeline(camera, image)
    fig, axes = plt.subplots(4, 2, figsize=(20, 20))
    fig.tight_layout()
    for ax, (img_name, img) in zip(axes.flatten(), stg_images.items()):
        if len(img.shape) == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap='gray')
        if img_name == 'combined_binary':
            src_ = np.vstack((camera.pt_src, camera.pt_src[0]))
            ax.plot(src_[:,0], src_[:,1], 'r')
        if img_name == 'warped':
            ax.plot(camera.pt_dst[[0,3],0], camera.pt_dst[[0,3],1], 'r')
            ax.plot(camera.pt_dst[[1,2],0], camera.pt_dst[[1,2],1], 'r')
        ax.set_title(img_name, fontsize=20)
    plt.subplots_adjust(top=1.1, hspace=0.3)
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()

#%%
