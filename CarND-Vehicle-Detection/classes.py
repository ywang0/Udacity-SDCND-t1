#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import cv2
import glob
import numpy as np
import os
import pickle
from collections import deque

#%%
# define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

#%%
def init_camera(cal_file, cal_path, cb_size, src, offsets):
    """
    Return a camera object with camera matrix, distortion coefficients, and
    perspective transformation matrix initialized.

    Args:
        cal_file (str): The file that stores camera's calibration coefficients
                        and transformation matrices
        cal_path (str): The path to calibration files
        cb_size (iny, int): The size of chess board (y, x)
        src (4x2 array(int)): The vertices of source trapezoid for perspective
                        transformation
        offsets (int, int): x and y offsets from warp edges to calculate the
                        vertices of the destination rectangle for perspective
                        transformation
    Return:
        A camera object with properties:
            img_size: image size
            mtx: camera matrix
            dist: distortion coefficients
            src: source vertices for perspective transformation
            dst: destination vertices for perspective transformation
            M: perspective transformation matrix
            Minv: perspective transformation matrix

    Note: If cal_file exists, camera object is created with the data read from
          cal_file, otherwise all data are calculated using calibration
          files located at cal_path and save to cal_file.

          To use an existing calibration file, cal_file='./camera_cal.p'.
          To use an existing calibration path, cal_path='./camera_cal/'.
    """
    if os.path.isfile(cal_file):
        with open(cal_file, 'rb') as f:
            params = pickle.load(f)
    else:
        params = {}
        cb_files = glob.glob(cal_path)
        img_size =  get_image_size(cb_files[0])
        params['img_size'] = img_size
        mtx, dist = get_mtx_dist(cb_files, cb_size, img_size)
        params['mtx'] = mtx
        params['dist'] = dist

        offsetx, offsety = offsets
        dst = np.float32([[offsetx, offsety],
                          [img_size[0]-offsetx, offsety],
                          [img_size[0]-offsetx, img_size[1]-offsety],
                          [offsetx, img_size[1]-offsety]])
        params['src'] = src
        params['dst'] = dst
        params['M'] = cv2.getPerspectiveTransform(src, dst)
        params['Minv'] = cv2.getPerspectiveTransform(dst, src)
        with open(cal_file, 'wb') as f:
            pickle.dump(params, f)

    return Camera(**params)

#%%
def get_image_size(img_file):
    img = cv2.imread(img_file)
    return img.shape[1], img.shape[0]


def get_mtx_dist(cb_files, cb_size, img_size):
    """Calculate camera matrix and distortion coefficients from the given
    calibration files."""
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    ny, nx = cb_size
    objp = np.zeros((ny*nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # search for chessboard corners
    for f in cb_files:
        img = cv2.imread(f)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    # get camera parameters
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                       img_size, None, None)
    if ret:
        return mtx, dist
    else:
        raise ValueError("Camera calibration failed!")

#%%
class Camera(object):
    """Camera class definition"""
    def __init__(self, **params):
        self.pt_src = params['src']  # perspective transform source vertices
        self.pt_dst = params['dst']  # perspective transform destination vertices
        self.img_size = params['img_size']  # image/frame size
        self.mtx = params['mtx']     # camera matrix
        self.dist = params['dist']   # camera distortion parameters
        self.M = params['M']         # perspective transformation matrix
        self.Minv = params['Minv']   # inverse perspective transformation matrix

    def undistort(self, image):
        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

    def perspective_transform(self, image):
        return cv2.warpPerspective(image, self.M, self.img_size,
                                   flags=cv2.INTER_LINEAR)

    def inverse_perspective_transform(self, image):
        return cv2.warpPerspective(image, self.Minv, self.img_size,
                                   flags=cv2.INTER_LINEAR)
    
    @classmethod
    def from_cal_file(cls, cal_file):
        """Create Camera object from the give calibration file."""
        if os.path.isfile(cal_file):
            with open(cal_file, 'rb') as f:
                params = pickle.load(f)
        else:
            raise OSError("calibration file {} is not found!".format(cal_file))
            
        return cls(**params)
        
#%%
class Line(object):
    """
    Line class for lane lines.

    Args:
        binary_warp (2-D array({0, 1})): An image of binary values (i.e., 0 or 1)
        xs (vector(int)): The x-coordinates of points for fitting a line
        ys (vector(int)): THe y-coordinates of points for fitting a line
        deg (int): The order of the polynomial of the line
        que_size (int): The maxlen of double-ended queues, which are to keep
                        points and fits of at most que_size consecutive good
                        frames to fit a line. The purpose is to smooth the
                        transition of lines from frame to frame.
    """
    def __init__(self, binary_warped, xs, ys, deg=2, que_size=3):
        self.binary_warped = np.copy(binary_warped)
        self.img_size = (self.binary_warped.shape[1], self.binary_warped.shape[0])
        self.recent_fits = deque(maxlen=que_size)
        self.recent_xs = deque(maxlen=que_size)
        self.recent_ys = deque(maxlen=que_size)
        self.xs = xs
        self.ys = ys
        self.all_xs = xs    # x-coordinates of all points to fit a line
        self.all_ys = ys    # y-coordinates of all points to fit a line
        self.deg = deg

        self.init_line()
        self.curverad = self.calc_curvature()
        self.dist2center = self.calc_dist2center()

    def init_line(self):
        self.fit = np.polyfit(self.ys, self.xs, self.deg)
        self.current_fit = self.fit
        self.recent_fits.append(self.current_fit)
        self.reset_line_base = False

    def get_fitx(self, y):
        # return x of the polyline given y
        return self.fit[0]*(y**2) + self.fit[1]*y + self.fit[2]

    def calc_curvature(self):
        # fit new polynomials to x,y in world space
        fit_cr = np.polyfit(self.all_ys*ym_per_pix, self.all_xs*xm_per_pix, 2)

        # calculate the new radius of curvature
        y_max = self.img_size[1] - 1
        return ((1 + (2*fit_cr[0]*y_max*ym_per_pix + fit_cr[1])**2)**1.5) / \
                np.absolute(2*fit_cr[0])

    def calc_dist2center(self):
        # calculate the distance from line to the center
        y_max = self.img_size[1] - 1
        x = self.fit[0]*(y_max**2) + self.fit[1]*y_max + self.fit[2]
        x_center = self.img_size[0] / 2
        return (x - x_center) * xm_per_pix

    def update_line(self, new_fit, xs, ys):
        self.current_fit = new_fit
        self.xs = xs
        self.ys = ys
        self.recent_fits.append(new_fit)
        self.recent_xs.append(xs)
        self.recent_ys.append(ys)
        self.all_xs = np.concatenate(self.recent_xs)
        self.all_ys = np.concatenate(self.recent_ys)
        self.fit = np.polyfit(self.all_ys, self.all_xs, self.deg)

        self.curverad = self.calc_curvature()
        self.dist2center = self.calc_dist2center()

    def update(self, binary_warped, margin=100, min_points=50):
        """Update the line of a new frame based on the previous fitted line
        with margin, rather than searching from scratch. If more than min_points
        plausible points are found, the line is updated with the new added
        points."""
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        lane_inds = ((nonzerox > (self.fit[0]*(nonzeroy**2) +
                                  self.fit[1]*nonzeroy +
                                  self.fit[2] - margin)
                     ) &
                     (nonzerox < (self.fit[0]*(nonzeroy**2) +
                                  self.fit[1]*nonzeroy +
                                  self.fit[2] + margin)
                     ))

        if len(lane_inds) >= min_points:
            # extract line pixel positions
            xs = nonzerox[lane_inds]
            ys = nonzeroy[lane_inds]

            new_fit = np.polyfit(ys, xs, self.deg)
            if new_fit is not None:
                self.update_line(new_fit, xs, ys)

#%%
