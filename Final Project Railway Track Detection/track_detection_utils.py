import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def abs_sobel_thresh(img, orient="x", thresh_min=0, thresh_max=255):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == "x":
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    else:  # if orient == "y":
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I"m using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def hls_select(img, channel="s", thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if channel == "h":
        channel = hls[:, :, 0]
    elif channel == "l":
        channel = hls[:, :, 1]
    else:
        channel = hls[:, :, 2]
    binary_output = np.zeros_like(channel)
    binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 1
    return binary_output


def luv_select(img, thresh=(0, 255)):
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    l_channel = luv[:, :, 0]
    binary_output = np.zeros_like(l_channel)
    binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 1
    return binary_output


def lab_select(img, thresh=(0, 255)):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    b_channel = lab[:, :, 2]
    binary_output = np.zeros_like(b_channel)
    binary_output[(b_channel > thresh[0]) & (b_channel <= thresh[1])] = 1
    return binary_output


def find_line(img_binary_trans: np.ndarray):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img_binary_trans[img_binary_trans.shape[0] // 2:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(img_binary_trans.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img_binary_trans.nonzero()
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
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img_binary_trans.shape[0] - (window + 1) * window_height
        win_y_high = img_binary_trans.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
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

    # Fit a second order polynomial to each
    left_fit = np.polyfit(x=lefty, y=leftx, deg=3)
    right_fit = np.polyfit(x=righty, y=rightx, deg=3)

    return left_fit, right_fit, left_lane_inds, right_lane_inds


def find_line_by_previous(img_binary_trans, left_fit, right_fit):
    nonzero = img_binary_trans.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = (
            (nonzerox > (left_fit[0] * (nonzeroy ** 3)
                         + left_fit[1] * (nonzeroy ** 2)
                         + left_fit[2] * nonzeroy
                         + left_fit[3]
                         - margin))
            & (nonzerox < (left_fit[0] * (nonzeroy ** 3) +
                           left_fit[1] * (nonzeroy ** 2) +
                           left_fit[2] * nonzeroy +
                           left_fit[3]
                           + margin))
    )

    right_lane_inds = (
            (nonzerox > (right_fit[0] * (nonzeroy ** 3)
                         + right_fit[1] * (nonzeroy ** 2)
                         + right_fit[2] * nonzeroy
                         + right_fit[3]
                         - margin))
            & (nonzerox < (right_fit[0] * (nonzeroy ** 3)
                           + right_fit[1] * (nonzeroy ** 2)
                           + right_fit[2] * nonzeroy
                           + right_fit[3]
                           + margin))
    )

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 3)
    right_fit = np.polyfit(righty, rightx, 3)
    return left_fit, right_fit, left_lane_inds, right_lane_inds


def expand(img):
    image = img
    height, width, _ = img.shape
    expand_mask = np.zeros((height, width))
    _, green, _ = cv2.split(image)  # split the GREEN channel of RGB
    s = np.sum(green, axis=1)  # sum up cnt of pure-green pixels (railway track) (by width) in a row
    for i in reversed(range(height)):  # iterate (by height) from bottom row to top row
        if s[i] < 200:  # skip rows that have insufficient railway-track pixels
            break
        for j in range(width):  # min x of the railway-track in the current row
            if green[i][j] == 255:
                break
        for k in reversed(range(width)):  # max x of the railway-track in the current row
            if green[i][k] == 255:
                break
        for l in range(int(s[i] / 255)):  # expand the railway-track region
            # from left
            if j - l >= 0:
                image[i, j - l, 2] = 255
                expand_mask[i, j - l] = 255
            # from right
            if k + l <= width - 1:
                image[i, k + l, 2] = 255
                expand_mask[i, k + l] = 255

    return image, expand_mask


def draw_area(img, img_binary_trans, trans_dst_2_src, left_fit, right_fit):
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_binary_trans.shape[0] - 1, img_binary_trans.shape[0])
    # left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    # right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    left_fitx = left_fit[0] * ploty ** 3 + left_fit[1] * ploty ** 2 + left_fit[2] * ploty + left_fit[3]
    right_fitx = right_fit[0] * ploty ** 3 + right_fit[1] * ploty ** 2 + right_fit[2] * ploty + right_fit[3]
    # Create an image to draw the lines on
    mask_track = np.zeros_like(img_binary_trans).astype(np.uint8)
    color_warp = np.dstack((mask_track, mask_track, mask_track))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))  # of shape (1, pt_cnt, 2)

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    cv2.fillPoly(mask_track, np.int_([pts]), 255)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    img = np.array(img)
    newwarp = cv2.warpPerspective(color_warp, trans_dst_2_src, (img.shape[1], img.shape[0]))
    newwarp, mask_expand = expand(newwarp)
    mask_track = cv2.warpPerspective(mask_track, trans_dst_2_src, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    return result, newwarp, mask_track, mask_expand
