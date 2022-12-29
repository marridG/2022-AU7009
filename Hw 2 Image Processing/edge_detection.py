from typing import List
from collections import deque
import cv2
import numpy as np
from matplotlib import pyplot as plt

import load_images
import gaussian_blur

DEBUG = False
SAVE = (DEBUG is False) and True

SOBEL_KERNEL_HORI = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
SOBEL_KERNEL_VERT = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
LAPLACIAN_KERNEL_BASE = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

PLT_FONT_SIZE_SUBPLOT_TITLE = 20
PLT_FONT_SIZE_SUBPLOT_XLABEL = 20
plt.rcParams.update({"axes.titlesize": PLT_FONT_SIZE_SUBPLOT_TITLE})
plt.rcParams.update({"axes.labelsize": PLT_FONT_SIZE_SUBPLOT_XLABEL})


# ========== SOBEL ==========

def _sobel_opencv_step_by_step(img: np.ndarray) -> (List[str], List[np.ndarray]):
    res_titles = ["OpenCV - Grad X", "OpenCV - Grad Y", "OpenCV - Sobel"]
    # res_imgs = []

    grad_x = cv2.Sobel(src=img, ddepth=cv2.CV_16S, dx=1, dy=0, ksize=3, scale=1, delta=0)
    grad_y = cv2.Sobel(src=img, ddepth=cv2.CV_16S, dx=0, dy=1, ksize=3, scale=1, delta=0)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    res = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    res_imgs = [grad_x, grad_y, res]
    return res_titles, res_imgs


def _sobel_step_by_step(img: np.ndarray) -> (List[str], List[np.ndarray]):
    res_titles = ["Impl. - Grad X", "Impl. - Grad Y", "Impl. - Sobel"]
    # res_imgs = []

    grad_hori = cv2.filter2D(src=img, kernel=SOBEL_KERNEL_HORI, ddepth=cv2.CV_16S, delta=0)
    grad_vert = cv2.filter2D(src=img, kernel=SOBEL_KERNEL_VERT, ddepth=cv2.CV_16S, delta=0)
    abs_grad_hori = cv2.convertScaleAbs(grad_hori)
    abs_grad_vert = cv2.convertScaleAbs(grad_vert)
    res = cv2.addWeighted(abs_grad_hori, 0.5, abs_grad_vert, 0.5, 0)

    res_imgs = [grad_hori, grad_vert, res]
    return res_titles, res_imgs


def sobel(img: np.ndarray, as_impl: bool = True) -> np.ndarray:
    # implemented
    if as_impl is True:
        _, res = _sobel_step_by_step(img=img)
        res = res[-1]
    # OpenCV used
    else:
        _, res = _sobel_opencv_step_by_step(img=img)
        res = res[-1]
    return res


def sobel_illustrate(img: np.ndarray):
    # init pyplot
    fig, _ax = plt.subplots(2, 4, figsize=(20, 10))
    gs = _ax[0, 0].get_gridspec()
    for ax in _ax[0:, 0]:  # remove the underlying axes
        ax.remove()
    ax_big = fig.add_subplot(gs[0:, 0])
    ax = _ax.flatten()
    # show original img
    ax_big.imshow(img, cmap="gray")
    ax_big.set_xticks([]), ax_big.set_yticks([])
    ax_big.set_xlabel("Original", color="red")
    # opencv
    res_steps_titles, res_steps_imgs = _sobel_opencv_step_by_step(img=img)
    for _step_idx in range(len(res_steps_imgs)):
        _title, _img = res_steps_titles[_step_idx], res_steps_imgs[_step_idx]
        _ax_idx = 1 + _step_idx
        ax[_ax_idx].imshow(_img, cmap="gray")
        ax[_ax_idx].set_xticks([]), ax[_ax_idx].set_yticks([])
        ax[_ax_idx].set_xlabel(_title, color="black" if _step_idx < len(res_steps_imgs) - 1 else "red")
    # implemented
    res_steps_titles, res_steps_imgs = _sobel_step_by_step(img=img)
    for _step_idx in range(len(res_steps_imgs)):
        _title, _img = res_steps_titles[_step_idx], res_steps_imgs[_step_idx]
        _ax_idx = 5 + _step_idx
        ax[_ax_idx].imshow(_img, cmap="gray")
        ax[_ax_idx].set_xticks([]), ax[_ax_idx].set_yticks([])
        ax[_ax_idx].set_xlabel(_title, color="black" if _step_idx < len(res_steps_imgs) - 1 else "red")

    plt.tight_layout()
    if SAVE is False:
        plt.show()
    else:
        res_fn = "res/2-1-sobel-0.png"
        plt.savefig(res_fn, dpi=200)
        print("Result Saved: %s" % res_fn)
    plt.clf()


# ========== LAPLACIAN ==========

def _laplacian_opencv_step_by_step(img: np.ndarray) -> (List[str], List[np.ndarray]):
    res_titles = ["OpenCV - Laplacian"]
    # res_imgs = []

    res = cv2.Laplacian(src=img, ddepth=cv2.CV_16S, ksize=3)
    res = cv2.convertScaleAbs(res)

    res_imgs = [res]
    return res_titles, res_imgs


def _laplacian_step_by_step(img: np.ndarray, kernel_scale: int = 4) -> (List[str], List[np.ndarray]):
    res_titles = [r"Impl. - Laplacian ($\times%d$)" % kernel_scale]
    # res_imgs = []

    grad_laplacian = cv2.filter2D(src=img, kernel=LAPLACIAN_KERNEL_BASE * kernel_scale, ddepth=cv2.CV_16S, delta=0)
    res = cv2.convertScaleAbs(grad_laplacian)

    res_imgs = [res]
    return res_titles, res_imgs


def laplacian(img: np.ndarray, as_impl: bool = True) -> np.ndarray:
    # implemented
    if as_impl is True:
        _, res = _laplacian_step_by_step(img=img, kernel_scale=4)
        res = res[-1]
    # OpenCV used
    else:
        _, res = _laplacian_opencv_step_by_step(img=img)
        res = res[-1]
    return res


def laplacian_illustrate(img: np.ndarray, impl_kernel_scale: int = 4):
    # init pyplot
    fig, _ax = plt.subplots(2, 3, figsize=(15, 10))
    ax = _ax.flatten()
    # show original img
    ax[0].imshow(img, cmap="gray")
    ax[0].set_xticks([]), ax[0].set_yticks([])
    ax[0].set_xlabel("Original", color="red")
    # implemented - selected
    _, res_steps_imgs = _laplacian_step_by_step(img=img, kernel_scale=impl_kernel_scale)
    ax[1].imshow(res_steps_imgs[-1], cmap="gray")
    ax[1].set_xticks([]), ax[1].set_yticks([])
    ax[1].set_xlabel(r"Impl. - Laplacian (selected=$\times%d$)" % impl_kernel_scale, color="red")
    # opencv
    res_steps_titles, res_steps_imgs = _laplacian_opencv_step_by_step(img=img)
    ax[2].imshow(res_steps_imgs[-1], cmap="gray")
    ax[2].set_xticks([]), ax[2].set_yticks([])
    ax[2].set_xlabel(res_steps_titles[-1], color="red")
    # implemented
    for _ratio_idx, _ratio in enumerate([1, 2, 4]):
        _res_steps_titles, _res_steps_imgs = _laplacian_step_by_step(img=img, kernel_scale=_ratio)
        _ax_idx = 3 + _ratio_idx
        ax[_ax_idx].imshow(_res_steps_imgs[-1], cmap="gray")
        ax[_ax_idx].set_xticks([]), ax[_ax_idx].set_yticks([])
        ax[_ax_idx].set_xlabel(_res_steps_titles[-1])

    plt.tight_layout()
    if SAVE is False:
        plt.show()
    else:
        res_fn = "res/2-2-laplacian-0.png"
        plt.savefig(res_fn, dpi=200)
        print("Result Saved: %s" % res_fn)
    plt.clf()


# ========== LOG ==========

def _log_opencv_step_by_step(img: np.ndarray,
                             blur_kernel_size: int = 3, blur_kernel_sigma: float = 1.0) \
        -> (List[str], List[np.ndarray]):
    res_titles = [r"OpenCV - Blurred ($3\times%d, \sigma=%.1f$)" % (blur_kernel_size, blur_kernel_sigma),
                  "OpenCV - LoG"]
    # res_imgs = []

    blurred = cv2.GaussianBlur(src=img, ksize=(blur_kernel_size, blur_kernel_size),
                               sigmaX=blur_kernel_sigma, sigmaY=0.)
    res = cv2.Laplacian(src=blurred, ddepth=cv2.CV_16S, ksize=3)
    res = cv2.convertScaleAbs(res)

    res_imgs = [blurred, res]
    return res_titles, res_imgs


def _log_step_by_step(img: np.ndarray,
                      blur_kernel_size: int = 3, blur_kernel_sigma: float = 1.0,
                      laplacian_kernel_scale: int = 4) \
        -> (List[str], List[np.ndarray]):
    res_titles = [r"Impl. - Blurred ($3\times%d, \sigma=%.1f$)" % (blur_kernel_size, blur_kernel_sigma),
                  r"Impl. - LoG (Laplacian=$\times%d$)" % laplacian_kernel_scale]
    # res_imgs = []

    blur_kernel = gaussian_blur.get_2d_gaussian_kernel(size=blur_kernel_size, sigma=blur_kernel_sigma)
    blurred = cv2.filter2D(src=img, kernel=blur_kernel, ddepth=-1, delta=0)
    grad_laplacian = cv2.filter2D(src=blurred, kernel=LAPLACIAN_KERNEL_BASE * laplacian_kernel_scale,
                                  ddepth=cv2.CV_16S, delta=0)
    res = cv2.convertScaleAbs(grad_laplacian)

    res_imgs = [blurred, res]
    return res_titles, res_imgs


def log(img: np.ndarray, as_impl: bool = True,
        blur_kernel_size: int = 3, blur_kernel_sigma: float = 1.0,
        laplacian_kernel_scale: int = 4) \
        -> np.ndarray:
    # implemented
    if as_impl is True:
        _, res = _log_step_by_step(
            img=img,
            blur_kernel_size=blur_kernel_size, blur_kernel_sigma=blur_kernel_sigma,
            laplacian_kernel_scale=laplacian_kernel_scale
        )
        res = res[-1]
    # OpenCV used
    else:
        _, res = _log_opencv_step_by_step(
            img=img,
            blur_kernel_size=blur_kernel_size, blur_kernel_sigma=blur_kernel_sigma
        )
        res = res[-1]
    return res


def log_illustrate(img: np.ndarray,
                   blur_kernel_size: int = 3, blur_kernel_sigma: float = 1.0,
                   laplacian_kernel_scale: int = 4):
    # init pyplot
    fig, _ax = plt.subplots(2, 3, figsize=(15, 10))
    gs = _ax[0, 0].get_gridspec()
    for ax in _ax[0:, 0]:  # remove the underlying axes
        ax.remove()
    ax_big = fig.add_subplot(gs[0:, 0])
    ax = _ax.flatten()
    # show original img
    ax_big.imshow(img, cmap="gray")
    ax_big.set_xticks([]), ax_big.set_yticks([])
    ax_big.set_xlabel("Original", color="red")
    # opencv
    res_steps_titles, res_steps_imgs = _log_opencv_step_by_step(
        img=img,
        blur_kernel_size=blur_kernel_size, blur_kernel_sigma=blur_kernel_sigma
    )
    for _step_idx in range(len(res_steps_imgs)):
        _title, _img = res_steps_titles[_step_idx], res_steps_imgs[_step_idx]
        _ax_idx = 1 + _step_idx
        ax[_ax_idx].imshow(_img, cmap="gray")
        ax[_ax_idx].set_xticks([]), ax[_ax_idx].set_yticks([])
        ax[_ax_idx].set_xlabel(_title, color="black" if _step_idx < len(res_steps_imgs) - 1 else "red")
    # implemented
    res_steps_titles, res_steps_imgs = _log_step_by_step(
        img=img,
        blur_kernel_size=blur_kernel_size, blur_kernel_sigma=blur_kernel_sigma,
        laplacian_kernel_scale=laplacian_kernel_scale
    )
    for _step_idx in range(len(res_steps_imgs)):
        _title, _img = res_steps_titles[_step_idx], res_steps_imgs[_step_idx]
        _ax_idx = 4 + _step_idx
        ax[_ax_idx].imshow(_img, cmap="gray")
        ax[_ax_idx].set_xticks([]), ax[_ax_idx].set_yticks([])
        ax[_ax_idx].set_xlabel(_title, color="black" if _step_idx < len(res_steps_imgs) - 1 else "red")

    plt.tight_layout()
    if SAVE is False:
        plt.show()
    else:
        res_fn = "res/2-3-log-0.png"
        plt.savefig(res_fn, dpi=200)
        print("Result Saved: %s" % res_fn)
    plt.clf()


# ========== CANNY ==========

def _canny_1_blur(img: np.ndarray, as_impl: bool = True,
                  size: int = 3, sigma: float = 1.0) \
        -> np.ndarray:
    in_img = img.copy()
    # implemented
    if as_impl is True:
        res = gaussian_blur.gaussian_blur(img=in_img, size=size, sigma=sigma)
    # OpenCV used
    else:
        res = gaussian_blur.gaussian_blur_opencv(img=in_img, size=size, sigma=sigma)
    return res


def _canny_2_gradient(img_blur: np.ndarray, as_impl: bool = True) -> (np.ndarray, np.ndarray, np.ndarray):
    """ using Sobel """
    in_img = img_blur.copy()
    # implemented
    if as_impl is True:
        _, [grad_x, grad_y, res] = _sobel_step_by_step(img=in_img)
    # OpenCV used
    else:
        _, [grad_x, grad_y, res] = _sobel_opencv_step_by_step(img=in_img)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    # return res, grad_x, grad_y
    return res, abs_grad_x, abs_grad_y


def _canny_3_nms(img: np.ndarray,
                 img_grad: np.ndarray, grad_x: np.ndarray, grad_y: np.ndarray) \
        -> np.ndarray:
    assert img.shape == img_grad.shape
    assert img.shape == grad_x.shape
    assert grad_x.shape == grad_y.shape

    in_img = img.copy()
    in_img_grad = img_grad.copy()
    in_grad_x = grad_x.copy()
    in_grad_y = grad_y.copy()

    height, width = in_img_grad.shape
    mask = np.ones_like(in_img_grad)  # 0=suppressed, 1=kept

    in_grad_dir = np.arctan2(in_grad_y, in_grad_x) * 180. / np.pi
    in_grad_dir[np.where(0 > in_grad_dir)] += 180

    def linear_interpolation(g_x, g_y, values):
        if g_y == 0:
            return values[0, 1], values[2, 1]
        if g_x == 0:
            return values[1, 0], values[1, 2]
        if g_y < 0:
            g_x *= -1
            g_y *= -1
        absx = g_x if g_x > 0 else - g_x
        if g_y > absx:
            weight = float(absx) / g_y
            if g_x > 0:
                dtmp1 = weight * values[0, 2] + (1 - weight) * values[0, 1]
                dtmp2 = weight * values[2, 0] + (1 - weight) * values[2, 1]
            else:
                dtmp1 = weight * values[0, 0] + (1 - weight) * values[0, 1]
                dtmp2 = weight * values[2, 2] + (1 - weight) * values[2, 1]
        else:
            weight = float(g_y) / absx
            if g_x > 0:
                dtmp1 = weight * values[0, 2] + (1 - weight) * values[1, 2]
                dtmp2 = weight * values[2, 0] + (1 - weight) * values[1, 0]
            else:
                dtmp1 = weight * values[0, 0] + (1 - weight) * values[1, 0]
                dtmp2 = weight * values[2, 2] + (1 - weight) * values[1, 2]
        '''
        if np.abs(grad_y) > np.abs(grad_x):
            weight = float(np.abs(grad_x)) / np.abs(grad_y)
            dTemp1 = weight * values[0][0] + (1 - weight) * values[0][1]
            dTemp2 = weight * values[2][2] + (1 - weight) * values[2][1]
        else:
            weight = float(np.abs(grad_y)) / np.abs(grad_x)
            dTemp1 = weight * values[2][0] + (1 - weight) * values[1][0]
            dTemp2 = weight * values[0][2] + (1 - weight) * values[1][2]
        '''
        return dtmp1, dtmp2

    for _height in range(1, height - 1):
        for _width in range(1, width - 1):
            # angle 0
            if (0 <= in_grad_dir[_height, _width] < 22.5) or (157.5 <= in_grad_dir[_height, _width] <= 180):
                _neigh_1 = in_img_grad[_height, _width + 1]
                _neigh_2 = in_img_grad[_height, _width - 1]
            # angle 45
            elif 22.5 <= in_grad_dir[_height, _width] < 67.5:
                _neigh_1 = in_img_grad[_height + 1, _width - 1]
                _neigh_2 = in_img_grad[_height - 1, _width + 1]
            # angle 90
            elif 67.5 <= in_grad_dir[_height, _width] < 112.5:
                _neigh_1 = in_img_grad[_height + 1, _width]
                _neigh_2 = in_img_grad[_height - 1, _width]
            # angle 135
            else:  # 112.5 <= grad_dir[_height, _width] < 157.5:
                _neigh_1 = in_img_grad[_height - 1, _width - 1]
                _neigh_2 = in_img_grad[_height + 1, _width + 1]

            # if (img[_height, _width] >= _neigh_1) and (img[_height, _width] >= _neigh_2):
            #     mask[_height, _width] = 1
            # else:
            if (in_img_grad[_height, _width] < _neigh_1) or (in_img_grad[_height, _width] < _neigh_2):
                mask[_height, _width] = 0

            # dtmp1, dtmp2 = linear_interpolation(in_grad_x[_height, _width], in_grad_y[_height, _width],
            #                                     in_img_grad[_height - 1: _height + 2, _width - 1: _width + 2])
            # grad = float(in_img_grad[_height, _width])
            # if dtmp1 > grad or dtmp2 > grad:
            #     mask[_height, _width] = 0

    # suppress by mask
    res = in_img_grad.copy()
    res[np.where(0 == mask)] = 0
    # res[np.where(1 == mask)] = 255
    return res


def _canny_4_double_thresh(img_grad: np.ndarray, img_nms: np.ndarray, as_impl: bool = True,
                           thresh_low: int = 100, thresh_high: int = 200) \
        -> (np.ndarray, np.ndarray, np.ndarray):
    assert img_grad.shape == img_nms.shape

    in_img_grad = img_grad.copy()
    in_img_nms = img_nms.copy()
    strong_val = 255
    weak_val = 128
    # zero_val = 0

    # implemented
    if as_impl is True:
        strong_ref = np.where((in_img_nms > 0) & (in_img_grad >= thresh_high))
        weak_ref = np.where((in_img_nms > 0) & (in_img_grad >= thresh_low))
        res_combined = np.zeros_like(in_img_grad)
        res_combined[strong_ref] = strong_val
        res_combined[weak_ref] = weak_val
        res_low = np.zeros_like(in_img_grad)
        res_low[weak_ref] = 255
        res_low[strong_ref] = 255
        res_high = np.zeros_like(in_img_grad)
        res_high[strong_ref] = 255
    # OpenCV used
    else:
        ret, res_low = cv2.threshold(in_img_grad, thresh_low, 255, cv2.THRESH_BINARY)
        ret, res_high = cv2.threshold(in_img_grad, thresh_high, 255, cv2.THRESH_BINARY)
        res_combined = np.zeros_like(in_img_grad)
        res_combined[np.where(255 == res_low)] = weak_val
        res_combined[np.where(255 == res_high)] = strong_val

    return res_low, res_high, res_combined


def _canny_5_edge_linking(low_img: np.ndarray, high_img: np.ndarray) -> np.ndarray:
    assert low_img.shape == high_img.shape

    height, width = low_img.shape
    mask = high_img.copy()  # 0=suppressed, 255=kept

    d_height = [0, 0, -1, -1, -1, 1, 1, 1]
    d_width = [1, -1, -1, 0, 1, -1, 0, 1]
    for _height in range(1, height - 1):
        for _width in range(1, width - 1):
            if 255 == low_img[_height, _width] and 0 == high_img[_height, _width]:
                # check 8-adj neighbours
                _strong_as_neighbour = False
                for _d_h, _d_w in zip(d_height, d_width):
                    __h = _height + _d_h
                    __w = _width + _d_w
                    if 255 == high_img[__h, __w]:
                        _strong_as_neighbour = True
                        break
                if _strong_as_neighbour is True:
                    mask[_height, _width] = 255

    return mask

    # in_img = img.copy()
    # mask = np.zeros_like(in_img)  # 0=suppressed, 1=kept
    #
    # strong_ref = np.where(255 == high_img)
    # strong_loc = set([(strong_ref[0][i], strong_ref[1][i]) for i in range(len(strong_ref[0]))])
    # weak_ref = np.where(255 == low_img)
    # weak_loc = set([(weak_ref[0][i], weak_ref[1][i]) for i in range(len(weak_ref[0]))])
    #
    # q = deque()
    # d_height = [0, 0, -1, -1, -1, 1, 1, 1]
    # d_width = [1, -1, -1, 0, 1, -1, 0, 1]
    #
    # for (_height, _width) in zip(strong_ref[0], strong_ref[1]):
    #     q.append((_height, _width))
    #     mask[_height, _width] = 1
    # while q:
    #     crt_height, crt_width = q.popleft()
    #     for _d_width, _d_height in zip(d_height, d_width):
    #         _x = crt_height + _d_width
    #         _y = crt_width + _d_height
    #         if (_x, _y) in weak_loc and (_x, _y) not in strong_loc:
    #             mask[_x, _y] = 1
    #             q.append((_x, _y))
    #             strong_loc.add((_x, _y))
    #
    # in_img[np.where(0 == mask)] = 0
    #
    # return in_img


def _canny_step_by_step(img: np.ndarray, as_impl: bool = True,
                        blur_size: int = 3, blur_sigma: float = 1.0,
                        thresh_low: int = 100, thresh_high: int = 200) \
        -> (List[str], List[np.ndarray]):
    _prefix = "Impl. - " if as_impl is True else "OpenCV - "
    res_titles = [r"%sBlur ($3\times%d, \sigma=%.1f$)" % (_prefix, blur_size, blur_sigma),
                  "%sGrad" % _prefix,
                  "NMS",
                  "%sThresh (%d/%d)" % (_prefix, thresh_high, thresh_low),
                  "Hysteresis"]
    res_imgs = []

    # ==1== gaussian blur
    res_1_blur = _canny_1_blur(img=img, as_impl=as_impl,
                               size=blur_size, sigma=blur_sigma)
    res_imgs.append(res_1_blur)
    print("%sCanny - 1 - Gaussian Blur" % _prefix)

    # ==2== gradient calculation (Sobel)
    res_2_grad, res_2_grad_x, res_2_grad_y = _canny_2_gradient(img_blur=res_1_blur, as_impl=as_impl)
    res_imgs.append(res_2_grad)
    print("%sCanny - 2 - Gradient Calculation" % _prefix)

    # ==3== NMS
    res_3_nms = _canny_3_nms(img=img, img_grad=res_2_grad, grad_x=res_2_grad_x, grad_y=res_2_grad_y)
    res_imgs.append(res_3_nms)
    print("* - Canny - 3 - NMS")

    # ==4== double thresholding
    res_4_low, res_4_high, res_4_combined = _canny_4_double_thresh(
        img_grad=res_2_grad, img_nms=res_3_nms, as_impl=True,
        thresh_low=thresh_low, thresh_high=thresh_high
    )
    res_imgs.append(res_4_combined)
    # res_imgs.append(res_4_low)
    # res_imgs.append(res_4_high)
    print("%sCanny - 4 - Double Thresholding" % _prefix)

    # ==5== edge linking
    res_5 = _canny_5_edge_linking(low_img=res_4_low, high_img=res_4_high)
    res_imgs.append(res_5)
    print("* - Canny - 5 - Edge Linking")

    return res_titles, res_imgs


def canny(img: np.ndarray, as_impl: bool = True,
          blur_size: int = 3, blur_sigma: float = 1.0,
          thresh_low: int = 100, thresh_high: int = 200) \
        -> np.ndarray:
    # implemented
    if as_impl is True:
        _, res = _canny_step_by_step(
            img=img, as_impl=False,
            blur_size=blur_size, blur_sigma=blur_sigma,
            thresh_low=thresh_low, thresh_high=thresh_high
        )
        res = res[-1]
    # OpenCV used
    else:
        res = gaussian_blur.gaussian_blur_opencv(img=img, size=blur_size, sigma=blur_sigma)
        res = cv2.Canny(res, thresh_low, thresh_high)
    return res


def canny_illustrate(img: np.ndarray,
                     blur_size: int = 3, blur_sigma: float = 1.0,
                     thresh_low: int = 50, thresh_high: int = 100):
    # init pyplot
    fig, _ax = plt.subplots(3, 5, figsize=(25, 15))
    ax = _ax.flatten()
    # show original img
    ax[0].imshow(img, cmap="gray")
    ax[0].set_xticks([]), ax[0].set_yticks([])
    ax[0].set_xlabel("Original", color="red")
    # end-to-end: OpenCV
    res_gr = cv2.Canny(img, thresh_low, thresh_high)
    ax[4].imshow(res_gr, cmap="gray")
    ax[4].set_xticks([]), ax[4].set_yticks([])
    ax[4].set_xlabel("OpenCV End-to-End (%d/%d)" % (thresh_high, thresh_low), color="red")
    # remove
    for _ax in ax[1:4]:
        _ax.remove()
    # step-by-step: OpenCV
    res_steps_titles, res_steps_imgs = _canny_step_by_step(
        img=img, as_impl=False,
        blur_size=blur_size, blur_sigma=blur_sigma,
        thresh_low=thresh_low, thresh_high=thresh_high
    )
    for _step_idx in range(len(res_steps_imgs)):
        _title, _img = res_steps_titles[_step_idx], res_steps_imgs[_step_idx]
        _ax_idx = 5 + _step_idx
        ax[_ax_idx].imshow(_img, cmap="gray")
        ax[_ax_idx].set_xticks([]), ax[_ax_idx].set_yticks([])
        ax[_ax_idx].set_xlabel(_title, color="black" if _step_idx < len(res_steps_imgs) - 1 else "red")
    # step-by-step: implemented
    res_steps_titles, res_steps_imgs = _canny_step_by_step(
        img=img, as_impl=True,
        blur_size=blur_size, blur_sigma=blur_sigma,
        thresh_low=thresh_low, thresh_high=thresh_high
    )
    for _step_idx in range(len(res_steps_imgs)):
        _title, _img = res_steps_titles[_step_idx], res_steps_imgs[_step_idx]
        _ax_idx = 10 + _step_idx
        ax[_ax_idx].imshow(_img, cmap="gray")
        ax[_ax_idx].set_xticks([]), ax[_ax_idx].set_yticks([])
        ax[_ax_idx].set_xlabel(_title, color="black" if _step_idx < len(res_steps_imgs) - 1 else "red")

    plt.tight_layout()
    if SAVE is False:
        plt.show()
    else:
        res_fn = "res/2-4-canny-0.png"
        plt.savefig(res_fn, dpi=200)
        print("Result Saved: %s" % res_fn)
    plt.clf()


if "__main__" == __name__:
    test_img = load_images.load_img(fn="images/1_gray-2.bmp")
    # test_img = load_images.load_img(fn="images/8_gray.bmp")
    # sobel_illustrate(test_img)
    # laplacian_illustrate(test_img)
    # log_illustrate(test_img)
    canny_illustrate(test_img)
