import cv2
import numpy as np
from matplotlib import pyplot as plt

import load_images
import edge_detection

DEBUG = False
SAVE = (DEBUG is False) and True
PLT_FONT_SIZE_SUBPLOT_TITLE = 20
PLT_FONT_SIZE_SUBPLOT_XLABEL = 20
plt.rcParams.update({"axes.titlesize": PLT_FONT_SIZE_SUBPLOT_TITLE})
plt.rcParams.update({"axes.labelsize": PLT_FONT_SIZE_SUBPLOT_XLABEL})

IMGS = load_images.load_img_set()
IMG_TITLES = ["a", "b", "c"]
# if DEBUG is True:
#     IMGS = IMGS[:1]

# gaussian kernels
GAUSS_KERNEL_SIZES = [5, 7, 11, 15, 19]
GAUSS_SIGMA = 1.0

# =====1===== gauss kernels

# init pyplot
fig, _ax = plt.subplots(3, 6, figsize=(30, 15))
ax = _ax.flatten()

for img_idx, img in enumerate(IMGS):
    ax_ori = ax[6 * img_idx]
    img_title_idx_str = IMG_TITLES[img_idx]
    # show original img
    ax_ori.imshow(img, cmap="gray")
    ax_ori.set_xticks([]), ax_ori.set_yticks([])
    ax_ori.set_xlabel("(%s-1) Original" % img_title_idx_str)
    # show processed img
    for _size_idx, _size in enumerate(GAUSS_KERNEL_SIZES):
        ax_proc = ax[6 * img_idx + _size_idx + 1]
        img_title = r"(%s-%d) LoG ($%d\times%d, \sigma=%.1f$)" \
                    % (img_title_idx_str, _size_idx + 1, _size, _size, GAUSS_SIGMA)

        img_proc = edge_detection.log(img=img, as_impl=False, blur_kernel_size=_size, blur_kernel_sigma=GAUSS_SIGMA)
        ax_proc.imshow(img_proc, cmap="gray")
        ax_proc.set_xticks([]), ax_proc.set_yticks([])
        ax_proc.set_xlabel(img_title)

plt.tight_layout()
if SAVE is False:
    plt.show()
else:
    res_fn = "res/2-3-log-1.png"
    plt.savefig(res_fn, dpi=200)
    print("Result Saved: %s" % res_fn)
plt.clf()


# =====2===== high thresh

def gen_gray_scale_histogram(in_img: np.ndarray, debug=False, **kwargs) -> np.ndarray:
    height, width = in_img.shape  # (H, W)

    res_histogram = np.zeros(shape=(256,))
    for _height in range(height):
        for _width in range(width):
            _gray_val = in_img[_height, _width]
            res_histogram[_gray_val] += 1

    if debug is True:
        plt.bar(range(256), res_histogram, color="black"), plt.show()
    if kwargs.get("do_norm") is True:
        res_histogram /= in_img.size
    return res_histogram


# init pyplot
fig, _ax = plt.subplots(3, 6, figsize=(30, 15))
ax = _ax.flatten()

for img_idx, img in enumerate(IMGS):
    ax_ori = ax[6 * img_idx]
    img_title_idx_str = IMG_TITLES[img_idx]
    # show original img
    ax_ori.imshow(img, cmap="gray")
    ax_ori.set_xticks([]), ax_ori.set_yticks([])
    ax_ori.set_xlabel("(%s-1) Original" % img_title_idx_str)
    # show processed img
    ax_proc = ax[6 * img_idx + 1]
    img_proc = edge_detection.log(img=img, as_impl=False, blur_kernel_size=5, blur_kernel_sigma=GAUSS_SIGMA)
    ax_proc.imshow(img_proc, cmap="gray")
    ax_proc.set_xticks([]), ax_proc.set_yticks([])
    ax_proc.set_xlabel(r"(%s-2) LoG ($%d\times%d, \sigma=%.1f$)" \
                       % (img_title_idx_str, 5, 5, GAUSS_SIGMA))
    # show histogram
    ax_proc_hist = ax[6 * img_idx + 2]
    img_proc_hist = gen_gray_scale_histogram(in_img=img_proc)
    ax_proc_hist.bar(range(256), img_proc_hist, color="black")
    ax_proc_hist.set_xlabel("(%s-3) Gray Scale Histogram of LoG" % img_title_idx_str)
    # ax_proc_hist.set_xticks([])
    ax_proc_hist.set_yticks([])
    # ax_proc_hist.set_xlim(0, 255), ax_proc_hist.set_ylim(0, 3000)

    # show processed+thresh img
    for _thresh_idx, _thresh in enumerate([30, 60, 90]):
        ax_proc_thresh_low = ax[6 * img_idx + 3 + _thresh_idx]
        _, img_proc_thresh = cv2.threshold(src=img_proc, thresh=_thresh, maxval=255, type=cv2.THRESH_BINARY)
        ax_proc_thresh_low.imshow(img_proc_thresh, cmap="gray")
        ax_proc_thresh_low.set_xticks([]), ax_proc_thresh_low.set_yticks([])
        ax_proc_thresh_low.set_xlabel(r"(%s-%d) LoG + Thresh (%d)" % (img_title_idx_str, 3 + _thresh_idx + 1, _thresh))

plt.tight_layout()
if SAVE is False:
    plt.show()
else:
    res_fn = "res/2-3-log-2.png"
    plt.savefig(res_fn, dpi=200)
    print("Result Saved: %s" % res_fn)
plt.clf()


# =====3===== high thresh + edge connect

def edge_linking(low_img: np.ndarray, high_img: np.ndarray) -> np.ndarray:
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


# init pyplot
fig, _ax = plt.subplots(3, 6, figsize=(30, 15))
ax = _ax.flatten()

for img_idx, img in enumerate(IMGS):
    ax_ori = ax[6 * img_idx]
    img_title_idx_str = IMG_TITLES[img_idx]
    # show original img
    ax_ori.imshow(img, cmap="gray")
    ax_ori.set_xticks([]), ax_ori.set_yticks([])
    ax_ori.set_xlabel("(%s-1) Original" % img_title_idx_str)
    # show processed img
    ax_proc = ax[6 * img_idx + 1]
    img_proc = edge_detection.log(img=img, as_impl=False, blur_kernel_size=5, blur_kernel_sigma=GAUSS_SIGMA)
    ax_proc.imshow(img_proc, cmap="gray")
    ax_proc.set_xticks([]), ax_proc.set_yticks([])
    ax_proc.set_xlabel(r"(%s-2) LoG ($%d\times%d, \sigma=%.1f$)" \
                       % (img_title_idx_str, 5, 5, GAUSS_SIGMA))
    # show histogram
    ax_proc_hist = ax[6 * img_idx + 2]
    img_proc_hist = gen_gray_scale_histogram(in_img=img_proc)
    ax_proc_hist.bar(range(256), img_proc_hist, color="black")
    ax_proc_hist.set_xlabel("(%s-3) Gray Scale Histogram of LoG" % img_title_idx_str)
    # ax_proc_hist.set_xticks([])
    ax_proc_hist.set_yticks([])
    # ax_proc_hist.set_xlim(0, 255), ax_proc_hist.set_ylim(0, 3000)

    # show processed+thresh img
    thresh_low, thresh_high = 60, 90
    _, img_proc_thresh_low = cv2.threshold(src=img_proc, thresh=thresh_low, maxval=255, type=cv2.THRESH_BINARY)
    _, img_proc_thresh_high = cv2.threshold(src=img_proc, thresh=thresh_high, maxval=255, type=cv2.THRESH_BINARY)
    img_proc_thresh_connect = edge_linking(low_img=img_proc_thresh_low, high_img=img_proc_thresh_high)
    ax_proc_thresh_low = ax[6 * img_idx + 3 + 0]
    ax_proc_thresh_low.imshow(img_proc_thresh_low, cmap="gray")
    ax_proc_thresh_low.set_xticks([]), ax_proc_thresh_low.set_yticks([])
    ax_proc_thresh_low.set_xlabel(r"(%s-%d) LoG + Thresh (low=%d)" % (img_title_idx_str, 4, thresh_low))
    ax_proc_thresh_high = ax[6 * img_idx + 3 + 1]
    ax_proc_thresh_high.imshow(img_proc_thresh_high, cmap="gray")
    ax_proc_thresh_high.set_xticks([]), ax_proc_thresh_high.set_yticks([])
    ax_proc_thresh_high.set_xlabel(r"(%s-%d) LoG + Thresh (high=%d)" % (img_title_idx_str, 5, thresh_high))
    ax_proc_thresh_connect = ax[6 * img_idx + 3 + 2]
    ax_proc_thresh_connect.imshow(img_proc_thresh_connect, cmap="gray")
    ax_proc_thresh_connect.set_xticks([]), ax_proc_thresh_connect.set_yticks([])
    ax_proc_thresh_connect.set_xlabel(r"(%s-%d) LoG + Thresh (%d/%d)" % (img_title_idx_str, 6, thresh_low, thresh_high))

plt.tight_layout()
if SAVE is False:
    plt.show()
else:
    res_fn = "res/2-3-log-3.png"
    plt.savefig(res_fn, dpi=200)
    print("Result Saved: %s" % res_fn)
plt.clf()
