import cv2
import numpy as np
from matplotlib import pyplot as plt

import load_images
import edge_detection

DEBUG = False
SAVE = (DEBUG is False) and True
PLT_FONT_SIZE_FIGURE_TITLE = 30
PLT_FONT_SIZE_SUBPLOT_TITLE = 20
PLT_FONT_SIZE_SUBPLOT_XLABEL = 20
plt.rcParams.update({"figure.titlesize": PLT_FONT_SIZE_FIGURE_TITLE})
plt.rcParams.update({"axes.titlesize": PLT_FONT_SIZE_SUBPLOT_TITLE})
plt.rcParams.update({"axes.labelsize": PLT_FONT_SIZE_SUBPLOT_XLABEL})

IMGS = load_images.load_img_set()
IMG_TITLES = ["a", "b", "c"]
# if DEBUG is True:
#     IMGS = IMGS[:1]

# gaussian kernels
GAUSS_KERNEL_SIZES = [5, 7, 11, 15, 19]
GAUSS_SIGMA = 1.0
# double thresh
THRESH_LOW = 50
THRESH_HIGH = 100

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
        img_title = r"(%s-%d) Canny ($%d\times%d, \sigma=%.1f$)" \
                    % (img_title_idx_str, _size_idx + 1, _size, _size, GAUSS_SIGMA)

        img_proc = edge_detection.canny(
            img=img, as_impl=False,
            blur_size=_size, blur_sigma=GAUSS_SIGMA,
            thresh_low=THRESH_LOW, thresh_high=THRESH_HIGH
        )
        ax_proc.imshow(img_proc, cmap="gray")
        ax_proc.set_xticks([]), ax_proc.set_yticks([])
        ax_proc.set_xlabel(img_title)

fig.suptitle("Various Gaussian Kernel Sizes. Thresh = %d/%d" % (THRESH_LOW, THRESH_HIGH))
plt.tight_layout()
if SAVE is False:
    plt.show()
else:
    res_fn = "res/2-4-canny-1.png"
    plt.savefig(res_fn, dpi=200)
    print("Result Saved: %s" % res_fn)
plt.clf()

# =====2===== different thresh

# init pyplot
fig, _ax = plt.subplots(3, 4, figsize=(20, 15))
ax = _ax.flatten()

_crt_blur_size = 11
for img_idx, img in enumerate(IMGS):
    ax_ori = ax[4 * img_idx]
    img_title_idx_str = IMG_TITLES[img_idx]
    # show original img
    ax_ori.imshow(img, cmap="gray")
    ax_ori.set_xticks([]), ax_ori.set_yticks([])
    ax_ori.set_xlabel("(%s-1) Original" % img_title_idx_str)
    # show processed img
    #   low + low
    ax_proc = ax[4 * img_idx + 1]
    img_title = r"(%s-2) Canny (%d/%d)" \
                % (img_title_idx_str, THRESH_LOW, THRESH_LOW)
    img_proc = edge_detection.canny(
        img=img, as_impl=False,
        blur_size=_crt_blur_size, blur_sigma=GAUSS_SIGMA,
        thresh_low=THRESH_LOW, thresh_high=THRESH_LOW
    )
    ax_proc.imshow(img_proc, cmap="gray")
    ax_proc.set_xticks([]), ax_proc.set_yticks([])
    ax_proc.set_xlabel(img_title)
    #   high + high
    ax_proc = ax[4 * img_idx + 2]
    img_title = r"(%s-3) Canny (%d/%d)" \
                % (img_title_idx_str, THRESH_HIGH, THRESH_HIGH)
    img_proc = edge_detection.canny(
        img=img, as_impl=False,
        blur_size=_crt_blur_size, blur_sigma=GAUSS_SIGMA,
        thresh_low=THRESH_HIGH, thresh_high=THRESH_HIGH
    )
    ax_proc.imshow(img_proc, cmap="gray")
    ax_proc.set_xticks([]), ax_proc.set_yticks([])
    ax_proc.set_xlabel(img_title)
    #   low + high
    ax_proc = ax[4 * img_idx + 3]
    img_title = r"(%s-4) Canny (%d/%d)" \
                % (img_title_idx_str, THRESH_LOW, THRESH_HIGH)
    img_proc = edge_detection.canny(
        img=img, as_impl=False,
        blur_size=_crt_blur_size, blur_sigma=GAUSS_SIGMA,
        thresh_low=THRESH_LOW, thresh_high=THRESH_HIGH
    )
    ax_proc.imshow(img_proc, cmap="gray")
    ax_proc.set_xticks([]), ax_proc.set_yticks([])
    ax_proc.set_xlabel(img_title)

fig.suptitle(r"Various Thresh Selection. Gaussian Blur $%d\times%d, \sigma=%.1f$"
             % (_crt_blur_size, _crt_blur_size, GAUSS_SIGMA))
plt.tight_layout()
if SAVE is False:
    plt.show()
else:
    res_fn = "res/2-4-canny-2.png"
    plt.savefig(res_fn, dpi=200)
    print("Result Saved: %s" % res_fn)
plt.clf()

# =====3===== step-by-step thresh

# init pyplot
fig, _ax = plt.subplots(3, 6, figsize=(30, 15))
ax = _ax.flatten()

_crt_blur_size = 11
for img_idx, img in enumerate(IMGS):
    ax_ori = ax[6 * img_idx]
    img_title_idx_str = IMG_TITLES[img_idx]
    # show original img
    ax_ori.imshow(img, cmap="gray")
    ax_ori.set_xticks([]), ax_ori.set_yticks([])
    ax_ori.set_xlabel("(%s-1) Original" % img_title_idx_str)
    # show processed img
    res_steps_titles, res_steps_imgs = edge_detection._canny_step_by_step(
        img=img, as_impl=False,
        blur_size=_crt_blur_size, blur_sigma=GAUSS_SIGMA,
        thresh_low=THRESH_LOW, thresh_high=THRESH_HIGH
    )
    for _step_idx in range(len(res_steps_imgs)):
        _title, _img = res_steps_titles[_step_idx], res_steps_imgs[_step_idx]
        if "OpenCV - " in _title:
            _title = _title.replace("OpenCV - ", "")
        _title = "(%s-%d) %s" % (img_title_idx_str, _step_idx + 2, _title)
        _ax_proc = ax[6 * img_idx + 1 + _step_idx]
        _ax_proc.imshow(_img, cmap="gray")
        _ax_proc.set_xticks([]), _ax_proc.set_yticks([])
        _ax_proc.set_xlabel(_title, color="black")

fig.suptitle(r"Canny Step-by-Step ($%d\times%d, \sigma=%.1f$; %d/%d)" \
             % (_crt_blur_size, _crt_blur_size, GAUSS_SIGMA, THRESH_LOW, THRESH_HIGH))
plt.tight_layout()
if SAVE is False:
    plt.show()
else:
    res_fn = "res/2-4-canny-3.png"
    plt.savefig(res_fn, dpi=200)
    print("Result Saved: %s" % res_fn)
plt.clf()
