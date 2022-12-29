import os
import cv2
from matplotlib import pyplot as plt

from thresh import cal_thresh

img_fns = ["images/1_gray.bmp", "images/13.bmp", ]
img_gr_fns = ["images/1_gray_gr.PNG", "images/13_gr.PNG", ]
thresh_schema = [
    {"ori": cv2.THRESH_BINARY_INV, "gr": cv2.THRESH_BINARY_INV, "ours": cv2.THRESH_BINARY_INV},
    {"ori": cv2.THRESH_BINARY, "gr": cv2.THRESH_BINARY_INV, "ours": cv2.THRESH_BINARY}
]

PLT_FONT_SIZE_SUBPLOT_TITLE = 20
PLT_FONT_SIZE_SUBPLOT_XLABEL = 20

for _img_fn, _img_gr_fn, _thresh_schema in zip(img_fns, img_gr_fns, thresh_schema):
    assert os.path.exists(_img_fn)
    assert os.path.exists(_img_gr_fn)

    _img = cv2.imread(_img_fn, cv2.IMREAD_GRAYSCALE)
    _thresh = cal_thresh(img=_img)
    _, _img_thresh = cv2.threshold(_img, thresh=_thresh, maxval=255, type=_thresh_schema["ours"])

    # === show plots ===
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    # original: gray scale
    # _, _img_binary = cv2.threshold(_img, thresh=127, maxval=255, type=_thresh_schema["ori"])
    ax[0].imshow(_img, cmap="gray")
    # ax[0].set_title("Original", fontsize=PLT_FONT_SIZE_SUBPLOT_TITLE)
    ax[0].set_xlabel("(a) Original", fontsize=PLT_FONT_SIZE_SUBPLOT_XLABEL)
    ax[0].set_xticks([]), ax[0].set_yticks([])
    # ground truth: gray scale -> binary
    _img_gr = cv2.imread(_img_gr_fn, cv2.IMREAD_GRAYSCALE)
    _, _img_gr_binary = cv2.threshold(_img_gr, thresh=127, maxval=255, type=_thresh_schema["gr"])
    ax[1].imshow(_img_gr_binary, cmap="binary")
    # ax[1].set_title("Paper", fontsize=PLT_FONT_SIZE_SUBPLOT_TITLE)
    ax[1].set_xlabel("(b) Paper", fontsize=PLT_FONT_SIZE_SUBPLOT_XLABEL)
    ax[1].set_xticks([]), ax[1].set_yticks([])
    # ours: binary
    ax[2].imshow(_img_thresh, cmap="binary")
    # ax[2].set_title("Ours (Thresh=%d)" % _thresh, fontsize=PLT_FONT_SIZE_SUBPLOT_TITLE)
    ax[2].set_xlabel("(c) Ours (Thresh=%d)" % _thresh, fontsize=PLT_FONT_SIZE_SUBPLOT_XLABEL)
    ax[2].set_xticks([]), ax[2].set_yticks([])

    plt.tight_layout()
    plt.show()
