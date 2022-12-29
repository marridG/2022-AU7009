import os
import cv2
from matplotlib import pyplot as plt

from thresh import cal_thresh

img_fns = ["images/6.jpg", "images/8_gray.bmp", "images/14.bmp"]
img_titles = ["Handwriting", "Rice", "Fingerprint"]

PLT_FONT_SIZE_SUBPLOT_TITLE = 20
PLT_FONT_SIZE_SUBPLOT_XLABEL = 20

fig, ax = plt.subplots(2, 3, figsize=(15, 10))
ax = ax.flatten()
for _img_idx, _img_fn in enumerate(img_fns):
    _img_title = img_titles[_img_idx]
    _ax_idx_ori = 0 + _img_idx
    _ax_idx_ours = 3 + _img_idx
    _ax_label_idx = _img_idx + 1
    assert os.path.exists(_img_fn)

    _img = cv2.imread(_img_fn, cv2.IMREAD_GRAYSCALE)
    _thresh = cal_thresh(img=_img)
    _, _img_thresh = cv2.threshold(_img, thresh=_thresh, maxval=255, type=cv2.THRESH_BINARY_INV)

    # === show plots ===
    # original: gray scale
    # _, _img_binary = cv2.threshold(_img, thresh=127, maxval=255, type=_thresh_schema["ori"])
    ax[_ax_idx_ori].imshow(_img, cmap="gray")
    # ax[_ax_idx_ori].set_title("Original", fontsize=PLT_FONT_SIZE_SUBPLOT_TITLE)
    ax[_ax_idx_ori].set_xlabel("(a-%d) Original - %s" % (_ax_label_idx, _img_title),
                               fontsize=PLT_FONT_SIZE_SUBPLOT_XLABEL)
    ax[_ax_idx_ori].set_xticks([]), ax[_ax_idx_ori].set_yticks([])
    # ours: binary
    ax[_ax_idx_ours].imshow(_img_thresh, cmap="binary")
    # ax[_ax_idx_ours].set_title("Ours (Thresh=%d)" % _thresh, fontsize=PLT_FONT_SIZE_SUBPLOT_TITLE)
    ax[_ax_idx_ours].set_xlabel("(b-%d) Ours - %s\n(Thresh=%d)" % (_ax_label_idx, _img_title, _thresh),
                                fontsize=PLT_FONT_SIZE_SUBPLOT_XLABEL)
    ax[_ax_idx_ours].set_xticks([]), ax[_ax_idx_ours].set_yticks([])

plt.tight_layout()
plt.show()
