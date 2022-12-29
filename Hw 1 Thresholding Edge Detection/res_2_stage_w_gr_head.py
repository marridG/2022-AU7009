import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

from thresh import cal_thresh, gen_gray_scale_histogram

img_fn = "images/22.bmp"
img_gr_fns = ["images/22_gr_1.PNG", "images/22_gr_2.PNG", "images/22_gr_3.PNG"]
img_gr_labels = ["Background", "Soft Tissue", "Bone"]
NUM_CLUSTER = 2

PLT_FONT_SIZE_SUBPLOT_TITLE = 20
PLT_FONT_SIZE_SUBPLOT_XLABEL = 20

assert os.path.exists(img_fn)
assert all([os.path.exists(_img_gr_fn) for _img_gr_fn in img_gr_fns])

img = cv2.imread(img_fn, cv2.IMREAD_GRAYSCALE)
# view histogram
img_gray_histogram = gen_gray_scale_histogram(img=img)

# thresh selection by clustering
thresh_list = cal_thresh(img=img, num_clusters=NUM_CLUSTER)  # , **{"canny_thresh_1": 100, "canny_thresh_2": 200})
thresh_range_list = [0] + list(thresh_list) + [255]
img_res_cluster_0 = cv2.inRange(img, 0 * 1., thresh_list[0] * 1.)
img_res_cluster_1 = cv2.inRange(img, thresh_list[0] * 1., thresh_list[1] * 1.)
img_res_cluster_2 = cv2.inRange(img, thresh_list[1] * 1., 255 * 1.)
img_res_list = [img_res_cluster_0, 255 - img_res_cluster_1, 255 - img_res_cluster_2]

# === show plots (with merged subplots) ===
fig, _ax = plt.subplots(3, NUM_CLUSTER + 1, figsize=(5 * (NUM_CLUSTER + 1), 15))
gs = _ax[0, 1].get_gridspec()
for ax in _ax[0, 1:]:  # remove the underlying axes
    ax.remove()
axbig = fig.add_subplot(gs[0, 1:])
ax = _ax.flatten()

# [0] original: gray scale
# _, img_binary = cv2.threshold(img, thresh=127, maxval=255, type=thresh_schema["ori"])
ax[0].imshow(img, cmap="gray")
# ax[0].set_title("Original", fontsize=PLT_FONT_SIZE_SUBPLOT_TITLE)
ax[0].set_xlabel("(a-1) Original", fontsize=PLT_FONT_SIZE_SUBPLOT_XLABEL)
ax[0].set_xticks([]), ax[0].set_yticks([])

# [1-2] histogram
axbig.bar(range(256), img_gray_histogram, color="black")
axbig.set_xlabel("(a-2) Original - Gray Scale Histogram ($y$-axis Truncated @3k)", fontsize=PLT_FONT_SIZE_SUBPLOT_XLABEL)
# axbig.set_xticks([])
axbig.set_yticks([])
axbig.set_xlim(0, 255), axbig.set_ylim(0, 3000)

# [3-5] ground truth: gray scale -> binary
for _gr_idx, _gr_fn in enumerate(img_gr_fns):
    _ax_idx = 3 + _gr_idx
    _ax_label = "(b-%d) Paper - %s" % (_gr_idx + 1, img_gr_labels[_gr_idx])

    _img_gr = cv2.imread(_gr_fn, cv2.IMREAD_GRAYSCALE)
    _, _img_gr_binary = cv2.threshold(_img_gr, thresh=127, maxval=255, type=cv2.THRESH_BINARY_INV)
    ax[_ax_idx].imshow(_img_gr_binary, cmap="binary")
    # ax[_ax_idx].set_title(_ax_label, fontsize=PLT_FONT_SIZE_SUBPLOT_TITLE)
    ax[_ax_idx].set_xlabel(_ax_label, fontsize=PLT_FONT_SIZE_SUBPLOT_XLABEL)
    ax[_ax_idx].set_xticks([]), ax[_ax_idx].set_yticks([])

# [6-8] ours: binary
for _res_idx, _res in enumerate(img_res_list):
    _ax_idx = 6 + _res_idx
    _ax_label = "(c-%d) Ours - %s\n(Thresh=%d-%d)" % (
        _res_idx + 1, img_gr_labels[_res_idx],
        thresh_range_list[_res_idx], thresh_range_list[_res_idx + 1]
    )

    ax[_ax_idx].imshow(_res, cmap="binary")
    # ax[_ax_idx].set_title(_ax_label, fontsize=PLT_FONT_SIZE_SUBPLOT_TITLE)
    ax[_ax_idx].set_xlabel(_ax_label, fontsize=PLT_FONT_SIZE_SUBPLOT_XLABEL)
    ax[_ax_idx].set_xticks([]), ax[_ax_idx].set_yticks([])
# ax[-1].imshow(img_copy, cmap="gray")

plt.tight_layout()
plt.show()
