import cv2
import numpy as np
from matplotlib import pyplot as plt

import load_images
import split_n_merge

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


# =====1===== thresh

# init pyplot
fig, _ax = plt.subplots(3, 4, figsize=(20, 15))
ax = _ax.flatten()

thresh_set = [5, 10, 15]
min_size = 3
for img_idx, img in enumerate(IMGS):
    ax_ori = ax[4 * img_idx]
    img_title_idx_str = IMG_TITLES[img_idx]
    # show original img
    ax_ori.imshow(img, cmap="gray")
    ax_ori.set_xticks([]), ax_ori.set_yticks([])
    ax_ori.set_xlabel("(%s-1) Original" % img_title_idx_str)
    for _size_idx, _size in enumerate(thresh_set):
        _obj = split_n_merge.SplitNMerge(img=img, thresh_std=_size, min_region_size=min_size)
        _res = _obj.split_n_merge()
        _ax_idx = 4 * img_idx + 1 + _size_idx
        ax[_ax_idx].imshow(_res, cmap="gray")
        ax[_ax_idx].set_xticks([]), ax[_ax_idx].set_yticks([])
        # ax[_ax_idx].set_xlabel(r"(%s-%d) $T_0=%d, c_0=%d$" % (img_title_idx_str, _thresh_idx, _thresh, min_size))
        ax[_ax_idx].set_xlabel(r"(%s-%d) Result Mask ($T_0=%d$)" % (img_title_idx_str, _size_idx, _size))
    print("IMG #%d Done" % img_idx)

fig.suptitle(r"$P(\cdot)=\mathcal{1}_{\sigma\leq T_0}$; Cell$\geq c_0\times c_0$ ($c_0=%d$)" % min_size)
plt.tight_layout()
if SAVE is False:
    plt.show()
else:
    res_fn = "res/4-split_n_merge-1.png"
    plt.savefig(res_fn, dpi=200)
    print("Result Saved: %s" % res_fn)
plt.clf()

# =====1===== cell size

# init pyplot
fig, _ax = plt.subplots(3, 4, figsize=(20, 15))
ax = _ax.flatten()

thresh = 10
min_size_set = [3, 5, 7]
for img_idx, img in enumerate(IMGS):
    ax_ori = ax[4 * img_idx]
    img_title_idx_str = IMG_TITLES[img_idx]
    # show original img
    ax_ori.imshow(img, cmap="gray")
    ax_ori.set_xticks([]), ax_ori.set_yticks([])
    ax_ori.set_xlabel("(%s-1) Original" % img_title_idx_str)
    for _size_idx, _size in enumerate(min_size_set):
        _obj = split_n_merge.SplitNMerge(img=img, thresh_std=thresh, min_region_size=_size)
        _res = _obj.split_n_merge()
        _ax_idx = 4 * img_idx + 1 + _size_idx
        ax[_ax_idx].imshow(_res, cmap="gray")
        ax[_ax_idx].set_xticks([]), ax[_ax_idx].set_yticks([])
        # ax[_ax_idx].set_xlabel(r"(%s-%d) $T_0=%d, c_0=%d$" % (img_title_idx_str, _thresh_idx, _thresh, min_size))
        ax[_ax_idx].set_xlabel(r"(%s-%d) Result Mask ($c_0=%d$)" % (img_title_idx_str, _size_idx, _size))
    print("IMG #%d Done" % img_idx)

fig.suptitle(r"$P(\cdot)=\mathcal{1}_{\sigma\leq T_0}$ ($T_0=%d$); Cell$\geq c_0\times c_0$" % thresh)
plt.tight_layout()
if SAVE is False:
    plt.show()
else:
    res_fn = "res/4-split_n_merge-2.png"
    plt.savefig(res_fn, dpi=200)
    print("Result Saved: %s" % res_fn)
plt.clf()
