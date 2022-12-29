import cv2
import numpy as np
from matplotlib import pyplot as plt

import load_images
import region_growing

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


# =====1===== all

# init pyplot
fig, _ax = plt.subplots(3, 4, figsize=(20, 15))
ax = _ax.flatten()

thresh = [10, 10, 75]
start_loc = ["max", "min", "min"]
for img_idx, img in enumerate(IMGS):
    img_thresh = thresh[img_idx]
    img_start_loc = start_loc[img_idx]
    ax_ori = ax[4 * img_idx]
    img_title_idx_str = IMG_TITLES[img_idx]
    # show original img
    ax_ori.imshow(img, cmap="gray")
    ax_ori.set_xticks([]), ax_ori.set_yticks([])
    ax_ori.set_xlabel("(%s-1) Original, using Loc: %s" % (img_title_idx_str, img_start_loc.upper()))
    # step-by-step
    res_steps_titles, res_steps_imgs, res_start_loc, res_sim_history = region_growing.region_growing_step_by_step(
        img=img, thresh=img_thresh, start_loc=img_start_loc)
    _, _, _ = ax[4 * img_idx + 1].hist(res_sim_history, 20, density=True, alpha=0.75)
    ax[4 * img_idx + 1].set_xlabel("Similarity Histogram")
    for _step_idx in range(len(res_steps_imgs)):
        _title, _img = res_steps_titles[_step_idx], res_steps_imgs[_step_idx]
        _ax_idx = 4 * img_idx + 2 + _step_idx
        ax[_ax_idx].imshow(_img, cmap="gray")
        ax[_ax_idx].set_xticks([]), ax[_ax_idx].set_yticks([])
        ax[_ax_idx].set_xlabel(_title, color="black")
    print("IMG #%d Done" % img_idx)

plt.tight_layout()
if SAVE is False:
    plt.show()
else:
    res_fn = "res/3-growing-1.png"
    plt.savefig(res_fn, dpi=200)
    print("Result Saved: %s" % res_fn)
plt.clf()

# =====2===== different thresh

# init pyplot
fig, _ax = plt.subplots(3, 6, figsize=(30, 15))
ax = _ax.flatten()

thresh_set = [
    [6, 10, 14, 18, 22],
    [3, 5, 10, 15, 20],
    [65, 70, 75, 80, 85],
]
start_loc = ["max", "min", "min"]
for img_idx, img in enumerate(IMGS):
    img_thresh_set = thresh_set[img_idx]
    img_start_loc = start_loc[img_idx]
    ax_ori = ax[6 * img_idx]
    img_title_idx_str = IMG_TITLES[img_idx]
    # show original img
    ax_ori.imshow(img, cmap="gray")
    ax_ori.set_xticks([]), ax_ori.set_yticks([])
    # ax_ori.set_xlabel("(%s-1) Original" % img_title_idx_str)
    # step-by-step
    for _thresh_idx, _thresh in enumerate(img_thresh_set):
        res_steps_titles, res_steps_imgs, res_start_loc, res_sim_history = region_growing.region_growing_step_by_step(
            img=img, thresh=_thresh, start_loc=img_start_loc)
        _title_mask = res_steps_titles[0]
        _img_mask = res_steps_imgs[0]
        _ax_idx = 6 * img_idx + 1 + _thresh_idx
        ax[_ax_idx].imshow(_img_mask, cmap="gray")
        ax[_ax_idx].set_xticks([]), ax[_ax_idx].set_yticks([])
        ax[_ax_idx].set_xlabel(_title_mask, color="black")
    ax_ori.set_xlabel("(%s-1) Original (Loc=%d,%d)" % (img_title_idx_str, res_start_loc[0], res_start_loc[1]))
    print("IMG #%d Done" % img_idx)

fig.suptitle("Loc: %s" % img_start_loc.upper())
plt.tight_layout()
if SAVE is False:
    plt.show()
else:
    res_fn = "res/3-growing-2.png"
    plt.savefig(res_fn, dpi=200)
    print("Result Saved: %s" % res_fn)
plt.clf()
