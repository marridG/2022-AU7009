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
# if DEBUG is True:
#     IMGS = IMGS[:1]

# # gaussian kernels
# GAUSS_KERNEL_SIZES = [5, 7, 11, 15, 19]
# GAUSS_SIGMA = 1.0

# init pyplot
fig, _ax = plt.subplots(2, 3, figsize=(15, 10))
ax = _ax.flatten()

for img_idx, img in enumerate(IMGS):
    ax_ori = ax[0 + img_idx]
    ax_proc = ax[3 + img_idx]
    img_title_idx = img_idx + 1

    # show original img
    ax_ori.imshow(img, cmap="gray")
    ax_ori.set_xticks([]), ax_ori.set_yticks([])
    ax_ori.set_xlabel("(a-%d) Original" % img_title_idx)

    # show processed img
    img_proc = edge_detection.laplacian(img=img, as_impl=False)
    ax_proc.imshow(img_proc, cmap="gray")
    ax_proc.set_xticks([]), ax_proc.set_yticks([])
    ax_proc.set_xlabel("(b-%d) Laplacian of (a-%d)" % (img_title_idx, img_title_idx))

plt.tight_layout()
if SAVE is False:
    plt.show()
else:
    res_fn = "res/2-2-laplacian-1.png"
    plt.savefig(res_fn, dpi=200)
    print("Result Saved: %s" % res_fn)
