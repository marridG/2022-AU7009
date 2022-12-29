import cv2
import numpy as np
from matplotlib import pyplot as plt

import load_images
import gaussian_blur

DEBUG = False
SAVE = (DEBUG is False) and True

PLT_FONT_SIZE_SUBPLOT_TITLE = 20
PLT_FONT_SIZE_SUBPLOT_XLABEL = 20
plt.rcParams.update({"axes.titlesize": PLT_FONT_SIZE_SUBPLOT_TITLE})
plt.rcParams.update({"axes.labelsize": PLT_FONT_SIZE_SUBPLOT_XLABEL})

IMGS = load_images.load_img_set()
if DEBUG is True:
    IMGS = IMGS[:1]

# gaussian kernels
GAUSS_KERNEL_SIZES = [5, 7, 11, 15, 19]
GAUSS_SIGMA = 1.0
GAUSS_VIS_FLOAT_DECIMAL = [5, 3, 1, 0, 0]
gauss_kernels = [
    gaussian_blur.get_2d_gaussian_kernel(size=_size, sigma=GAUSS_SIGMA)
    for _size in GAUSS_KERNEL_SIZES
]

for img_idx, img in enumerate(IMGS):

    # init pyplot
    fig, _ax = plt.subplots(2, 6, figsize=(30, 10))
    gs = _ax[0, 0].get_gridspec()
    for ax in _ax[0:, 0]:  # remove the underlying axes
        ax.remove()
    ax_big = fig.add_subplot(gs[0:, 0])
    ax = _ax.flatten()

    # show original img
    ax_big.imshow(img, cmap="gray")
    ax_big.set_xticks([]), ax_big.set_yticks([])
    ax_big.set_xlabel("(a) Original")

    # show Gaussian kernel & blurred img
    for _kernel_idx in range(len(GAUSS_KERNEL_SIZES)):
        _kernel_size = GAUSS_KERNEL_SIZES[_kernel_idx]
        _kernel = gauss_kernels[_kernel_idx]
        _kernel_vis_float_decimal = GAUSS_VIS_FLOAT_DECIMAL[_kernel_idx]
        _ax_idx_kernel = 1 + _kernel_idx
        _ax_idx_blur = (1 + 6) + _kernel_idx

        # show kernel
        gaussian_blur.visualize_gaussian_kernel(
            kernel=_kernel,
            float_decimal=_kernel_vis_float_decimal, target_ax=ax[_ax_idx_kernel])
        ax[_ax_idx_kernel].set_xticks([]), ax[_ax_idx_kernel].set_yticks([])
        ax[_ax_idx_kernel].set_xlabel(r"(b-%d) Kernel ($%d\times%d, \sigma=%.1f$)"
                                      % (_kernel_idx + 1, _kernel_size, _kernel_size, GAUSS_SIGMA))
        # show blurred
        _img_blur = cv2.filter2D(src=img, ddepth=-1, kernel=_kernel)
        # _img_blur = cv2.GaussianBlur(src=img, ksize=(_kernel_size, _kernel_size), sigmaX=GAUSS_SIGMA, sigmaY=0.)
        ax[_ax_idx_blur].imshow(_img_blur, cmap="gray")
        ax[_ax_idx_blur].set_xticks([]), ax[_ax_idx_blur].set_yticks([])
        ax[_ax_idx_blur].set_xlabel(r"(c-%d) Blurred, using (b-%d)" % (_kernel_idx + 1, _kernel_idx + 1))

    plt.tight_layout()
    if SAVE is False:
        plt.show()
    else:
        res_fn = "res/1-gauss-%d.png" % (img_idx + 1)
        plt.savefig(res_fn, dpi=200)
        print("Result Saved: %s" % res_fn)
    plt.clf()
