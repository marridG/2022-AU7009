import cv2
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

import load_images

DEBUG = False
SAVE = (DEBUG is False) and True

PLT_FONT_SIZE_SUBPLOT_TITLE = 20
PLT_FONT_SIZE_SUBPLOT_XLABEL = 20
plt.rcParams.update({"axes.titlesize": PLT_FONT_SIZE_SUBPLOT_TITLE})
plt.rcParams.update({"axes.labelsize": PLT_FONT_SIZE_SUBPLOT_XLABEL})


def get_2d_gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Get a 2D Gaussian kernel of the given size & given sigma
    :param size:            size of the kernel, should be a positive odd number
    :param sigma:           sigma of the Gaussian distribution of both x&y-axis
    :return:                Gaussian kernel of shape (`size`, `size`) and of x-/y-axis sigma `sigma`
    """
    assert 1 == size % 2 and size > 0
    assert sigma >= 0.  # otherwise, NOT implemented

    axis_1d = np.linspace(start=-1. * (size - 1) / 2., stop=1. * (size - 1) / 2., num=size)
    # (NOT NORMED) g(x) = exp(- x^2 / (2 * \sigma^2) )
    gauss_1d = np.exp(-0.5 * np.square(axis_1d) / np.square(sigma))
    gauss_2d = np.outer(gauss_1d, gauss_1d)
    # norm
    res = gauss_2d * 1. / np.sum(gauss_2d)
    return res


def visualize_gaussian_kernel(kernel: np.ndarray, float_decimal: int = 3, target_ax=None) -> None:
    assert kernel.shape[0] == kernel.shape[1]

    size = kernel.shape[0]

    if target_ax is None:
        target_ax = plt
        do_show = True
    else:
        do_show = False

    # draw heatmap
    im = target_ax.imshow(kernel)

    # add text annotations, if required
    if float_decimal > 0:
        for i in range(size):
            for j in range(size):
                text = target_ax.text(x=j, y=i, s="%.*f" % (float_decimal, kernel[i, j]),
                                      ha="center", va="center", color="w")

    if do_show is True:
        target_ax.show()


def gaussian_blur(img: np.ndarray, size: int = 5, sigma: float = 1.0) -> np.ndarray:
    kernel = get_2d_gaussian_kernel(size=size, sigma=sigma)
    res = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    return res


def gaussian_blur_opencv(img: np.ndarray, size: int = 5, sigma: float = 1.0) -> np.ndarray:
    res = cv2.GaussianBlur(src=img, ksize=(size, size), sigmaX=sigma, sigmaY=0.)
    return res


def illustrate(img: np.ndarray, kernel_size: int = 5, kernel_sigma: float = 1.0, vis_float_decimal: int = 5):
    kernel = get_2d_gaussian_kernel(size=kernel_size, sigma=kernel_sigma)
    blur = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    blur_gr = cv2.GaussianBlur(src=img, ksize=(kernel_size, kernel_size), sigmaX=kernel_sigma, sigmaY=0.)

    fig, _ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = _ax.flatten()

    ax[0].imshow(img, cmap="gray")
    ax[0].set_xticks([]), ax[0].set_yticks([])
    ax[0].set_xlabel("Original")

    visualize_gaussian_kernel(kernel=kernel, float_decimal=vis_float_decimal, target_ax=ax[1])
    ax[1].set_xticks([]), ax[1].set_yticks([])
    ax[1].set_xlabel(r"Kernel (Impl.) ($%d\times%d, \sigma=%.1f$)" % (kernel_size, kernel_size, kernel_sigma))

    ax[2].imshow(blur_gr, cmap="gray")
    ax[2].set_xticks([]), ax[2].set_yticks([])
    ax[2].set_xlabel(r"Blurred (OpenCV) ($%d\times%d, \sigma=%.1f$)" % (kernel_size, kernel_size, kernel_sigma))

    ax[3].imshow(blur, cmap="gray")
    ax[3].set_xticks([]), ax[3].set_yticks([])
    ax[3].set_xlabel("Blurred (Impl.)")

    plt.tight_layout()
    if SAVE is False:
        plt.show()
    else:
        res_fn = "res/1-gauss-0.png"
        plt.savefig(res_fn, dpi=200)
        print("Result Saved: %s" % res_fn)
    plt.clf()


if "__main__" == __name__:
    # test_img = load_images.load_img(fn="images/1_gray-2.bmp")
    test_img = load_images.load_img(fn="images/gray高斯噪声3-1.jpg")
    illustrate(img=test_img, kernel_size=5, kernel_sigma=1.0, vis_float_decimal=5)
