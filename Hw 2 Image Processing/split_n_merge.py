from typing import List
from collections import deque
import cv2
import numpy as np
from matplotlib import pyplot as plt

import load_images
import gaussian_blur

DEBUG = False
SAVE = (DEBUG is False) and True

SOBEL_KERNEL_HORI = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
SOBEL_KERNEL_VERT = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
LAPLACIAN_KERNEL_BASE = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

PLT_FONT_SIZE_SUBPLOT_TITLE = 20
PLT_FONT_SIZE_SUBPLOT_XLABEL = 20
plt.rcParams.update({"axes.titlesize": PLT_FONT_SIZE_SUBPLOT_TITLE})
plt.rcParams.update({"axes.labelsize": PLT_FONT_SIZE_SUBPLOT_XLABEL})


class SplitNMerge:
    def __init__(self, img: np.ndarray, thresh_std: float = 10, min_region_size: int = 3):
        self.img = img.copy()
        self.width, self.height = self.img.shape

        self.thresh = thresh_std
        self.min_region_size = min_region_size

        # results
        self.res_mask = np.zeros_like(self.img)
        self._res_done = False

    def _val_index(self, h0: int, w0: int, dh: int, dw: int):
        assert 0 <= h0 < self.height
        assert 0 <= w0 < self.width
        assert 0 < dh <= self.height - h0
        assert 0 < dw <= self.width - w0

    def _judge_as_distinct(self, h0: int, w0: int, dh: int, dw: int) -> bool:
        self._val_index(h0=h0, w0=w0, dh=dh, dw=dw)

        area = self.img[h0:h0 + dh, w0:w0 + dw]
        std = np.std(area, ddof=1)

        res = std > self.thresh
        return res

    def _split_n_merge(self, h0: int, w0: int, dh: int, dw: int):
        area = self.img[h0:h0 + dh, w0:w0 + dw]
        std = np.std(area, ddof=1)

        # region as a whole is NOT distinct
        if std <= self.thresh:
            self.res_mask[h0:h0 + dh, w0:w0 + dw] = 255
            return

        # region as a whole is distinct
        dh_half = int((dh + 1) // 2)
        dw_half = int((dw + 1) // 2)
        # print(as_distinct, "is true", h0, w0, dh, dw)

        # remove the old region, if 4 smaller sub-regions can be created
        if dh_half >= self.min_region_size and dw_half >= self.min_region_size:
            # print(as_distinct, "is true", h0, w0, dh, dw, "size match")
            # upper-left
            self._split_n_merge(h0=h0, w0=w0, dh=dh_half, dw=dw_half)
            # upper-right
            self._split_n_merge(h0=h0, w0=w0 + dw_half, dh=dh_half, dw=dw_half)
            # lower-left
            self._split_n_merge(h0=h0 + dh_half, w0=w0, dh=dh_half, dw=dw_half)
            # lower-right
            self._split_n_merge(h0=h0 + dh_half, w0=w0 + dw_half, dh=dh_half, dw=dw_half)

    def split_n_merge(self) -> np.ndarray:
        self._split_n_merge(h0=0, w0=0, dh=self.height, dw=self.width)
        self._res_done = True
        return self.res_mask

    def illustrate(self):
        if self._res_done is False:
            res_mask = self.split_n_merge()
        else:
            res_mask = self.res_mask

        # init pyplot
        fig, _ax = plt.subplots(1, 2, figsize=(10, 5))
        ax = _ax.flatten()
        # show original img
        ax_ori = ax[0]
        ax_ori.imshow(self.img, cmap="gray")
        ax_ori.set_xticks([]), ax_ori.set_yticks([])
        ax_ori.set_xlabel("Original")
        # show result mask
        ax_res = ax[1]
        ax_res.imshow(res_mask, cmap="gray")
        ax_res.set_xticks([]), ax_res.set_yticks([])
        ax_res.set_xlabel("Result Mask")

        fig.suptitle(r"$P(\cdot)=\mathcal{1}_{\sigma\leq%d}$; Cell$\geq%d\times%d$"
                     % (self.thresh, self.min_region_size, self.min_region_size))
        plt.tight_layout()
        if SAVE is False:
            plt.show()
        else:
            res_fn = "res/4-split_n_merge-0.png"
            plt.savefig(res_fn, dpi=200)
            print("Result Saved: %s" % res_fn)
        plt.clf()


if "__main__" == __name__:
    test_img = load_images.load_img(fn="images/1_gray-2.bmp")
    # test_img = load_images.load_img(fn="images/8_gray.bmp")
    obj = SplitNMerge(img=test_img)
    # obj.split_n_merge()
    obj.illustrate()
