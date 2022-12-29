from typing import List, Union, Tuple
from collections import deque
import cv2
import numpy as np
from matplotlib import pyplot as plt

import load_images

DEBUG = False
SAVE = (DEBUG is False) and True

SOBEL_KERNEL_HORI = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
SOBEL_KERNEL_VERT = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
LAPLACIAN_KERNEL_BASE = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

PLT_FONT_SIZE_FIGURE_TITLE = 30
PLT_FONT_SIZE_SUBPLOT_TITLE = 20
PLT_FONT_SIZE_SUBPLOT_XLABEL = 20
plt.rcParams.update({"figure.titlesize": PLT_FONT_SIZE_FIGURE_TITLE})
plt.rcParams.update({"axes.titlesize": PLT_FONT_SIZE_SUBPLOT_TITLE})
plt.rcParams.update({"axes.labelsize": PLT_FONT_SIZE_SUBPLOT_XLABEL})


def similarity(val_1: int, val_2: int) -> float:
    # L1-norm of pixels' gray scale values
    res = abs(int(val_1) - int(val_2))
    return res


def get_1st_min_loc(img: np.ndarray) -> (int, int):
    idx_c_order = np.argmin(img)
    idx_x = idx_c_order // img.shape[1]
    idx_y = idx_c_order % img.shape[1]
    return idx_x, idx_y


def get_1st_max_loc(img: np.ndarray) -> (int, int):
    idx_c_order = np.argmax(img)
    idx_x = idx_c_order // img.shape[1]
    idx_y = idx_c_order % img.shape[1]
    return idx_x, idx_y


def region_growing_step_by_step(img: np.ndarray, thresh: float, start_loc: Union[str, Tuple[int, int]] = "max") \
        -> (List[str], List[np.ndarray], Tuple[int, int], List[float]):
    if isinstance(start_loc, str):
        start_loc = start_loc.lower()
        assert start_loc in ["max", "min"]
        if "max" == start_loc:
            start_loc = get_1st_max_loc(img=img)
        else:
            start_loc = get_1st_min_loc(img=img)
    else:
        start_loc = (start_loc[0] % img.shape[0], start_loc[1] % img.shape[1])

    res_titles = [
        r"Mask ($\Delta<%d$)" % thresh,
        "Result (Start=%d,%d)" % (start_loc[0], start_loc[1])
    ]
    # res_imgs = []

    # height, width = img.shape
    d_height = [0, 0, -1, -1, -1, 1, 1, 1]
    d_width = [1, -1, -1, 0, 1, -1, 0, 1]
    res_mask = np.zeros_like(img)  # 0=init/background, 1=foreground
    sim_history = []
    q = deque()  # element = (height_loc, width_loc)
    q.append(start_loc)

    while q:
        seed_x, seed_y = q.popleft()
        res_mask[seed_x, seed_y] = 1
        for _d_height, _d_width in zip(d_height, d_width):
            _x = seed_x + _d_height
            _y = seed_y + _d_width
            try:
                _status = res_mask[_x, _y]
            except IndexError:
                continue
            # ignore if processed
            if 0 != _status:
                continue
            else:
                _sim = similarity(img[seed_x, seed_y], img[_x, _y])
                sim_history.append(_sim)
                # grow iff. thresh > sim
                if thresh > _sim:
                    q.append((_x, _y))
                    res_mask[_x, _y] = 1

    res = img.copy()
    res[np.where(1 == res_mask)] = 0

    # print(start_loc)
    # print(sim_history)
    res_imgs = [res_mask, res]
    return res_titles, res_imgs, start_loc, sim_history


def region_growing_illustration(img: np.ndarray, thresh: float, start_loc: Union[str, Tuple[int, int]] = "max"):
    # init pyplot
    fig, _ax = plt.subplots(1, 4, figsize=(20, 6))
    ax = _ax.flatten()
    # show original img
    ax[0].imshow(img, cmap="gray")
    ax[0].set_xticks([]), ax[0].set_yticks([])
    ax[0].set_xlabel("Original")
    # step-by-step
    res_steps_titles, res_steps_imgs, res_start_loc, res_sim_history = region_growing_step_by_step(
        img=img, thresh=thresh, start_loc=start_loc)
    _, _, _ = ax[1].hist(res_sim_history, 20, density=True, alpha=0.75)
    ax[1].set_xlabel("Similarity Histogram")
    for _step_idx in range(len(res_steps_imgs)):
        _title, _img = res_steps_titles[_step_idx], res_steps_imgs[_step_idx]
        _ax_idx = 2 + _step_idx
        ax[_ax_idx].imshow(_img, cmap="gray")
        ax[_ax_idx].set_xticks([]), ax[_ax_idx].set_yticks([])
        ax[_ax_idx].set_xlabel(_title, color="black")

    fig.suptitle(r"Region Growing (Start=%s; iff. Sim=$\Delta < %d$)"
                 % (start_loc.upper() if isinstance(start_loc, str) else "Selected", thresh))
    plt.tight_layout()
    if SAVE is False:
        plt.show()
    else:
        res_fn = "res/3-growing-0.png"
        plt.savefig(res_fn, dpi=200)
        print("Result Saved: %s" % res_fn)
    plt.clf()


if "__main__" == __name__:
    # test_img = load_images.load_img(fn="images/1_gray-2.bmp")
    # region_growing_illustration(test_img, thresh=10, start_loc="min")

    test_img = load_images.load_img(fn="images/简单图像3.jpg")
    region_growing_illustration(test_img, thresh=15, start_loc="max")
