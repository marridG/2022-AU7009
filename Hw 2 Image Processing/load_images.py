import os
from typing import List
import cv2
import numpy as np

EASY_IMG_FNS = ["images/简单图像3.jpg", "images/basket.jpg", ]
COMPLEX_IMG_FNS = ["images/1_gray-2.bmp", "images/8_gray.bmp", "images/gray高斯多尺度平滑.jpg", ]
NOISE_IMG_FNS = ["images/gray椒盐噪声1-1.jpg", "images/gray高斯噪声3-1.jpg", ]


def load_img(fn: str) -> np.ndarray:
    assert os.path.exists(fn)
    # res = cv2.imread(_fn, cv2.IMREAD_GRAYSCALE)
    # handles CHN filename problem: https://blog.csdn.net/LJ1120142576/article/details/127676051
    res = cv2.imdecode(np.fromfile(fn, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    return res


def load_img_set(easy_idx: int = 0, complex_idx: int = 0, noise_idx: int = 1) -> List[np.ndarray]:
    assert 0 <= easy_idx <= len(EASY_IMG_FNS)
    assert 0 <= complex_idx <= len(COMPLEX_IMG_FNS)
    assert 0 <= noise_idx <= len(NOISE_IMG_FNS)

    res_fn = [EASY_IMG_FNS[easy_idx], COMPLEX_IMG_FNS[complex_idx], NOISE_IMG_FNS[noise_idx]]
    res_img = [
        load_img(fn=_fn)
        for _fn in res_fn
    ]

    _res_str = "Image Loaded: "
    for _fn, _img in zip(res_fn, res_img):
        _res_str += "\"%s\"=(%d, %d); " % (os.path.split(_fn)[-1], _img.shape[0], _img.shape[1])
    _res_str = _res_str[:-2]
    print(_res_str)

    return res_img


if "__main__" == __name__:
    _ = load_img_set()
