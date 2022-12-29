import cv2
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


# class ThresholdSelector:
#     def __init__(self, img: str or np.ndarray):
#         # load/store gray scale img
#         if isinstance(img, str):
#             self.img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
#         else:
#             assert 2 == len(img.shape), \
#                 "Input Image is Expected to be of Gray Scale having 2 Dimensions, Got %d Dimensions" % len(img.shape)
#             self.img = img
#
#         # params
#         self.params_canny_thresh_1 = 100
#         self.params_canny_thresh_2 = 200
#
#         # history
#         self.img_edge_detection = None
#         self.val_boundary_pts = []


def cal_thresh(img: np.ndarray, mask: np.ndarray = None, num_clusters: int = 0,
               debug=False, **kwargs) -> int or np.ndarray(dtype=int):
    height, width = img.shape  # (H, W)

    _img_copy = img.copy()
    if mask is None:
        mask = np.full_like(img, True)

    # ===1=== edge detection
    # canny edge detection: get gray scale edges (of shape (H,W); 0 for non-edge points, !0=255 for on-edge points)
    edges = cv2.Canny(
        image=_img_copy,
        threshold1=kwargs.get("canny_thresh_1", 100),
        threshold2=kwargs.get("canny_thresh_2", 200)
    )
    if debug is True:
        plt.imshow(edges, cmap="gray"), plt.show()

    # ===2=== discrete boundary sampling
    val_boundary_pts = []
    _adj_8_deltas = [
        (-1, 0), (1, 0), (0, -1), (0, 1),  # 4-adjacency
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]
    for _height in range(height):
        for _width in range(width):
            # skip if the pixel is masked off
            if 0 == mask[_height, _width]:  # note: True/False will be cast to 1/0
                continue
            # skip if the pixel is NOT on the edges
            if 0 == edges[_height, _width]:
                continue

            # for on-edge points: add values of its 8-adj points if existent
            for __adj_8_delta in _adj_8_deltas:
                __adj_pt_h = _height + __adj_8_delta[0]
                __adj_pt_w = _width + __adj_8_delta[1]
                # check for existence: whether is out-of-range
                if __adj_pt_h < 0 or __adj_pt_h >= height or __adj_pt_w < 0 or __adj_pt_w >= width:
                    continue

                # check for existence: whether is masked off
                if 0 == mask[__adj_pt_h, __adj_pt_w]:  # note: True/False will be cast to 1/0
                    continue
                val_boundary_pts.append(img[__adj_pt_h, __adj_pt_w])

    val_boundary_pts = np.array(val_boundary_pts)
    if debug is True:
        res_histogram = np.zeros(shape=(256,))
        for _val in val_boundary_pts:
            res_histogram[_val] += 1
        plt.bar(range(256), res_histogram, color="black"), plt.show()

    # ===3=== clustering (if necessary)
    # when NO clustering is required
    if num_clusters <= 1:
        res_thresh = np.mean(val_boundary_pts)
        res_thresh = int(res_thresh)
    # when clustering is required
    else:
        kmeans_operator = KMeans(n_clusters=num_clusters, random_state=kwargs.get("kmeans_random_state", 0))
        kmeans_cluster_labels = kmeans_operator.fit_predict(val_boundary_pts.reshape(-1, 1))  # vals: (N,) -> (N,1)
        # find cluster centroids
        kmeans_cluster_centroids = []
        for _cluster_idx in range(num_clusters):  # (N,)
            _cluster_vals = val_boundary_pts[np.where(_cluster_idx == kmeans_cluster_labels)]
            _cluster_vals_mean = np.mean(_cluster_vals).astype(int)
            kmeans_cluster_centroids.append(_cluster_vals_mean)
        kmeans_cluster_centroids.sort()
        print("Cluster Centroids:", kmeans_cluster_centroids)
        res_thresh = np.array(kmeans_cluster_centroids, dtype=int)

    if debug is True:
        print("Threshold:", res_thresh)
    return res_thresh


def gen_gray_scale_histogram(img: np.ndarray, debug=False, **kwargs) -> np.ndarray:
    height, width = img.shape  # (H, W)

    res_histogram = np.zeros(shape=(256,))
    for _height in range(height):
        for _width in range(width):
            _gray_val = img[_height, _width]
            res_histogram[_gray_val] += 1

    if debug is True:
        plt.bar(range(256), res_histogram, color="black"), plt.show()
    if kwargs.get("do_norm") is True:
        res_histogram /= img.size
    return res_histogram


if "__main__" == __name__:
    # fn = "images/1_gray.bmp"
    # im = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    # thresh = cal_thresh(img=im)
    # _, im_thresh = cv2.threshold(im, thresh=thresh, maxval=255, type=cv2.THRESH_BINARY_INV)
    # plt.imshow(im_thresh, cmap="binary"), plt.show()

    fn = "images/23.bmp"
    im = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    gen_gray_scale_histogram(im)
