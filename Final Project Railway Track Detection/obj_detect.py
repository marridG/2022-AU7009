import os
from typing import Tuple
import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt


class NMS:
    # [reference] https://zhuanlan.zhihu.com/p/547804478
    def __init__(self):
        pass

    @staticmethod
    def _box_intersection(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) \
            -> Tuple[int, int, int, int]:
        x11, y11, x12, y12 = b1
        x21, y21, x22, y22 = b2

        xl = max(x11, x21)
        xr = min(x12, x22)
        yt = max(y11, y21)
        yb = min(y12, y22)
        return xl, yt, xr, yb

    @staticmethod
    def _area(box: Tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = box
        width = max(x2 - x1, 0)
        height = max(y2 - y1, 0)
        return width * height

    def _iou(self, b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
        intersection = self._box_intersection(b1, b2)
        inter_area = self._area(intersection)
        union_area = self._area(b1) + self._area(b2) - inter_area
        return inter_area / union_area

    def nms(self, predicts: np.ndarray, score_thresh: float = 0.6, iou_thresh: float = 0.3):
        """
        Non-Maximum Suppression
        :param predicts:            <np.ndarray> shaped (n, 5). The second dimension includes:
                                        1 probability and 4 numbers x_min, y_min, x_max, y_max denoting a bounding box
        :param score_thresh:        b-boxes with probability lower than `score_thresh` will be discarded
        :param iou_thresh:          threshold determining whether two boxes are "overlapped"
        :return:                    filtered predictions (n, 5), and the indices of remaining boxes list[n]
        """
        n_remainder = len(predicts)
        vis = [False] * n_remainder

        # Filter predicts with low probability
        for i, predict in enumerate(predicts):
            if predict[0] < score_thresh:
                vis[i] = True
                n_remainder -= 1

        # NMS
        output_predicts = []
        output_indices = []
        while n_remainder > 0:
            max_pro = -1
            max_index = 0
            # Find argmax
            for i, p in enumerate(predicts):
                if not vis[i]:
                    if max_pro < p[0]:
                        max_index = i
                        max_pro = p[0]

            # Append output
            max_p = predicts[max_index]
            output_predicts.append(max_p)
            output_indices.append(max_index)

            # Suppress
            for i, p in enumerate(predicts):
                if not vis[i] and i != max_index:
                    if self._iou(p[1:5], max_p[1:5]) > iou_thresh:
                        vis[i] = True
                        n_remainder -= 1
            vis[max_index] = True
            n_remainder -= 1

        return output_predicts, output_indices


class ObjectDetect:
    def __init__(self,
                 model_path_yolo: str = "yolov5-master/res_models/yolov5s.pt",
                 model_path_fine_tuned: str = "yolov5-master/res_models/text_removed__epoch=50.pt",
                 _debug: bool = False):
        self._DEBUG = _debug

        assert os.path.exists(model_path_yolo)
        assert os.path.exists(model_path_fine_tuned)

        self.model_yolo = torch.hub.load("ultralytics/yolov5", "custom", path=model_path_yolo)
        self.model_fine_tuned = torch.hub.load("ultralytics/yolov5", "custom", path=model_path_fine_tuned)

        self.nms_handler = NMS()

    def infer_rgb_img(self, img_rgb: np.ndarray,
                      conf_thresh: float = 0.3, iou_thresh: float = 0.3,
                      _vis_only_title_subfix: str = "Result",
                      _vis_only_path: str = "res/yolo_res.png") -> np.ndarray:
        do_visualize = (0 < len(_vis_only_title_subfix)) and (0 < len(_vis_only_path))
        if do_visualize:
            assert os.path.exists(os.path.split(_vis_only_path)[0])
            fig, _ax = plt.subplots(1, 2, figsize=(12, 4.5))
            ax = _ax.flatten()
            for _ax in ax:  # remove x/y ticks & tick labels
                _ax.set_xticks([]), _ax.set_yticks([])

        res_yolo = self.model_yolo(img_rgb)
        res_fine_tuned = self.model_fine_tuned(img_rgb)

        res_yolo_arr = res_yolo.xyxy[0].numpy()
        res_fine_tuned_arr = res_fine_tuned.xyxy[0].numpy()

        # filter res
        #   yolo res:
        #       (1) by confidence: objs with confidence >= `conf_thresh` are kept
        #       (2) by class labels: COCO-cls 0-7=person/bicycle/car/motorcycle/airplane/bus/train/truck are kept
        #   yolo (fined-tuned) res:
        #       (1) by confidence: objs with confidence >= `conf_thresh` are kept
        res_yolo_arr_filtered = res_yolo_arr[np.where((conf_thresh <= res_yolo_arr[:, 4]) & (7 >= res_yolo_arr[:, 5]))]
        res_fine_tuned_arr_filtered = res_fine_tuned_arr[np.where(conf_thresh <= res_fine_tuned_arr[:, 4])]

        # res objs = {yolo-filtered-objs} \union {yolo-fine-tuned-filtered-objs}
        res_concat = np.concatenate([
            res_yolo_arr_filtered, res_fine_tuned_arr_filtered
        ])  # (obj_cnt, 6): x_min/y_min/x_max/y_max/confidence/class

        # apply NMS
        _res_concat_reorder = res_concat[:, [4, 0, 1, 2, 3]]  # (obj_cnt, 5): confidence/x_min/y_min/x_max/y_max
        _, _res_nms_idx = self.nms_handler.nms(predicts=_res_concat_reorder,
                                               score_thresh=conf_thresh, iou_thresh=iou_thresh)
        res_concat_nms = res_concat[_res_nms_idx, ::]  # (obj_cnt, 6): x_min/y_min/x_max/y_max/confidence/class

        res = res_concat_nms[:, 0:4]  # remove confidence & class column. (obj_cnt, 4): x_min/y_min/x_max/y_max
        res = np.round(res).astype(int)  # cast to int

        if do_visualize:
            _img_rgb_show = img_rgb.copy()
            for _x_min, _y_min, _x_max_, _y_max, _, _ in res_yolo_arr_filtered:
                cv2.rectangle(_img_rgb_show,
                              color=(0, 255, 0),  # BGR, green
                              pt1=(_x_min, _y_min), pt2=(_x_max_, _y_max), thickness=1)
            for _x_min, _y_min, _x_max_, _y_max, _, _ in res_fine_tuned_arr_filtered:
                cv2.rectangle(_img_rgb_show,
                              color=(255, 0, 0),  # BGR, blue
                              pt1=(_x_min, _y_min), pt2=(_x_max_, _y_max), thickness=1)
            _img_show = cv2.cvtColor(_img_rgb_show, cv2.COLOR_RGB2BGR)
            # ax[0].scatter([0], [1])
            ax[0].imshow(_img_show), ax[0].set_xlabel("(a) YOLOv5s (green) & YOLOv5s-Fine-Tuned (blue)")

            _img_rgb_show = img_rgb.copy()
            for _x_min, _y_min, _x_max_, _y_max in res:
                cv2.rectangle(_img_rgb_show,
                              color=(0, 0, 255),  # BGR, red
                              pt1=(_x_min, _y_min), pt2=(_x_max_, _y_max), thickness=1)
            _img_show = cv2.cvtColor(_img_rgb_show, cv2.COLOR_RGB2BGR)
            ax[1].imshow(_img_show), ax[1].set_xlabel(r"(b) NMS Result (conf$\geq$%.3f, iou$\geq$%.3f)"
                                                      % (conf_thresh, iou_thresh))

            fig.suptitle("Object Detection " + _vis_only_title_subfix.strip())
            plt.tight_layout()
            if self._DEBUG is True:
                plt.show()
            else:
                _res_fn_split = os.path.splitext(_vis_only_path)
                res_fn = _res_fn_split[0] + "__conf=%.3f__iou=%.3f" % (conf_thresh, iou_thresh) + _res_fn_split[1]
                plt.savefig(res_fn, dpi=200)
                print("Frame-Related Illustration Saved: \"%s\"" % res_fn)

        return res


if "__main__" == __name__:
    obj = ObjectDetect()
    im = cv2.imread("frames/VID-1-opt/VID-1-opt-0_frame_0.png")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    _ = obj.infer_rgb_img(
        img_rgb=im,
        _vis_only_title_subfix="for \"1.mp4\" (Frame #0)",
        _vis_only_path="res/yolo_res__VID-1-opt-0_frame_0.png"
    )
