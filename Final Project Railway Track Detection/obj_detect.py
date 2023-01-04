import os
import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt


class ObjectDetect:
    def __init__(self,
                 model_path_yolo: str = "yolov5-master/res_models/yolov5s.pt",
                 model_path_fine_tuned: str = "yolov5-master/res_models/text_removed__epoch=50.pt"):
        assert os.path.exists(model_path_yolo)
        assert os.path.exists(model_path_fine_tuned)

        self.model_yolo = torch.hub.load("ultralytics/yolov5", "custom", path=model_path_yolo)
        self.model_fine_tuned = torch.hub.load("ultralytics/yolov5", "custom", path=model_path_fine_tuned)

    def infer_rgb_img(self, img_rgb: np.ndarray,
                      conf_thresh: float = 0.3,
                      _vis_only_path: str = "res/yolo_res.png") -> np.ndarray:
        do_visualize = 0 < len(_vis_only_path)
        if do_visualize:
            assert os.path.exists(os.path.split(_vis_only_path)[0])

        res_yolo = self.model_yolo(img_rgb)
        res_fine_tuned = self.model_fine_tuned(img_rgb)

        res_yolo_arr = res_yolo.xyxy[0].numpy()
        res_fine_tuned_arr = res_fine_tuned.xyxy[0].numpy()

        # filter res
        #   yolo res:
        #       (1) by confidence: objs with confidence >= `conf_thresh` are kept
        #       (2) by class labels: COCO-cls 0-7=person/bicycle/car/motorcycle/airplane/bus/train/truck are kept
        res_yolo_arr_filtered = res_yolo_arr[np.where((conf_thresh <= res_yolo_arr[:, 4]) & (7 >= res_yolo_arr[:, 5]))]
        res_fine_tuned_arr_filtered = res_fine_tuned_arr[np.where(conf_thresh <= res_fine_tuned_arr[:, 4])]

        if do_visualize:
            _img_rgb_show = img_rgb.copy()
            for _x_min, _y_min, _x_max_, _y_max, _, _ in res_yolo_arr_filtered:
                cv2.rectangle(_img_rgb_show,
                              color=(0, 255, 0),  # green in RGB
                              pt1=(_x_min, _y_min), pt2=(_x_max_, _y_max), thickness=1)
            for _x_min, _y_min, _x_max_, _y_max, _, _ in res_fine_tuned_arr_filtered:
                cv2.rectangle(_img_rgb_show,
                              color=(0, 0, 255),  # blue in RGB
                              pt1=(_x_min, _y_min), pt2=(_x_max_, _y_max), thickness=1)
            _img_show = cv2.cvtColor(_img_rgb_show, cv2.COLOR_RGBA2BGR)
            cv2.imwrite(_vis_only_path, _img_show)

        res = np.concatenate([res_yolo_arr_filtered, res_fine_tuned_arr_filtered])  # (obj_cnt, 6)
        res = np.delete(res, 4, axis=1)  # remove confidence column. (obj_cnt, 5): x_min/y_min/x_max/y_max/class
        res = np.round(res).astype(int)  # cast to int

        return res


if "__main__" == __name__:
    obj = ObjectDetect()
    im = cv2.imread("frames/VID-1-opt/VID-1-opt-0_frame_0.png")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    _ = obj.infer_rgb_img(img_rgb=im, _vis_only_path="res/yolo_res__VID-1-opt-0_frame_0__thresh=0.3.png")
