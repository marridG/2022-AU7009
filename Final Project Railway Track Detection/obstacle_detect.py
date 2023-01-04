import os
from tqdm import tqdm
import numpy as np
import cv2
from matplotlib import pyplot as plt

import video_handler
import roi_extract
import track_detection
import text_remove
import obj_detect


class ObstacleDetect:
    def __init__(self, video_path: str,
                 res_dir: str = "res", temp_dir: str = "temp",
                 _debug: bool = False, _use_cache: bool = True):
        self._DEBUG = _debug
        self._USE_CACHE = _use_cache

        assert os.path.exists(video_path)
        self.video_path = video_path
        self.video_handler = video_handler.VideoHandler(video_path=self.video_path, _use_cache=self._USE_CACHE)
        self.video_alias = self.video_handler.video_alias

        assert os.path.exists(res_dir)
        self.res_dir = res_dir
        assert os.path.exists(temp_dir)
        self.temp_dir = temp_dir

        roi_extract_handler = roi_extract.ROIExtract(video_path=self.video_path, _debug=self._DEBUG,
                                                     _use_cache=self._USE_CACHE)
        _, roi = roi_extract_handler.get_roi()
        self.track_detection_handler = track_detection.TrackDetection(video_path=self.video_path, img_roi=roi,
                                                                      _debug=self._DEBUG, _use_cache=self._USE_CACHE)
        self.text_remove_handler = text_remove.TextRemove(_debug=self._DEBUG)
        self.obj_detect_handler = obj_detect.ObjectDetect(_debug=self._DEBUG)

        self._MID_RES_FN_TEMPLATE = {
            "res_vid": os.path.join(self.res_dir, self.video_alias + "-9.mp4"),
            "res_vid_test": os.path.join(self.res_dir, self.video_alias + "-9__range=%d-%d-%d.mp4"),
        }
        print("Obstacle Detection Handler Initialized")

    def _process_frame(self, img: np.ndarray, obstacle_intersection_pixel_cnt_thresh: int = 10,
                       _vis_only_path: str = None):
        do_visualize = _vis_only_path is not None
        if do_visualize is True:
            assert os.path.exists(os.path.split(_vis_only_path)[0])

        # apply track detection
        img_track_drawn = img.copy()
        img_track_drawn, _, mask_track, mask_expand = self.track_detection_handler.process_img(img_track_drawn)
        if do_visualize is True:
            print("!!!!! Track Detection Finished !!!!!")

        # apply text removal
        _img_text_removed = img.copy()
        _img_text_removed = self.text_remove_handler.process_img(img=_img_text_removed)

        # apply object detection
        _img_text_removed_rgb = _img_text_removed.copy()
        cv2.cvtColor(_img_text_removed, cv2.COLOR_BGR2RGB)  # OopenCV `cv2.imread()` BGR -> PIL `Image.open()` RGB
        obj_bbox = self.obj_detect_handler.infer_rgb_img(img_rgb=_img_text_removed_rgb)  # (obj_cnt, 4)
        # draw bbox
        img_all_obj = _img_text_removed.copy()
        for _x_min, _y_min, _x_max, _y_max in obj_bbox:
            cv2.rectangle(img_all_obj,
                          color=(0, 0, 255),  # BGR, red
                          pt1=(_x_min, _y_min), pt2=(_x_max, _y_max), thickness=2)
        if do_visualize is True:
            print("!!!!! Object Detection Finished !!!!!")

        # find obstacle objects
        _obj_on_track = np.zeros(obj_bbox.shape[0])
        _obj_on_expand = np.zeros(obj_bbox.shape[0])
        for _obj_idx, (_x_min, _y_min, _x_max, _y_max) in enumerate(obj_bbox):
            _mask_obj = np.zeros_like(mask_track)
            _mask_obj[_y_min:_y_max, _x_min:_x_max] = 255

            _mask_obj_on_track = cv2.bitwise_and(mask_track, _mask_obj)
            _mask_obj_on_expand = cv2.bitwise_and(mask_expand, _mask_obj)

            # # debug codes
            # fig, _ax = plt.subplots(2, 3, figsize=(15, 6))
            # ax = _ax.flatten()
            # for _ax in ax:  # remove x/y ticks & tick labels
            #     _ax.set_xticks([]), _ax.set_yticks([])
            # ax[0].imshow(mask_track), ax[0].set_xlabel("track mask")
            # ax[1].imshow(_mask_obj), ax[1].set_xlabel("obj mask")
            # ax[2].imshow(_mask_obj_on_track), ax[2].set_xlabel("obj on track")
            # ax[3].imshow(mask_expand), ax[3].set_xlabel("expand mask")
            # ax[4].imshow(_mask_obj), ax[4].set_xlabel("obj mask")
            # ax[5].imshow(_mask_obj_on_expand), ax[5].set_xlabel("obj on expand")
            # plt.tight_layout()
            # plt.savefig("temp/temp.png", dpi=200)

            _pixel_cnt_obj_n_track = len(_mask_obj_on_track.nonzero()[0])
            _pixel_cnt_obj_n_expand = len(_mask_obj_on_expand.nonzero()[0])
            if _pixel_cnt_obj_n_track >= obstacle_intersection_pixel_cnt_thresh:
                _obj_on_track[_obj_idx] = 1
            elif _pixel_cnt_obj_n_expand >= obstacle_intersection_pixel_cnt_thresh:
                _obj_on_expand[_obj_idx] = 1
        obj_bbox_obstacle_track = obj_bbox[_obj_on_track.nonzero()[0], ::]
        obj_bbox_obstacle_expand = obj_bbox[_obj_on_expand.nonzero()[0], ::]
        # obj_bbox_non_obstacle = obj_bbox[0 == cv2.bitwise_or(_obj_on_track, _obj_on_expand)]
        if do_visualize is True:
            print("!!!!! Obstacle Object Filtering Finished !!!!!")

        # draw filtered objects, as result
        img_result = img_track_drawn.copy()
        for _x_min, _y_min, _x_max, _y_max in obj_bbox_obstacle_track:  # obstacle objects of track
            cv2.rectangle(img_result,
                          color=(0, 255, 0),  # BGR, green
                          pt1=(_x_min, _y_min), pt2=(_x_max, _y_max), thickness=3)

        for _x_min, _y_min, _x_max, _y_max in obj_bbox_obstacle_expand:  # obstacle objects of expand
            cv2.rectangle(img_result,
                          color=(0, 255, 0),  # BGR, red
                          pt1=(_x_min, _y_min), pt2=(_x_max, _y_max), thickness=2)

        # construct full res
        _img_height, _img_width, _img_channel = img.shape
        _font_dict = {
            "fontFace": cv2.FONT_HERSHEY_DUPLEX,
            "fontScale": 1,
            "color": (255, 255, 255)  # BGR, white
        }
        _text_loc_bias_hori = -10
        img_full_result = np.zeros((_img_height * 2, _img_width * 2, _img_channel), dtype=np.uint8)
        #   up-left: original img
        img_full_result[0:_img_height, 0:_img_width, ::] = img[::]
        cv2.putText(img_full_result, text="Original",
                    org=(0, _img_height + _text_loc_bias_hori), **_font_dict)
        #   up-right: track & expand
        img_full_result[0:_img_height, _img_width:_img_width * 2, ::] = img_track_drawn[::]
        cv2.putText(img_full_result, text="Track + Expand",
                    org=(_img_width, _img_height + _text_loc_bias_hori), **_font_dict)
        #   bot-left: all detected objects
        img_full_result[_img_height:_img_height * 2, 0:_img_width, ::] = img_all_obj[::]
        cv2.putText(img_full_result, text="All Detected Objects",
                    org=(0, _img_height * 2 + _text_loc_bias_hori), **_font_dict)
        #   bot-right : result (track & obstacles)
        img_full_result[_img_height:_img_height * 2, _img_width:_img_width * 2, ::] = img_result[::]
        cv2.putText(img_full_result, text="Result: Track + Expand + Obstacles",
                    org=(_img_width, _img_height * 2 + _text_loc_bias_hori), **_font_dict)
        if do_visualize is True:
            cv2.imwrite(_vis_only_path, img_full_result)
            print("!!!!! Results Drawn & Saved (\"%s\") !!!!!" % _vis_only_path)

        return img_result, img_full_result

    def process_video(self):
        res = []
        _frame_idx_iterator = tqdm(range(0, self.video_handler.video_frame_cnt))
        for _frame_idx in _frame_idx_iterator:
            _frame = self.video_handler.get_frame_by_idx(frame_idx=_frame_idx)
            _, _res_frame = self._process_frame(img=_frame)
            res.append(_res_frame)

        res = np.array(res)  # (frame_cnt, height, width, 3)
        res_path = self._MID_RES_FN_TEMPLATE["res_vid"]
        self.video_handler.frames_to_video(frames=res, res_path=res_path)
        print("Results Saved: \"%s\"" % res_path)


if "__main__" == __name__:
    # obj = ObstacleDetect(video_path="data/4.mp4")
    # obj._process_frame(img=cv2.imread("frames/VID-4/VID-4-0_frame_0.png"), _vis_only_path="res/temp-4-0.png")
    #
    # obj = ObstacleDetect(video_path="data/7.mp4")
    # obj._process_frame(img=cv2.imread("frames/VID-7/VID-7-0_frame_0.png"), _vis_only_path="res/temp-7-0.png")

    obj = ObstacleDetect(video_path="data/4.mp4")
    obj.process_video()
    print("========================================")
    obj = ObstacleDetect(video_path="data/7.mp4")
    obj.process_video()
    print("========================================")
