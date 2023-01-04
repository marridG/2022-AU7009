import os
from typing import List
from tqdm import tqdm
import numpy as np
import cv2
from matplotlib import pyplot as plt

import video_handler
import track_detection_utils as td_utils


# Define a class to receive the characteristics of each line detection
class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_fitted = []  # [np.array([False])]
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = []  # [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0, 0], dtype="float")
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def check_detected(self):
        if (self.diffs[0] < 0.01 and self.diffs[1] < 10.0 and self.diffs[2] < 1000.) and len(self.recent_fitted) > 0:
            return True
        else:
            return False

    def update(self, fit):
        if fit is not None:
            if self.best_fit is not None:
                self.diffs = abs(fit - self.best_fit)
                if self.check_detected():
                    self.detected = True
                    if len(self.recent_fitted) > 10:
                        self.recent_fitted = self.recent_fitted[1:]
                        self.recent_fitted.append(fit)
                    else:
                        self.recent_fitted.append(fit)
                    self.best_fit = np.average(self.recent_fitted, axis=0)
                    self.current_fit = fit
                else:
                    self.detected = False
            else:
                self.best_fit = fit
                self.current_fit = fit
                self.detected = True
                self.recent_fitted.append(fit)


class TrackDetection:
    def __init__(self, video_path: str, img_roi: np.ndarray, res_dir: str = "res", temp_dir: str = "temp",
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

        # ROI: (4,2), up-left / up-right / bot-left / bot-right
        self.img_roi_up_left = (int(img_roi[0, 0]), int(img_roi[0, 1]))
        self.img_roi_up_right = (int(img_roi[1, 0]), int(img_roi[1, 1]))
        self.img_roi_bot_left = (int(img_roi[2, 0]), int(img_roi[2, 1]))
        self.img_roi_bot_right = (int(img_roi[3, 0]), int(img_roi[3, 1]))

        self.left_line = Line()
        self.right_line = Line()

        # img_roi: (4,2), up - left / up - right / bot - left / bot - right
        self.trans_src_2_dst, self.trans_dst_2_src = self.get_transform(
            up_left_src=img_roi[0], up_right_src=img_roi[1], bot_left_src=img_roi[2], bot_right_src=img_roi[3],
            # up_left_dst=(0, 0), up_right_dst=(639, 0), bot_left_dst=(0, 367), bot_right_dst=(639, 367)
            up_left_dst=(150, 0), up_right_dst=(500, 0), bot_left_dst=(0, 367), bot_right_dst=(639, 367)
        )

        self._MID_RES_FN_TEMPLATE = {
            "res_vid": os.path.join(self.res_dir, self.video_alias + "-3_track.mp4"),
            "res_vid_test": os.path.join(self.res_dir, self.video_alias + "-3_track__range=%d-%d-%d.mp4"),
            "res_arr_all": os.path.join(self.temp_dir, self.video_alias + "-3_track_arr_all.npy"),
            "res_arr_trans": os.path.join(self.temp_dir, self.video_alias + "-3_track_arr_trans.npy"),
            "res_arr_mask_track": os.path.join(self.temp_dir, self.video_alias + "-3_track_arr_mask_track.npy"),
            "res_arr_mask_expand": os.path.join(self.temp_dir, self.video_alias + "-3_track_arr_mask_expand.npy"),
            "res_pic_frame": os.path.join(self.res_dir, self.video_alias
                                          + "-3_track"
                                          + "__frame=%d"
                                          + ".png"),
        }

    @staticmethod
    def get_transform(
            up_left_src: (int, int), up_right_src: (int, int), bot_left_src: (int, int), bot_right_src: (int, int),
            up_left_dst: (int, int), up_right_dst: (int, int), bot_left_dst: (int, int), bot_right_dst: (int, int)
    ):
        src = np.float32([[up_left_src, up_right_src, bot_left_src, bot_right_src]])
        dst = np.float32([[up_left_dst, up_right_dst, bot_left_dst, bot_right_dst]])
        # src = np.float32([[(600, 1080), (850, 300), (1600, 1080), (1000, 300)]])
        # dst = np.float32([[(500, 1080), (0, 0), (1500, 1080), (1300, 0)]])
        src_2_dst = cv2.getPerspectiveTransform(src, dst)
        dst_2_src = cv2.getPerspectiveTransform(dst, src)
        return src_2_dst, dst_2_src

    @staticmethod
    def _apply_threshold(img):
        _img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        _, _, _img_hsv_v = cv2.split(_img_hsv)
        _img_hsv_v = _img_hsv_v.ravel()[np.flatnonzero(_img_hsv_v)]
        _avg_v = sum(_img_hsv_v) / len(_img_hsv_v)
        # print(_avg_v)
        # exit(0)
        if _avg_v <= 100:  # [daytime] 2=136; [night] 5=73, 7=59
            edged = cv2.Canny(img, 30, 90)
        else:
            edged = cv2.Canny(img, 50, 150)
        res = np.zeros_like(edged)  # (H, W)
        res[edged == 255] = 1

        return res

    @staticmethod
    def _apply_roi_mask(img, up_left: (int, int), up_right: (int, int), bot_left: (int, int), bot_right: (int, int)):
        roi_vertices = np.array([bot_left, up_left, up_right, bot_right])

        mask = np.zeros_like(img)
        # mask = cv2.polylines(mask, [roi_vertices], True, (255, 255, 255), 2)
        mask = cv2.fillPoly(mask, [roi_vertices], (255, 255, 255))  # mark ROI as white
        res = cv2.bitwise_and(mask, img)
        return res

    def _process_frame(self, img: np.ndarray, _vis_title_only_frame_idx: int = -1):
        do_visualize = (0 <= _vis_title_only_frame_idx)
        if do_visualize:
            fig, _ax = plt.subplots(2, 3, figsize=(15, 6))
            ax = _ax.flatten()
            for _ax in ax:  # remove x/y ticks & tick labels
                _ax.set_xticks([]), _ax.set_yticks([])

        img_thresh = self._apply_threshold(img=img)
        img_thresh_roi = self._apply_roi_mask(
            img=img_thresh,
            up_left=self.img_roi_up_left, up_right=self.img_roi_up_right,
            bot_left=self.img_roi_bot_left, bot_right=self.img_roi_bot_right
        )
        img_thresh_roi_trans = cv2.warpPerspective(src=img_thresh_roi, M=self.trans_src_2_dst,
                                                   dsize=img.shape[1::-1], flags=cv2.INTER_LINEAR)
        # mask_track = mask_expand = img_area = np.zeros_like(img)
        # perform detection
        if self.left_line.detected and self.right_line.detected:  # when is NOT 1-ST frame
            _left_fit, _right_fit, _, _ = td_utils.find_line_by_previous(
                img_thresh_roi_trans,
                self.left_line.current_fit,
                self.right_line.current_fit
            )
            _, _, mask_track, _ = td_utils.draw_area(
                img=img, img_binary_trans=img_thresh_roi_trans,
                trans_dst_2_src=self.trans_dst_2_src,
                left_fit=_left_fit, right_fit=_right_fit)
            # print(mask_track.nonzero()[0].size)
            if 5000 < mask_track.nonzero()[0].size:
                left_fit = _left_fit
                right_fit = _right_fit
            else:
                left_fit = self.left_line.current_fit
                right_fit = self.right_line.current_fit
        else:  # when is 1-ST frame
            left_fit, right_fit, _, _ = td_utils.find_line(img_thresh_roi_trans)
        self.left_line.update(left_fit)
        self.right_line.update(right_fit)

        img_area, _, mask_track, mask_expand = td_utils.draw_area(
            img=img, img_binary_trans=img_thresh_roi_trans,
            trans_dst_2_src=self.trans_dst_2_src,
            left_fit=left_fit, right_fit=right_fit)

        if do_visualize:
            ax[0].imshow(img_thresh, cmap="gray"), ax[0].set_xlabel("(a) Edge Detection")
            ax[1].imshow(img_thresh_roi, cmap="gray"), ax[1].set_xlabel("(b) ROI-Only Masked")
            ax[2].imshow(img_thresh_roi_trans, cmap="gray"), ax[2].set_xlabel("(c) Perspective Transformed")
            ax[3].imshow(mask_track, cmap="gray"), ax[3].set_xlabel("(d) Result - Track Region")
            ax[4].imshow(mask_expand, cmap="gray"), ax[4].set_xlabel("(e) Result - Track Expanded Region")
            ax[5].imshow(img_area), ax[5].set_xlabel("(f) Result - Overall Illustration")
            fig.suptitle("Track Detection for \"%s\" (Frame #%d)"
                         % (os.path.split(self.video_path)[-1], _vis_title_only_frame_idx))
            plt.tight_layout()
            if self._DEBUG is True:
                plt.show()
            else:
                res_fn = self._MID_RES_FN_TEMPLATE["res_pic_frame"] % _vis_title_only_frame_idx
                plt.savefig(res_fn, dpi=200)
                print("Frame-Related Illustration Saved: \"%s\"" % res_fn)

        return img_area, img_thresh_roi_trans, mask_track, mask_expand

    def process_n_visualize_frame(self, frame_idx: int):
        frame = self.video_handler.get_frame_by_idx(frame_idx=frame_idx)
        self._process_frame(img=frame, _vis_title_only_frame_idx=frame_idx)

    def process_img(self, img: np.ndarray):
        return self._process_frame(img=img)

    def process_video(self, frame_idx_start: int = 0, frame_idx_end: int = None) \
            -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        res = []  # frame_cnt * (height, width, 3)
        res_trans = []  # frame_cnt * (height, width)
        res_mask_track = []  # frame_cnt * (height, width)
        res_mask_expand = []  # frame_cnt * (height, width)

        if (0 != frame_idx_start) or (frame_idx_end is not None):
            assert 0 <= frame_idx_start <= self.video_handler.video_frame_cnt - 1
            if frame_idx_end is not None:
                assert frame_idx_end <= self.video_handler.video_frame_cnt
            else:
                frame_idx_end = self.video_handler.video_frame_cnt
            assert frame_idx_start <= frame_idx_end - 1
            video_as_test = True
        else:
            frame_idx_end = self.video_handler.video_frame_cnt
            video_as_test = False

        _frame_idx_iterator = tqdm(range(frame_idx_start, frame_idx_end))
        for _frame_idx in _frame_idx_iterator:
            _frame = self.video_handler.get_frame_by_idx(frame_idx=_frame_idx)
            _res, _res_trans, _res_mask_track, _res_mask_expand = self._process_frame(img=_frame)
            res.append(_res)
            res_trans.append(_res_trans)
            res_mask_track.append(_res_mask_track)
            res_mask_expand.append(_res_mask_expand)
            # if 0 == _frame_idx % 20:
            #     print("Processed Frame #%d" % _frame_idx)

        res = np.array(res)  # (frame_cnt, height, width, 3)
        res_trans = np.array(res_trans)  # (frame_cnt, height, width)
        res_mask_track = np.array(res_mask_track)  # (frame_cnt, height, width)
        res_mask_expand = np.array(res_mask_expand)  # (frame_cnt, height, width)

        if video_as_test is True:
            # save result (all, but as TEST) as a video
            res_fn = self._MID_RES_FN_TEMPLATE["res_vid_test"] \
                     % (frame_idx_start, frame_idx_end, self.video_handler.video_frame_cnt)
            self.video_handler.frames_to_video(frames=res, res_path=res_fn)
            print("Results (1 TEST Video) Saved")
        else:
            # save result (all) as a video
            self.video_handler.frames_to_video(frames=res, res_path=self._MID_RES_FN_TEMPLATE["res_vid"])
            # save result as arrays
            np.save(file=self._MID_RES_FN_TEMPLATE["res_arr_all"], arr=res)
            np.save(file=self._MID_RES_FN_TEMPLATE["res_arr_trans"], arr=res_trans)
            np.save(file=self._MID_RES_FN_TEMPLATE["res_arr_mask_track"], arr=res_mask_track)
            np.save(file=self._MID_RES_FN_TEMPLATE["res_arr_mask_expand"], arr=res_mask_expand)
            print("Results (1 Video, 4 Arrays) Saved")

        return res, res_trans, res_mask_track, res_mask_expand


if "__main__" == __name__:
    roi = np.load("temp/VID-7-2_roi_res_arr_scaled__ver=2__len=30__ratio=0.600.npy")
    obj = TrackDetection(video_path="data/7.mp4", img_roi=roi, _debug=True)
    # obj.process_n_visualize_frame(frame_idx=300)
    # obj.process_video(frame_idx_start=300, frame_idx_end=600)
    obj.process_video()

    # for file, roi_file, (f_simple, f_hard), video_flag in zip(
    #         ["data/%d.mp4" % idx for idx in range(1, 8)],
    #         ["temp/VID-%d-2_roi_res_arr_scaled__ver=2__len=30__ratio=0.600.npy" % idx for idx in range(1, 8)],
    #         [(0, 2900), (0, 3700), (0, 1000), (900, 0), (1500, 100), (600, 2100), (0, 2200)],
    #         # [False, True, False, True, False, False, True]
    # ):
    #     roi = np.load(roi_file)
    #     obj = TrackDetection(video_path=file, img_roi=roi)
    #     obj.process_n_visualize_frame(frame_idx=f_simple)
    #     obj.process_n_visualize_frame(frame_idx=f_hard)
    #     if video_flag is True:
    #         obj.process_video()
    #     print("========================================")
