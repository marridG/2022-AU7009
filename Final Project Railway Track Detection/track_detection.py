import os
from typing import List
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
        self.recent_fitted = [np.array([False])]
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0, 0], dtype='float')
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
    def __init__(self, video_path: str, img_roi: np.ndarray, res_dir: str = "res",
                 _debug: bool = False, _use_cache: bool = True):
        self._DEBUG = _debug
        self._USE_CACHE = (_debug is False) and _use_cache

        assert os.path.exists(video_path)
        self.video_path = video_path
        self.video_alias = "VID-" + os.path.splitext(os.path.split(video_path)[-1])[0]  # "data/1.mp4" -> "VID-1"
        self.video_handler = video_handler.VideoHandler(video_path=self.video_path, _use_cache=self._USE_CACHE)

        assert os.path.exists(res_dir)
        self.res_dir = res_dir

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
            "res": os.path.join(self.res_dir, self.video_alias + "-3_track.mp4"),
            "res_frame": os.path.join(self.res_dir, self.video_alias
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
        fig, _ax = plt.subplots(2, 3, figsize=(15, 6))
        ax = _ax.flatten()
        for _ax in ax:  # remove x/y ticks & tick labels
            _ax.set_xticks([]), _ax.set_yticks([])

        img_thresh = self._apply_threshold(img=img)
        img_thresh_roi = self._apply_roi_mask(
            img=img_thresh,
            up_left=self.img_roi_up_left, up_right=self.img_roi_up_right,
            bot_left=self.img_roi_bot_left, bot_right=self.img_roi_bot_right)
        img_thresh_roi_trans = cv2.warpPerspective(src=img_thresh_roi, M=self.trans_src_2_dst,
                                                   dsize=img.shape[1::-1], flags=cv2.INTER_LINEAR)
        # perform detection
        if self.left_line.detected and self.right_line.detected:
            left_fit, right_fit, left_lane_inds, right_lane_inds = td_utils.find_line_by_previous(
                img_thresh_roi_trans,
                self.left_line.current_fit,
                self.right_line.current_fit)
        else:
            left_fit, right_fit, left_lane_inds, right_lane_inds = td_utils.find_line(img_thresh_roi_trans)
        self.left_line.update(left_fit)
        self.right_line.update(right_fit)

        img_area, gre1, mask_track, mask_expand = td_utils.draw_area(
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
                res_fn = self._MID_RES_FN_TEMPLATE["res_frame"] % _vis_title_only_frame_idx
                plt.savefig(res_fn, dpi=200)

    def process_n_visualize_frame(self, frame_idx: int):
        frame = self.video_handler.get_frame_by_idx(frame_idx=frame_idx)
        self._process_frame(img=frame, _vis_title_only_frame_idx=frame_idx)


if "__main__" == __name__:
    im_roi = np.load("temp/VID-4-2_roi_res_arr_scaled__ver=2__len=30__ratio=0.600.npy")
    obj = TrackDetection(video_path="data/4.mp4", img_roi=im_roi)
    obj.process_n_visualize_frame(frame_idx=0)
