import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

import video_handler


class ROIExtract:
    def __init__(self,
                 video_path: str,
                 res_dir: str = "res", temp_dir: str = "temp",
                 roi_len_top: int = 30,
                 roi_ratio_bottom: float = 0.6,
                 _debug: bool = False,
                 _use_cache: bool = True,
                 _roi_version: int = 2):
        assert os.path.exists(video_path)
        self.video_path = video_path
        self.video_alias = "VID-" + os.path.splitext(os.path.split(video_path)[-1])[0]  # "data/1.mp4" -> "VID-1"
        self.video_handler = video_handler.VideoHandler(video_path=self.video_path)

        assert os.path.exists(res_dir)
        self.res_dir = res_dir
        assert os.path.exists(temp_dir)
        self.temp_dir = temp_dir

        assert roi_len_top > 1  # width in pixel of the top edge of the trapezoid-shaped ROI
        self.roi_len_top = roi_len_top
        assert 0 < roi_ratio_bottom < 1  # shrinkage ratio of the bottom edge of the trapezoid-shaped ROI
        self.roi_ratio_bottom = roi_ratio_bottom

        self._DEBUG = _debug
        self._USE_CACHE = (_debug is False) and _use_cache
        self._ROI_VERSION = _roi_version

        self._MID_RES_FN_TEMPLATE = {
            "optical_flow": os.path.join(self.temp_dir, self.video_alias + "-1_optical_flow.png"),
            "roi_details": os.path.join(self.temp_dir, self.video_alias
                                        + "-2_roi_details"
                                        + "__ver=" + ("%d" % self._ROI_VERSION)
                                        + "__len=" + ("%d" % self.roi_len_top)
                                        + "__ratio=" + ("%.3f" % self.roi_ratio_bottom)
                                        + ".png"),
            "roi_res": os.path.join(self.res_dir, self.video_alias
                                    + "-2_roi_res"
                                    + "__ver=" + ("%d" % self._ROI_VERSION)
                                    + "__len=" + ("%d" % self.roi_len_top)
                                    + "__ratio=" + ("%.3f" % self.roi_ratio_bottom)
                                    + "__frame=%d_%d"
                                    + ".png"),
            "roi_res_arr_ori": os.path.join(self.temp_dir, self.video_alias
                                            + "-2_roi_res_arr_ori"
                                            + "__ver=" + ("%d" % self._ROI_VERSION)
                                            + "__len=" + ("%d" % self.roi_len_top)
                                            + "__ratio=" + ("%.3f" % self.roi_ratio_bottom)
                                            + ".npy"),
            "roi_res_arr_scaled": os.path.join(self.temp_dir, self.video_alias
                                               + "-2_roi_res_arr_scaled"
                                               + "__ver=" + ("%d" % self._ROI_VERSION)
                                               + "__len=" + ("%d" % self.roi_len_top)
                                               + "__ratio=" + ("%.3f" % self.roi_ratio_bottom)
                                               + ".npy"),
        }

    def _optical_flow(self) -> np.ndarray:
        # reference: https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html

        cap = cv2.VideoCapture(self.video_path)
        # params for ShiTomasi corner detection
        feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Create some random colors
        color = np.random.randint(0, 255, (100, 3))
        # Take first frame and find corners in it
        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Optical Flow Finished on All Frames")
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # calculate optical flow: returns nextPts, status (1 iff. nextPts match prevPts), error
            p1, st, err = cv2.calcOpticalFlowPyrLK(prevImg=old_gray, nextImg=frame_gray, prevPts=p0, nextPts=None,
                                                   **lk_params)
            # Select good points
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(img=mask, pt1=(int(a), int(b)), pt2=(int(c), int(d)), color=color[i].tolist(),
                                thickness=2)
                # frame = cv2.circle(img=frame, center=(int(a), int(b)), radius=5, color=color[i].tolist(), thickness=-1)
            if self._DEBUG is True:
                img = cv2.add(frame, mask)
                cv2.imshow("Frames with Accumulated Optical Flow", img)
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

        if self._DEBUG is True:
            cv2.destroyAllWindows()
        res_fn = self._MID_RES_FN_TEMPLATE["optical_flow"]
        cv2.imwrite(filename=res_fn, img=mask)
        return mask

    @staticmethod
    def _get_intersect(a1, a2, b1, b2):
        """
        Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
        reference: https://stackoverflow.com/a/42727584
        a1: [x, y] a point on the first line
        a2: [x, y] another point on the first line
        b1: [x, y] a point on the second line
        b2: [x, y] another point on the second line
        """
        s = np.vstack([a1, a2, b1, b2])  # s for stacked
        h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
        l1 = np.cross(h[0], h[1])  # get first line
        l2 = np.cross(h[2], h[3])  # get second line
        x, y, z = np.cross(l1, l2)  # point of intersection
        if z == 0:  # lines are parallel
            return float('inf'), float('inf')
        return x / z, y / z

    def _get_roi(self) -> (np.ndarray, np.ndarray):
        """
        :return: two <np.ndarray> of shape (4, 2), non-scaled & scaled group of ROI vertices.
                    axis-1: up-left/up-right/bot-left/bot-right vertices;
                    axis-2: x/y of a vertex
        """
        print("Start Extracting ROI (Video \"%s\") ..." % self.video_path)
        if os.path.exists(self._MID_RES_FN_TEMPLATE["optical_flow"]) is False:
            img = self._optical_flow()
        else:
            img = cv2.imread(self._MID_RES_FN_TEMPLATE["optical_flow"])

        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        img_gray_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
        img_gray_blur_edges = cv2.Canny(img_gray_blur, 100, 200)

        lines = cv2.HoughLinesP(image=img_gray_blur_edges, rho=1, theta=np.pi / 180, threshold=50,
                                minLineLength=img.shape[0] // 3, maxLineGap=15)
        lines_filtered = []  # [ ( start-x, start-y, end-x, end-y), ... ], all <int>
        lines_filtered_slope = []  # [ slope_val, ... ], all <float>
        if lines is not None:
            for _line_idx in range(len(lines)):
                _start_x, _start_y, _end_x, _end_y = lines[_line_idx][0]
                # avoid divide-by-zero error
                if 1e-3 >= abs(_start_x - _end_x):
                    continue
                # calculate slope
                _slope_k = (_start_y - _end_y) * 1. / (_start_x - _end_x)

                # filter out nearly-vertical lines
                if 1e2 < abs(_slope_k):
                    continue
                # filter out nearly-horizontal lines
                if 1e-2 >= abs(_slope_k):
                    continue

                lines_filtered.append((int(_start_x), int(_start_y), int(_end_x), int(_end_y)))
                lines_filtered_slope.append(float(_slope_k))

        # find the largest & smallest slope
        lines_filtered_slope_argmax = np.argmax(lines_filtered_slope)
        lines_filtered_slope_argmin = np.argmin(lines_filtered_slope)

        img_lines = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)  # (B, G, R)
        # draw all the lines
        for _line_idx in range(len(lines_filtered)):
            _start_x, _start_y, _end_x, _end_y = lines_filtered[_line_idx]

            if lines_filtered_slope_argmax == _line_idx:
                # _line_color = (0, 0, 255)  # (B, G, R) = red
                _line_color = (255, 0, 0)  # (B, G, R) = blue
                _line_width = 3
            elif lines_filtered_slope_argmin == _line_idx:
                _line_color = (255, 0, 0)  # (B, G, R) = blue
                _line_width = 3
            else:
                _line_color = (0, 255, 0)  # (B, G, R) = green
                _line_width = 1
            cv2.line(img_lines, (_start_x, _start_y), (_end_x, _end_y), _line_color, _line_width, cv2.LINE_AA)
        # calculate vanishing point, as the intersection of the two lines with the largest & smallest slopes
        lines_filtered_max_slope = lines_filtered[lines_filtered_slope_argmax]
        lines_filtered_min_slope = lines_filtered[lines_filtered_slope_argmin]
        pt_vanish_x, pt_vanish_y = self._get_intersect(
            (lines_filtered_max_slope[0], lines_filtered_max_slope[1]),
            (lines_filtered_max_slope[2], lines_filtered_max_slope[3]),
            (lines_filtered_min_slope[0], lines_filtered_min_slope[1]),
            (lines_filtered_min_slope[2], lines_filtered_min_slope[3]),
        )
        pt_vanish_x, pt_vanish_y = int(pt_vanish_x), int(pt_vanish_y)
        # draw vanishing point
        cv2.circle(img_lines, center=(pt_vanish_x, pt_vanish_y), radius=5, color=(0, 0, 255), thickness=-1)  # red
        # draw line: vanishing point <--> argmax/argmin slope line
        cv2.line(img_lines, (pt_vanish_x, pt_vanish_y), (lines_filtered_max_slope[0], lines_filtered_max_slope[1]),
                 color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)  # blue
        cv2.line(img_lines, (pt_vanish_x, pt_vanish_y), (lines_filtered_min_slope[0], lines_filtered_min_slope[1]),
                 color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)  # blue

        # determine ROI (bottom): find intersection of image-bottom-line-segment with largest- & smallest-sloped lines
        # left-most: using min slope (origin is at img's top-left)
        _cand_leftmost_x, _cand_leftmost_y = self._get_intersect(
            (lines_filtered_min_slope[0], lines_filtered_min_slope[1]),
            (lines_filtered_min_slope[2], lines_filtered_min_slope[3]),
            (0, img.shape[0] - 1), (img.shape[1] - 1, img.shape[0] - 1),  # bottom-right# bottom-right, bottom-right
        )
        if 0 <= _cand_leftmost_x < img.shape[1] // 2:
            leftmost_x = int(_cand_leftmost_x)
            cv2.circle(img_lines, center=(leftmost_x, img.shape[0] - 1), radius=2, color=(255, 0, 0),
                       thickness=-1)  # blue
            cv2.line(img_lines, (leftmost_x, img.shape[0] - 1),
                     (lines_filtered_min_slope[0], lines_filtered_min_slope[1]),
                     color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)  # blue
        else:
            leftmost_x = 0
        # right-most: using max slope (origin is at img's top-left)
        _cand_rightmost_x, _cand_rightmost_y = self._get_intersect(
            (lines_filtered_max_slope[0], lines_filtered_max_slope[1]),
            (lines_filtered_max_slope[2], lines_filtered_max_slope[3]),
            (0, img.shape[0] - 1), (img.shape[1] - 1, img.shape[0] - 1),  # bottom-right# bottom-right, bottom-right
        )
        if img.shape[1] // 2 < _cand_rightmost_x <= img.shape[1] - 1:
            rightmost_x = int(_cand_rightmost_x)
            cv2.circle(img_lines, center=(rightmost_x, img.shape[0] - 1), radius=5, color=(255, 0, 0),
                       thickness=-1)  # blue
            cv2.line(img_lines, (rightmost_x, img.shape[0] - 1),
                     (lines_filtered_max_slope[0], lines_filtered_max_slope[1]),
                     color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)  # blue
        else:
            rightmost_x = img.shape[1] - 1

        # calculate ROI: bottom loc + top len: triangle -> trapezoid
        roi_pt_bottom_left = (leftmost_x, img.shape[0] - 1)
        roi_pt_bottom_right = (rightmost_x, img.shape[0] - 1)
        roi_pt_bottom_left_scaled = (
            leftmost_x + int(self.roi_ratio_bottom * (pt_vanish_x - leftmost_x)),
            img.shape[0] - 1
        )
        roi_pt_bottom_right_scaled = (
            rightmost_x + int(self.roi_ratio_bottom * (pt_vanish_x - rightmost_x)),
            img.shape[0] - 1
        )
        if 1 == self._ROI_VERSION:
            roi_pt_upper_left = (pt_vanish_x - self.roi_len_top // 2, pt_vanish_y)
            roi_pt_upper_right = (pt_vanish_x + self.roi_len_top // 2, pt_vanish_y)
            roi_pt_upper_left_scaled = (pt_vanish_x - self.roi_len_top // 2, pt_vanish_y)
            roi_pt_upper_right_scaled = (pt_vanish_x + self.roi_len_top // 2, pt_vanish_y)
        elif 2 == self._ROI_VERSION:
            _cal_delta_y = lambda delta_x, full_x, full_y: int(delta_x * full_y * 1. / full_x)
            roi_delta_y = _cal_delta_y(
                delta_x=self.roi_len_top,
                full_x=roi_pt_bottom_right[0] - roi_pt_bottom_left[0],
                full_y=img.shape[0] - 1 - pt_vanish_y)
            roi_delta_y_scaled = _cal_delta_y(
                delta_x=self.roi_len_top,
                full_x=roi_pt_bottom_right_scaled[0] - roi_pt_bottom_left_scaled[0],
                full_y=img.shape[0] - 1 - pt_vanish_y)
            roi_pt_upper_left = (pt_vanish_x - self.roi_len_top // 2, pt_vanish_y + roi_delta_y)
            roi_pt_upper_right = (pt_vanish_x + self.roi_len_top // 2, pt_vanish_y + roi_delta_y)
            roi_pt_upper_left_scaled = (pt_vanish_x - self.roi_len_top // 2, pt_vanish_y + roi_delta_y_scaled)
            roi_pt_upper_right_scaled = (pt_vanish_x + self.roi_len_top // 2, pt_vanish_y + roi_delta_y_scaled)
        else:
            raise NotImplementedError("Unknown ROI_VERSION: \"%d\". Supported: 1 or 2" % self._ROI_VERSION)

        # draw ROI - ori
        cv2.line(img_lines,
                 (roi_pt_bottom_left[0], roi_pt_bottom_left[1]),
                 (roi_pt_upper_left[0], roi_pt_upper_left[1]),
                 color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)  # red
        cv2.line(img_lines,
                 (roi_pt_upper_left[0], roi_pt_upper_left[1]),
                 (roi_pt_upper_right[0], roi_pt_upper_right[1]),
                 color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)  # red
        cv2.line(img_lines,
                 (roi_pt_upper_right[0], roi_pt_upper_right[1]),
                 (roi_pt_bottom_right[0], roi_pt_bottom_right[1]),
                 color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)  # red
        # draw ROI - scaled
        cv2.line(img_lines,
                 (roi_pt_bottom_left_scaled[0], roi_pt_bottom_left_scaled[1]),
                 (roi_pt_upper_left_scaled[0], roi_pt_upper_left_scaled[1]),
                 color=(0, 0, 255), thickness=3, lineType=cv2.LINE_AA)  # red
        cv2.line(img_lines,
                 (roi_pt_upper_left_scaled[0], roi_pt_upper_left_scaled[1]),
                 (roi_pt_upper_right_scaled[0], roi_pt_upper_right_scaled[1]),
                 color=(0, 0, 255), thickness=3, lineType=cv2.LINE_AA)  # red
        cv2.line(img_lines,
                 (roi_pt_upper_right_scaled[0], roi_pt_upper_right_scaled[1]),
                 (roi_pt_bottom_right_scaled[0], roi_pt_bottom_right_scaled[1]),
                 color=(0, 0, 255), thickness=3, lineType=cv2.LINE_AA)  # red

        res_img_fn = self._MID_RES_FN_TEMPLATE["roi_details"]
        cv2.imwrite(filename=res_img_fn, img=img_lines)
        print("ROI Extraction Mid-Info Illustration Saved: \"%s\"" % res_img_fn)

        res_arr_ori = np.array([
            roi_pt_upper_left, roi_pt_upper_right, roi_pt_bottom_left, roi_pt_bottom_right
        ], dtype=int)
        res_arr_scaled = np.array([
            roi_pt_upper_left_scaled, roi_pt_upper_right_scaled, roi_pt_bottom_left_scaled, roi_pt_bottom_right_scaled
        ], dtype=int)
        res_arr_fn_ori = self._MID_RES_FN_TEMPLATE["roi_res_arr_ori"]
        res_arr_fn_scaled = self._MID_RES_FN_TEMPLATE["roi_res_arr_scaled"]
        np.save(file=res_arr_fn_ori, arr=res_arr_ori)
        np.save(file=res_arr_fn_scaled, arr=res_arr_scaled)
        print("ROI Vertices Saved: ORI=\"%s\", SCALED=\"%s\"" % (res_arr_fn_ori, res_arr_fn_scaled))

        return res_arr_ori, res_arr_scaled

    def get_roi(self) -> (np.ndarray, np.ndarray):
        _ori_fn = self._MID_RES_FN_TEMPLATE["roi_res_arr_ori"]
        _scaled_fn = self._MID_RES_FN_TEMPLATE["roi_res_arr_scaled"]
        if (self._USE_CACHE is True) and (os.path.exists(_ori_fn) is True) and (os.path.exists(_scaled_fn) is True):
            ori = np.load(_ori_fn)
            scaled = np.load(_scaled_fn)
        else:
            ori, scaled = self._get_roi()

        print("===== ROI Got (Video \"%s\")" % self.video_path)
        return ori, scaled

    def get_roi_n_visualize(self, frame_idx_simple: int, frame_idx_hard: int) -> (np.ndarray, np.ndarray):
        ori, scaled = self.get_roi()  # (4,2), (4,2)

        # init pyplot
        fig, _ax = plt.subplots(1, 4, figsize=(20, 3.5))
        for ax in _ax:  # remove the underlying axes
            ax.set_xticks([]), ax.set_yticks([])
        ax = _ax.flatten()

        ax[0].imshow(mpimg.imread(self._MID_RES_FN_TEMPLATE["optical_flow"]))
        ax[0].set_xlabel("(a) All-Frame Optical Flow")

        ax[1].imshow(mpimg.imread(self._MID_RES_FN_TEMPLATE["roi_details"]))
        ax[1].set_xlabel("(b) ROI Details")

        for _ax_idx, _frame_idx, _title in zip(
                [2, 3], [frame_idx_simple, frame_idx_hard], ["(c) ROI Example", "(d) ROI Example"]
        ):
            img = self.video_handler.get_frame_by_idx(frame_idx=_frame_idx)
            for __from_idx, __end_idx in [(2, 0), (0, 1), (1, 3)]:  # up-left/up-right/bot-left/bot-right vertices;
                cv2.line(img,
                         (ori[__from_idx, 0], ori[__from_idx, 1]),
                         (ori[__end_idx, 0], ori[__end_idx, 1]),
                         # color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)  # blue
                         color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)  # red
                cv2.line(img,
                         (scaled[__from_idx, 0], scaled[__from_idx, 1]),
                         (scaled[__end_idx, 0], scaled[__end_idx, 1]),
                         color=(255, 0, 0), thickness=3, lineType=cv2.LINE_AA)  # red

            ax[_ax_idx].imshow(img)
            ax[_ax_idx].set_xlabel(_title + " (Frame #%d)" % _frame_idx)

        fig.suptitle("ROI Extraction for \"%s\"" % os.path.split(self.video_path)[-1])
        plt.tight_layout()
        # plt.show()
        res_fn = self._MID_RES_FN_TEMPLATE["roi_res"] % (frame_idx_simple, frame_idx_hard)
        plt.savefig(res_fn, dpi=200)
        print("Frame-Related Illustration Saved: \"%s\"" % res_fn)

        return ori, scaled


if "__main__" == __name__:
    # obj = ROIExtract(video_path="data/1.mp4")
    # # obj.get_roi()
    # obj.get_roi_n_visualize(frame_idx_simple=0, frame_idx_hard=2900)
    for file, (f_simple, f_hard) in zip(
            ["data/%d.mp4" % idx for idx in range(1, 8)],
            [(0, 2900), (0, 3700), (0, 1000), (900, 0), (1500, 100), (600, 2100), (0, 2200)]):
        obj = ROIExtract(video_path=file)
        # obj.get_roi()
        obj.get_roi_n_visualize(frame_idx_simple=f_simple, frame_idx_hard=f_hard)
