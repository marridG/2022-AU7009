import os
from tqdm import tqdm
import numpy as np
import cv2
from matplotlib import pyplot as plt

import video_handler


class TextRemove:
    def __init__(self, res_dir: str = "res", temp_dir: str = "temp", _debug: bool = False, ):
        self._DEBUG = _debug

        assert os.path.exists(res_dir)
        self.res_dir = res_dir
        assert os.path.exists(temp_dir)
        self.temp_dir = temp_dir

        # ROI definition of big-illustration/bold-text/regular-text-1/regular-text-2,
        #   given as range of indices of <np.ndarray>
        # [e.g.] ax0=(100,200), ax1=(300,400)
        #   for <np.ndarray>: img[100:200+1, 300:400+1]
        #   for image: height=100~200, width=300~400
        #   for OpenCV axis: up_left=[300,100], bot_right=[400,200]
        self.roi_big_idx_range_ax0 = (245, 345)  # big-illustration ROI: axis-0
        self.roi_big_idx_range_ax1 = (400, 600)  # big-illustration ROI: axis-1
        self.roi_bold_idx_range_ax0 = (245, 288)  # bold-text ROI: axis-0
        self.roi_bold_idx_range_ax1 = (502, 563)  # bold-text ROI: axis-1
        self.roi_reg_1_idx_range_ax0 = (288, 315)  # regular-text-1 ROI: axis-0
        self.roi_reg_1_idx_range_ax1 = (453, 600)  # regular-text-1 ROI: axis-1
        self.roi_reg_2_idx_range_ax0 = (315, 345)  # regular-text-2 ROI: axis-0
        self.roi_reg_2_idx_range_ax1 = (400, 600)  # regular-text-2 ROI: axis-1

        # HLS's L channel sensitivity: regard pixels as those of white texts iff (255-sens <= L <= 255)
        self.lightness_sensitivity_bold = 30
        self.lightness_sensitivity_regular = 60

        self._MID_RES_FN_TEMPLATE = {
            "res_vid": os.path.join(self.temp_dir, "%s-4_text_removal.mp4"),
            "res_frame": os.path.join(self.res_dir, "%s-4_text_removal__frame=%d.png"),
        }

    def _pointer_slice_img_2_big(self, img_to_edit: np.ndarray) -> np.ndarray:
        """slice the image to big-illustration ROI. if call `res[::]=X`, `img_to_edit`'s elements will be updated."""
        res = img_to_edit[self.roi_big_idx_range_ax0[0]: self.roi_big_idx_range_ax0[1] + 1,
              self.roi_big_idx_range_ax1[0]:self.roi_big_idx_range_ax1[1] + 1]
        return res

    def _pointer_slice_img_2_bold(self, img_to_edit: np.ndarray) -> np.ndarray:
        """slice the image to bold-text ROI. if call `res[::]=X`, `img_to_edit`'s elements will be updated."""
        res = img_to_edit[self.roi_bold_idx_range_ax0[0]: self.roi_bold_idx_range_ax0[1] + 1,
              self.roi_bold_idx_range_ax1[0]:self.roi_bold_idx_range_ax1[1] + 1]
        return res

    def _pointer_slice_img_2_reg_1(self, img_to_edit: np.ndarray) -> np.ndarray:
        """slice the image to regular-text-1 ROI. if call `res[::]=X`, `img_to_edit`'s elements will be updated."""
        res = img_to_edit[self.roi_reg_1_idx_range_ax0[0]: self.roi_reg_1_idx_range_ax0[1] + 1,
              self.roi_reg_1_idx_range_ax1[0]:self.roi_reg_1_idx_range_ax1[1] + 1]
        return res

    def _pointer_slice_img_2_reg_2(self, img_to_edit: np.ndarray) -> np.ndarray:
        """slice the image to regular-text-2 ROI. if call `res[::]=X`, `img_to_edit`'s elements will be updated."""
        res = img_to_edit[self.roi_reg_2_idx_range_ax0[0]: self.roi_reg_2_idx_range_ax0[1] + 1,
              self.roi_reg_2_idx_range_ax1[0]:self.roi_reg_2_idx_range_ax1[1] + 1]
        return res

    @staticmethod
    def _lightness_thresh(roi: np.ndarray, sensitivity: int) -> np.ndarray:
        assert 0 < sensitivity < 255
        # apply HLS (L-channel) thresholding, to select text. res=ori-text
        roi_hls = cv2.cvtColor(src=roi, code=cv2.COLOR_BGR2HLS)
        roi_hls_l_channel = roi_hls[:, :, 1]
        roi_mask = cv2.inRange(roi_hls_l_channel, 255 - sensitivity, 255)
        return roi_mask

    def _process_frame(self, img: np.ndarray,
                       _vis_title_only_vid_fullname: str = "",
                       _vis_title_only_vid_alias: str = "",
                       _vis_title_only_frame_idx: int = -1) -> np.ndarray:
        do_visualize = (0 <= _vis_title_only_frame_idx) \
                       and 0 < len(_vis_title_only_vid_fullname) \
                       and 0 < len(_vis_title_only_vid_fullname)
        if do_visualize:
            fig, _ax = plt.subplots(2, 3, figsize=(15, 6))
            ax = _ax.flatten()
            for _ax in ax:  # remove x/y ticks & tick labels
                _ax.set_xticks([]), _ax.set_yticks([])

        # extract the target ROI
        _img = img.copy()
        img_roi = self._pointer_slice_img_2_big(img_to_edit=_img)
        _img_roi_bold = self._pointer_slice_img_2_bold(img_to_edit=_img)
        _img_roi_reg_1 = self._pointer_slice_img_2_reg_1(img_to_edit=_img)
        _img_roi_reg_2 = self._pointer_slice_img_2_reg_2(img_to_edit=_img)
        # illustrate: draw rectangles on ORI image & then slice the big ROI
        if do_visualize:
            img_roi_show = img.copy()
            cv2.rectangle(img=img_roi_show,
                          pt1=(self.roi_bold_idx_range_ax1[0], self.roi_bold_idx_range_ax0[0]),
                          pt2=(self.roi_bold_idx_range_ax1[1], self.roi_bold_idx_range_ax0[1]),
                          color=(255, 0, 0),  # RGB
                          thickness=2)  # bold text: red
            cv2.rectangle(img=img_roi_show,
                          pt1=(self.roi_reg_1_idx_range_ax1[0], self.roi_reg_1_idx_range_ax0[0]),
                          pt2=(self.roi_reg_1_idx_range_ax1[1], self.roi_reg_1_idx_range_ax0[1]),
                          color=(0, 255, 0),  # RGB
                          thickness=2)  # regular text-1: green
            cv2.rectangle(img=img_roi_show,
                          pt1=(self.roi_reg_2_idx_range_ax1[0], self.roi_reg_2_idx_range_ax0[0]),
                          pt2=(self.roi_reg_2_idx_range_ax1[1], self.roi_reg_2_idx_range_ax0[1]),
                          color=(0, 0, 255),  # RGB
                          thickness=2)  # regular text-1: blue
            img_roi_show = self._pointer_slice_img_2_big(img_to_edit=img_roi_show)

        # apply HLS (L-channel) thresholding, to select text. res=ori-text
        img_roi_mask = np.zeros((img.shape[0], img.shape[1]),
                                dtype=np.uint8)  # replace sub-masks onto ORI-img-shaped mask
        _mask_bold = self._lightness_thresh(roi=_img_roi_bold, sensitivity=self.lightness_sensitivity_bold)
        _mask_reg_1 = self._lightness_thresh(roi=_img_roi_reg_1, sensitivity=self.lightness_sensitivity_regular)
        _mask_reg_2 = self._lightness_thresh(roi=_img_roi_reg_2, sensitivity=self.lightness_sensitivity_regular)
        self._pointer_slice_img_2_bold(img_to_edit=img_roi_mask)[::] = _mask_bold
        self._pointer_slice_img_2_reg_1(img_to_edit=img_roi_mask)[::] = _mask_reg_1
        self._pointer_slice_img_2_reg_2(img_to_edit=img_roi_mask)[::] = _mask_reg_2
        img_roi_mask = self._pointer_slice_img_2_big(img_to_edit=img_roi_mask)
        img_roi_res = img_roi - cv2.bitwise_and(img_roi, img_roi, mask=img_roi_mask)

        # dilate text mask as the `inpaint` mask
        _KERNEL_SIZE = 5
        kernel = np.ones((_KERNEL_SIZE, _KERNEL_SIZE), np.uint8)
        # roi_white_res_gray = cv2.cvtColor(roi_white_res, cv2.COLOR_BGR2GRAY)
        img_roi_mask_edit = cv2.dilate(img_roi_mask, kernel, iterations=1)
        # roi_white_mask_edit = cv2.erode(roi_white_mask_edit, kernel, iterations=1)
        img_roi_res_fixed = cv2.inpaint(
            src=img_roi_res, inpaintMask=img_roi_mask_edit,
            inpaintRadius=1, flags=cv2.INPAINT_TELEA
            # inpaintRadius=1, flags=cv2.INPAINT_NS
        )

        # cast back to the ORI-image-shaped
        img_res = img.copy()
        self._pointer_slice_img_2_big(img_to_edit=img_res)[::] = img_roi_res_fixed

        if do_visualize:
            ax[0].imshow(img_roi_show), ax[0].set_xlabel("(a) Text ROI")
            ax[1].imshow(img_roi_mask, cmap="gray"), ax[1].set_xlabel("(b) Text Mask")
            ax[2].imshow(img_roi_res), ax[2].set_xlabel("(c) Text ROI with Text Masked-off")
            ax[3].imshow(img_roi_mask_edit, cmap="gray"), ax[3].set_xlabel("(d) Text Mask Dilated")
            ax[4].imshow(img_roi_res_fixed), ax[4].set_xlabel("(d) Text ROI Fixed after Text Removal")
            ax[5].imshow(img_res), ax[5].set_xlabel("(e) Result")
            # ax[1].remove(), ax[2].remove(), ax[3].remove(), ax[4].remove(), ax[5].remove()

            fig.suptitle("Text Removal for \"%s\" (Frame #%d)"
                         % (_vis_title_only_vid_fullname, _vis_title_only_frame_idx))
            plt.tight_layout()
            if self._DEBUG is True:
                plt.show()
            else:
                res_fn = self._MID_RES_FN_TEMPLATE["res_frame"] \
                         % (_vis_title_only_vid_alias, _vis_title_only_frame_idx)
                plt.savefig(res_fn, dpi=200)
                print("Frame-Related Illustration Saved: \"%s\"" % res_fn)

        return img_res

    def process_frame(self, img: np.ndarray, vid_fullname: str, vid_alias: str, frame_idx: int) -> np.ndarray:
        res = self._process_frame(
            img=img,
            _vis_title_only_vid_fullname=vid_fullname,
            _vis_title_only_vid_alias=vid_alias,
            _vis_title_only_frame_idx=frame_idx
        )
        return res

    def process_video(self, handler: video_handler.VideoHandler):
        res = []
        _frame_idx_iterator = tqdm(range(0, handler.video_frame_cnt))
        for _frame_idx in _frame_idx_iterator:
            _frame = handler.get_frame_by_idx(frame_idx=_frame_idx)
            _res = self._process_frame(img=_frame)
            res.append(_res)
            # if 0 == _frame_idx % 20:
            #     print("Processed Frame #%d" % _frame_idx)
        res = np.array(res)  # (frame_cnt, height, width, 3)

        res_path = self._MID_RES_FN_TEMPLATE["res_vid"] % handler.video_alias
        handler.frames_to_video(frames=res, res_path=res_path)
        print("Text-Removal Results Saved: \"%s\"" % res_path)


if "__main__" == __name__:
    # obj = TextRemove()
    # h = video_handler.VideoHandler(video_path="data/1.mp4")
    # obj.process_frame(img=h.get_frame_by_idx(frame_idx=0),
    #                   vid_fullname=h.video_fullname, vid_alias=h.video_alias, frame_idx=0)
    # obj.process_video(handler=h)

    for file, f, flag in zip(
            ["data/%d.mp4" % idx for idx in range(1, 8)],
            [0, 0, 0, 0, 0, 0, 0],
            [False, False, False, False, True, True, True]):
        obj = TextRemove()
        h = video_handler.VideoHandler(video_path=file)
        obj.process_frame(img=h.get_frame_by_idx(frame_idx=f),
                          vid_fullname=h.video_fullname, vid_alias=h.video_alias, frame_idx=f)
        if flag is True:
            obj.process_video(handler=h)
        print("========================================")
