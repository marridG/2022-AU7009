import os
import numpy as np
import cv2


class VideoHandler:
    def __init__(self, video_path: str, frames_dir: str = "frames", _use_cache: bool = True):
        assert os.path.exists(video_path)
        self.video_path = video_path
        self.video_alias = "VID-" + os.path.splitext(os.path.split(video_path)[-1])[0]  # "data/1.mp4" -> "VID-1"
        self.video_frame_cnt = -1

        assert os.path.exists(frames_dir)
        self.frames_dir = os.path.join(frames_dir, self.video_alias)
        if os.path.exists(self.frames_dir) is False:
            os.mkdir(self.frames_dir)

        self._USE_CACHE = _use_cache

        self._MID_RES_FN_TEMPLATE = {
            "frames": os.path.join(self.frames_dir, self.video_alias + "-0_frame_%d.png"),
        }

    def _save_frames(self):
        print("Start Saving Frames of \"%s\" ..." % self.video_path)
        cap = cv2.VideoCapture(self.video_path)
        ret, _frame = cap.read()
        _frame_idx = 0
        while True:
            res_fn = self._MID_RES_FN_TEMPLATE["frames"] % _frame_idx
            cv2.imwrite(filename=res_fn, img=_frame)
            ret, _frame = cap.read()
            if not ret:
                print("=== All %d Frames are Saved to: \"%s\"" % (_frame_idx + 1, self.frames_dir))
                break
            _frame_idx += 1

        self.video_frame_cnt = _frame_idx + 1

    def get_frame_by_idx(self, frame_idx: int) -> np.ndarray:
        _frames_fn_cnt = len(os.listdir(self.frames_dir))
        res_fn = self._MID_RES_FN_TEMPLATE["frames"] % frame_idx
        _err_msg = "Frame Index Out-of-Range. Attempt to get #%d out of %d Frames"

        if (self._USE_CACHE is True) and (1 < _frames_fn_cnt):
            self.video_frame_cnt = _frames_fn_cnt
        else:
            self._save_frames()

        if frame_idx >= self.video_frame_cnt - 1:
            raise IndexError(_err_msg % (self.video_frame_cnt, frame_idx))
        if os.path.exists(res_fn) is False:
            print("Frames Files Missing: \"%s\". Re-Saving Frames ..." % res_fn)
        return cv2.imread(res_fn)
