import os
import numpy as np
import cv2


class VideoHandler:
    def __init__(self, video_path: str, frames_dir: str = "frames", _use_cache: bool = True):
        assert os.path.exists(video_path)
        self.video_path = video_path
        self.video_alias = "VID-" + os.path.splitext(os.path.split(video_path)[-1])[0]  # "data/1.mp4" -> "VID-1"
        self.video_fps = -1
        self.video_frame_shape_hw = (-1, -1)  # (height, width)
        self.video_frame_cnt = -1
        self._parse_info()

        assert os.path.exists(frames_dir)
        self.frames_dir = os.path.join(frames_dir, self.video_alias)
        if os.path.exists(self.frames_dir) is False:
            os.mkdir(self.frames_dir)

        self._USE_CACHE = _use_cache

        self._MID_RES_FN_TEMPLATE = {
            "frames": os.path.join(self.frames_dir, self.video_alias + "-0_frame_%d.png"),
        }
        print("Handler Init for Video \"%s\": ALIAS=\"%s\", FPS=%d, H/W=%d/%d, CNT=%d"
              % (self.video_path, self.video_alias, self.video_fps,
                 self.video_frame_shape_hw[0], self.video_frame_shape_hw[1], self.video_frame_cnt))

    def _parse_info(self):
        cap = cv2.VideoCapture(self.video_path)
        # get FPS
        # (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        # if int(major_ver) < 3:
        #     fps = cap.get(cv2.cv2.CV_CAP_PROP_FPS)
        # else:
        #     fps = cap.get(cv2.CAP_PROP_FPS)
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.video_fps = fps

        _frame_cnt = 0
        while True:
            ret, _frame = cap.read()
            if 0 == _frame_cnt:
                self.video_frame_shape_hw = (_frame.shape[0], _frame.shape[1])
            if not ret:
                break
            _frame_cnt += 1
        self.video_frame_cnt = _frame_cnt

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

        if (self._USE_CACHE is False) or (self.video_frame_cnt != _frames_fn_cnt):
            self._save_frames()

        if frame_idx >= self.video_frame_cnt - 1:
            raise IndexError(_err_msg % (self.video_frame_cnt, frame_idx))
        if os.path.exists(res_fn) is False:
            print("Frames Files Missing: \"%s\". Re-Saving Frames ..." % res_fn)
        return cv2.imread(res_fn)

    def frames_to_video(self, frames: np.ndarray, res_path: str):
        # frames should be of shape (frames_cnt, frame_height, frame_width (, 3))
        assert 3 == len(frames.shape) or 4 == len(frames.shape)
        assert self.video_frame_shape_hw == (frames.shape[1], frames.shape[2])
        # output should be of ".mp4"
        assert ".mp4" == os.path.splitext(res_path)[-1]
        # output directory should be existent
        assert os.path.exists(os.path.split(res_path)[0])

        frames_as_color = (4 == len(frames.shape))
        # if as color image, must be 3-channeled
        if frames_as_color is True:
            assert 3 == frames.shape[3]

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # output as mp4
        cap_fps = self.video_fps  # set FPS

        size = self.video_frame_shape_hw[::-1]  # (height, width) -> (width, height)
        video = cv2.VideoWriter(filename=res_path, fourcc=fourcc, fps=cap_fps, frameSize=size, isColor=frames_as_color)

        for _frame in frames:
            video.write(_frame)
        video.release()
        print("All %d Frames have been Saved to Video \"%s\" (FPS=%d)" % (frames.shape[0], res_path, self.video_fps))


if "__main__" == __name__:
    obj = VideoHandler(video_path="data/1.mp4")
    t_frames = np.array([obj.get_frame_by_idx(_idx) for _idx in range(100)])
    obj.frames_to_video(frames=t_frames, res_path="res/test.mp4")
    print()
