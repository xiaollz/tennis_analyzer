"""视频处理工具模块。

提供视频读取、自动旋转校正和视频写入功能。
确保无论输入视频是什么方向，输出视频和叠加文字都是正确朝向的。
"""

import cv2
import numpy as np
import subprocess
import json
from pathlib import Path
from typing import Generator, Tuple, Optional


def get_video_rotation(video_path: str) -> int:
    """通过 ffprobe 获取视频元数据中的旋转角度。

    Returns:
        旋转角度（0, 90, 180, 270）
    """
    try:
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            '-select_streams', 'v:0',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            streams = data.get('streams', [])
            if streams:
                side_data = streams[0].get('side_data_list', [])
                for sd in side_data:
                    if 'rotation' in sd:
                        return int(sd['rotation'])
                tags = streams[0].get('tags', {})
                if 'rotate' in tags:
                    return int(tags['rotate'])
    except (FileNotFoundError, json.JSONDecodeError, subprocess.SubprocessError):
        pass
    return 0


def detect_rotation_from_pose(keypoints: np.ndarray, confidence: np.ndarray) -> int:
    """通过姿态关键点检测视频是否需要旋转。

    站立的人应该头在上、脚在下。如果头在脚的左/右侧，说明视频需要旋转。

    Returns:
        建议旋转角度: 0, 90, -90, 或 180
    """
    NOSE = 0
    L_ANKLE, R_ANKLE = 15, 16
    min_conf = 0.3

    if confidence[NOSE] < min_conf:
        return 0
    if confidence[L_ANKLE] < min_conf and confidence[R_ANKLE] < min_conf:
        return 0

    head = keypoints[NOSE]
    if confidence[L_ANKLE] >= min_conf and confidence[R_ANKLE] >= min_conf:
        feet = (keypoints[L_ANKLE] + keypoints[R_ANKLE]) / 2
    elif confidence[L_ANKLE] >= min_conf:
        feet = keypoints[L_ANKLE]
    else:
        feet = keypoints[R_ANKLE]

    dx = head[0] - feet[0]
    dy = head[1] - feet[1]

    if abs(dx) > abs(dy) * 1.5:
        return 90 if dx > 0 else -90
    elif dy > 0:
        return 180
    return 0


def rotate_frame(frame: np.ndarray, rotation: int) -> np.ndarray:
    """根据旋转角度旋转帧。"""
    rotation = rotation % 360
    if rotation == 0:
        return frame
    elif rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    else:
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, -rotation, 1.0)
        return cv2.warpAffine(frame, matrix, (w, h))


class VideoProcessor:
    """视频读取处理器，支持自动旋转校正。

    读取时自动检测并校正旋转，确保输出帧始终是正确朝向的。
    支持两种旋转检测方式：
    1. 元数据旋转（ffprobe）
    2. 姿态检测旋转（前几帧分析）
    """

    def __init__(self, video_path: str, auto_rotate: bool = True):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        self._raw_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._raw_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

        self.auto_rotate = auto_rotate
        self.rotation = get_video_rotation(str(video_path)) if auto_rotate else 0

        # 计算旋转后的实际尺寸
        if abs(self.rotation) in (90, 270):
            self.width = self._raw_height
            self.height = self._raw_width
        else:
            self.width = self._raw_width
            self.height = self._raw_height

    def __del__(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """应用旋转校正。"""
        if self.auto_rotate and self.rotation != 0:
            return rotate_frame(frame, self.rotation)
        return frame

    def detect_rotation_from_content(self, pose_estimator=None) -> int:
        """通过分析前几帧的姿态来检测是否需要额外旋转。

        这在元数据旋转不准确时使用。

        Args:
            pose_estimator: PoseEstimator 实例

        Returns:
            额外需要的旋转角度
        """
        if pose_estimator is None:
            return 0

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        votes = []
        for _ in range(min(10, self.total_frames)):
            ret, frame = self.cap.read()
            if not ret:
                break
            # 先应用元数据旋转
            frame = self._process_frame(frame)
            result = pose_estimator.predict(frame)
            if result["num_persons"] > 0:
                person = result["persons"][0]
                rot = detect_rotation_from_pose(
                    np.array(person["keypoints"]),
                    np.array(person["confidence"]),
                )
                if rot != 0:
                    votes.append(rot)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if votes:
            from collections import Counter
            most_common = Counter(votes).most_common(1)[0]
            if most_common[1] >= 2:  # 至少2帧同意
                return most_common[0]
        return 0

    def apply_additional_rotation(self, extra_rotation: int):
        """应用额外的旋转（在元数据旋转之上）。"""
        if extra_rotation == 0:
            return
        self.rotation = (self.rotation + extra_rotation) % 360
        if abs(self.rotation) in (90, 270):
            self.width = self._raw_height
            self.height = self._raw_width
        else:
            self.width = self._raw_width
            self.height = self._raw_height

    def read_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """生成器：逐帧读取视频（已旋转校正）。"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame_idx, self._process_frame(frame)
            frame_idx += 1

    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """获取指定帧（已旋转校正）。"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            return self._process_frame(frame)
        return None

    @property
    def info(self) -> dict:
        return {
            "path": str(self.video_path),
            "width": self.width,
            "height": self.height,
            "raw_width": self._raw_width,
            "raw_height": self._raw_height,
            "rotation": self.rotation,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration": self.duration,
        }


class VideoWriter:
    """视频写入器，支持音频保留。

    输出的视频已经是正确朝向的（所有旋转在读取时已处理）。
    """

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        codec: str = "mp4v",
        input_path: str = None,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.input_path = input_path

        if input_path:
            self.temp_path = self.output_path.with_suffix('.temp.mp4')
            write_path = self.temp_path
        else:
            self.temp_path = None
            write_path = self.output_path

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(str(write_path), fourcc, fps, (width, height))

        if not self.writer.isOpened():
            raise ValueError(f"无法创建视频写入器: {output_path}")

        self.frame_count = 0

    def write(self, frame: np.ndarray):
        self.writer.write(frame)
        self.frame_count += 1

    def release(self):
        if self.writer is not None:
            self.writer.release()

        if self.temp_path and self.input_path:
            import shutil
            try:
                cmd = [
                    'ffmpeg', '-y',
                    '-i', str(self.temp_path),
                    '-i', str(self.input_path),
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-map', '0:v:0',
                    '-map', '1:a:0?',
                    '-shortest',
                    str(self.output_path)
                ]
                subprocess.run(cmd, capture_output=True, check=True)
                self.temp_path.unlink(missing_ok=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                shutil.move(str(self.temp_path), str(self.output_path))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
