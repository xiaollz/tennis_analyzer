"""Video processing utilities."""

import cv2
import numpy as np
import subprocess
import json
from pathlib import Path
from typing import Generator, Tuple, Optional


def get_video_rotation(video_path: str) -> int:
    """
    Get video rotation from metadata using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        Rotation angle in degrees (0, 90, 180, 270)
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
                # Check for rotation in side_data_list
                side_data = streams[0].get('side_data_list', [])
                for sd in side_data:
                    if 'rotation' in sd:
                        return int(sd['rotation'])
                # Check for rotation tag
                tags = streams[0].get('tags', {})
                if 'rotate' in tags:
                    return int(tags['rotate'])
    except (FileNotFoundError, json.JSONDecodeError, subprocess.SubprocessError):
        pass

    return 0


def detect_rotation_from_pose(keypoints: np.ndarray, confidence: np.ndarray) -> int:
    """
    Detect if video needs rotation based on pose orientation.

    A standing person should have head above feet (lower y value).
    If head is to the left/right of feet, video needs rotation.

    Args:
        keypoints: (17, 2) array of keypoint coordinates
        confidence: (17,) confidence scores

    Returns:
        Suggested rotation: 0, 90, -90, or 180
    """
    # Key indices
    NOSE = 0
    L_ANKLE, R_ANKLE = 15, 16
    L_HIP, R_HIP = 11, 12

    min_conf = 0.3

    # Check if we have enough confident keypoints
    if (confidence[NOSE] < min_conf or
        (confidence[L_ANKLE] < min_conf and confidence[R_ANKLE] < min_conf)):
        return 0

    # Get head position (nose)
    head = keypoints[NOSE]

    # Get feet position (average of ankles)
    if confidence[L_ANKLE] >= min_conf and confidence[R_ANKLE] >= min_conf:
        feet = (keypoints[L_ANKLE] + keypoints[R_ANKLE]) / 2
    elif confidence[L_ANKLE] >= min_conf:
        feet = keypoints[L_ANKLE]
    else:
        feet = keypoints[R_ANKLE]

    # Calculate direction from feet to head
    dx = head[0] - feet[0]
    dy = head[1] - feet[1]

    # In a normal upright video, head should be above feet (dy < 0, since y increases downward)
    # If |dx| > |dy|, the person is horizontal and video needs rotation

    if abs(dx) > abs(dy) * 1.5:  # Person is more horizontal than vertical
        if dx > 0:
            # Head is to the right of feet -> rotate 90 clockwise
            return 90
        else:
            # Head is to the left of feet -> rotate 90 counter-clockwise
            return -90
    elif dy > 0:
        # Head is below feet -> upside down
        return 180

    return 0


def rotate_frame(frame: np.ndarray, rotation: int) -> np.ndarray:
    """
    Rotate frame based on rotation angle.

    Args:
        frame: Input frame
        rotation: Rotation angle (0, 90, 180, 270, -90, -180, -270)

    Returns:
        Rotated frame
    """
    # Normalize rotation to positive
    rotation = rotation % 360

    if rotation == 0:
        return frame
    elif rotation == 90 or rotation == -270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == 180 or rotation == -180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270 or rotation == -90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    else:
        # For non-standard angles, use warpAffine
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, -rotation, 1.0)
        return cv2.warpAffine(frame, matrix, (w, h))


class VideoProcessor:
    """Handle video reading and writing with auto-rotation."""

    def __init__(self, video_path: str, auto_rotate: bool = True):
        """
        Initialize video processor.

        Args:
            video_path: Path to input video file
            auto_rotate: Whether to auto-detect and correct rotation
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Get raw video properties
        self._raw_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._raw_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

        # Detect rotation
        self.auto_rotate = auto_rotate
        self.rotation = get_video_rotation(str(video_path)) if auto_rotate else 0

        # Calculate actual dimensions after rotation
        if self.rotation in (90, 270, -90, -270):
            self.width = self._raw_height
            self.height = self._raw_width
        else:
            self.width = self._raw_width
            self.height = self._raw_height

    def __del__(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply rotation if needed."""
        if self.auto_rotate and self.rotation != 0:
            return rotate_frame(frame, self.rotation)
        return frame

    def read_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generator that yields frames from video.

        Yields:
            Tuple of (frame_index, frame) - frames are auto-rotated if needed
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame_idx, self._process_frame(frame)
            frame_idx += 1

    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get a specific frame by index (auto-rotated)."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            return self._process_frame(frame)
        return None

    @property
    def info(self) -> dict:
        """Get video information."""
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
    """Handle video writing with pose overlay and audio preservation."""

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        codec: str = "mp4v",
        input_path: str = None  # Original video path for audio copying
    ):
        """
        Initialize video writer.

        Args:
            output_path: Path for output video
            width: Frame width
            height: Frame height
            fps: Frames per second
            codec: Video codec (mp4v, avc1, etc.)
            input_path: Original video path to copy audio from
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.input_path = input_path

        # Write to temp file first if we need to add audio
        if input_path:
            self.temp_path = self.output_path.with_suffix('.temp.mp4')
            write_path = self.temp_path
        else:
            self.temp_path = None
            write_path = self.output_path

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            str(write_path),
            fourcc,
            fps,
            (width, height)
        )

        if not self.writer.isOpened():
            raise ValueError(f"Cannot create video writer: {output_path}")

        self.frame_count = 0

    def write(self, frame: np.ndarray):
        """Write a frame to the video."""
        self.writer.write(frame)
        self.frame_count += 1

    def release(self):
        """Release the video writer and merge audio if needed."""
        if self.writer is not None:
            self.writer.release()

        # Copy audio from original video using ffmpeg
        if self.temp_path and self.input_path:
            import subprocess
            import shutil

            try:
                # Use ffmpeg to merge video (from temp) with audio (from original)
                cmd = [
                    'ffmpeg', '-y',
                    '-i', str(self.temp_path),  # Video without audio
                    '-i', str(self.input_path),  # Original video with audio
                    '-c:v', 'copy',  # Copy video stream
                    '-c:a', 'aac',  # Encode audio as AAC
                    '-map', '0:v:0',  # Take video from first input
                    '-map', '1:a:0?',  # Take audio from second input (optional)
                    '-shortest',  # Match shortest stream
                    str(self.output_path)
                ]
                subprocess.run(cmd, capture_output=True, check=True)

                # Remove temp file
                self.temp_path.unlink(missing_ok=True)

            except (subprocess.CalledProcessError, FileNotFoundError):
                # ffmpeg failed or not installed, just rename temp to output
                shutil.move(str(self.temp_path), str(self.output_path))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
