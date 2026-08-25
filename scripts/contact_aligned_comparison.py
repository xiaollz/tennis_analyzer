"""Create a contact-aligned coach/user forehand comparison.

The script deliberately uses manually verified contact frames. Automatic impact
detection is useful for finding candidates, but a one-to-three-frame error is
large enough to invalidate racket-face comparisons.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_OFFSETS = (-30, -24, -18, -12, -6, -3, 0, 3, 9, 18)
PHASE_LABELS = (
    "T-500ms",
    "T-400ms",
    "T-300ms",
    "T-200ms",
    "T-100ms",
    "T-50ms",
    "T0 CONTACT",
    "T+50ms",
    "T+150ms",
    "T+300ms",
)


def parse_crop(value: str) -> tuple[float, float, float, float]:
    parts = tuple(float(item) for item in value.split(","))
    if len(parts) != 4 or any(item < 0 or item > 1 for item in parts):
        raise argparse.ArgumentTypeError("crop must be x0,y0,x1,y1 fractions in [0,1]")
    x0, y0, x1, y1 = parts
    if x1 <= x0 or y1 <= y0:
        raise argparse.ArgumentTypeError("crop x1/y1 must be greater than x0/y0")
    return parts


def parse_offsets(value: str) -> tuple[int, ...]:
    offsets = tuple(int(item) for item in value.split(","))
    if not offsets:
        raise argparse.ArgumentTypeError("at least one offset is required")
    return offsets


def read_frame(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
    ok, frame = cap.read()
    if not ok:
        raise ValueError(f"cannot read frame {frame_idx}")
    return frame


def rotate(frame: np.ndarray, mode: str) -> np.ndarray:
    if mode == "cw":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if mode == "ccw":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if mode == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def crop_fraction(
    frame: np.ndarray, crop: tuple[float, float, float, float]
) -> np.ndarray:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = crop
    return frame[
        int(round(y0 * h)) : int(round(y1 * h)),
        int(round(x0 * w)) : int(round(x1 * w)),
    ]


def letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(
        frame,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 20, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def label(frame: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (10, 10, 10), -1)
    cv2.putText(
        frame,
        text,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
        cv2.LINE_AA,
    )


def comparison_cell(
    coach: np.ndarray,
    user: np.ndarray,
    phase: str,
    coach_frame: int,
    user_frame: int,
    panel_width: int,
    panel_height: int,
) -> np.ndarray:
    left = letterbox(coach, panel_width, panel_height)
    right = letterbox(user, panel_width, panel_height)
    label(left, f"COACH  F{coach_frame}", (0, 215, 255))
    label(right, f"USER  F{user_frame}", (80, 255, 110))
    pair = np.hstack((left, right))
    cv2.line(pair, (panel_width, 0), (panel_width, panel_height), (255, 255, 255), 2)
    banner = np.full((42, pair.shape[1], 3), 245, dtype=np.uint8)
    color = (25, 25, 210) if "CONTACT" in phase else (25, 25, 25)
    cv2.putText(
        banner,
        phase,
        (14, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        color,
        2,
        cv2.LINE_AA,
    )
    return np.vstack((banner, pair))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coach-video", required=True)
    parser.add_argument("--user-video", required=True)
    parser.add_argument("--coach-contact", required=True, type=int)
    parser.add_argument("--user-contact", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--coach-rotate", choices=("none", "cw", "ccw", "180"), default="none")
    parser.add_argument("--user-rotate", choices=("none", "cw", "ccw", "180"), default="none")
    parser.add_argument("--coach-crop", type=parse_crop, default=(0.0, 0.0, 1.0, 1.0))
    parser.add_argument("--user-crop", type=parse_crop, default=(0.0, 0.0, 1.0, 1.0))
    parser.add_argument("--offsets", type=parse_offsets, default=DEFAULT_OFFSETS)
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--panel-height", type=int, default=390)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    coach_cap = cv2.VideoCapture(args.coach_video)
    user_cap = cv2.VideoCapture(args.user_video)
    if not coach_cap.isOpened() or not user_cap.isOpened():
        raise ValueError("could not open coach or user video")

    coach_fps = float(coach_cap.get(cv2.CAP_PROP_FPS))
    user_fps = float(user_cap.get(cv2.CAP_PROP_FPS))
    offsets = tuple(args.offsets)
    if offsets == DEFAULT_OFFSETS:
        phases = PHASE_LABELS
    else:
        phases = tuple(f"T{offset / user_fps * 1000:+.0f}ms" for offset in offsets)

    cells: list[np.ndarray] = []
    manifest_frames = []
    for index, (offset, phase) in enumerate(zip(offsets, phases), start=1):
        # Convert the user's relative time to each video's native frame rate.
        relative_seconds = offset / user_fps
        coach_frame_idx = args.coach_contact + int(round(relative_seconds * coach_fps))
        user_frame_idx = args.user_contact + offset

        coach = rotate(read_frame(coach_cap, coach_frame_idx), args.coach_rotate)
        user = rotate(read_frame(user_cap, user_frame_idx), args.user_rotate)
        coach = crop_fraction(coach, args.coach_crop)
        user = crop_fraction(user, args.user_crop)

        cell = comparison_cell(
            coach,
            user,
            phase,
            coach_frame_idx,
            user_frame_idx,
            args.panel_width,
            args.panel_height,
        )
        cells.append(cell)
        frame_path = frames_dir / f"phase_{index:02d}.jpg"
        cv2.imwrite(str(frame_path), cell, [cv2.IMWRITE_JPEG_QUALITY, 95])
        manifest_frames.append(
            {
                "phase": phase,
                "relative_seconds": relative_seconds,
                "coach_frame": coach_frame_idx,
                "user_frame": user_frame_idx,
                "path": str(frame_path),
            }
        )

    # Two phase cells per row keeps the racket face large enough to inspect.
    if len(cells) % 2:
        cells.append(np.full_like(cells[0], 245))
    rows = [np.hstack(cells[i : i + 2]) for i in range(0, len(cells), 2)]
    grid = np.vstack(rows)
    grid_path = output_dir / "contact_aligned_10_phase.jpg"
    cv2.imwrite(str(grid_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 96])

    # Continuous synchronized slow-motion comparison around contact.
    video_start = min(offsets)
    video_end = max(offsets) + 6
    video_path = output_dir / "contact_aligned_slow_motion.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        12.0,
        (args.panel_width * 2, args.panel_height + 42),
    )
    for offset in range(video_start, video_end + 1):
        relative_seconds = offset / user_fps
        coach_idx = args.coach_contact + int(round(relative_seconds * coach_fps))
        user_idx = args.user_contact + offset
        coach = crop_fraction(
            rotate(read_frame(coach_cap, coach_idx), args.coach_rotate), args.coach_crop
        )
        user = crop_fraction(
            rotate(read_frame(user_cap, user_idx), args.user_rotate), args.user_crop
        )
        phase = "T0 CONTACT" if offset == 0 else f"T{relative_seconds * 1000:+.0f}ms"
        writer.write(
            comparison_cell(
                coach,
                user,
                phase,
                coach_idx,
                user_idx,
                args.panel_width,
                args.panel_height,
            )
        )
    writer.release()
    coach_cap.release()
    user_cap.release()

    manifest = {
        "coach_video": args.coach_video,
        "user_video": args.user_video,
        "coach_contact_frame": args.coach_contact,
        "user_contact_frame": args.user_contact,
        "coach_fps": coach_fps,
        "user_fps": user_fps,
        "coach_crop": args.coach_crop,
        "user_crop": args.user_crop,
        "frames": manifest_frames,
        "grid": str(grid_path),
        "video": str(video_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(grid_path)
    print(video_path)


if __name__ == "__main__":
    main()
