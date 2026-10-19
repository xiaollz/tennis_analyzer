#!/usr/bin/env python3
"""Objective clip screen with YOLO11 pose (COCO-17): is there a clearly visible person who makes repeated fast arm swings?

usage: /workspace/venv_pose/bin/python clip_pose_screen.py <clip.mp4> [--json out.json]
For each tracked person: median body height as % of frame height, fraction of frames where shoulders+hips are confident,
and the number of fast wrist-speed peaks (speed in body-heights per second). The 'swinger' is the track with the most peaks.
Criteria (this script only reports; the verdict rule is in `verdict()`): swinger height >= 25 %, >= 3 peaks, visible >= 60 % of frames.
No model weights are committed; the weights file is read from /workspace/models/yolo11m-pose.pt.
"""
import json
import sys
from pathlib import Path

import numpy as np

WEIGHTS = "/workspace/models/yolo11m-pose.pt"
STRIDE = 2  # frames


def analyse(path):
    from ultralytics import YOLO
    model = YOLO(WEIGHTS)
    import cv2
    cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0; H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT); cap.release()
    tracks = {}
    res = model.track(source=str(path), stream=True, persist=True, verbose=False, vid_stride=STRIDE, conf=0.3)
    for fi, r in enumerate(res):
        if r.boxes is None or r.boxes.id is None:
            continue
        ids = r.boxes.id.int().tolist(); kp = r.keypoints.data.cpu().numpy(); bx = r.boxes.xyxy.cpu().numpy()
        for tid, k, b in zip(ids, kp, bx):
            tracks.setdefault(tid, []).append((fi * STRIDE / fps, k, b))
    out = []
    for tid, rows in tracks.items():
        if len(rows) < 8:
            continue
        t = np.array([r[0] for r in rows]); hgt = np.array([r[2][3] - r[2][1] for r in rows])
        body = np.median(hgt)
        vis = np.mean([(k[5][2] > .4 and k[6][2] > .4 and k[11][2] > .4 and k[12][2] > .4) for _, k, _ in rows])
        sp = []
        for wi in (9, 10):
            w = np.array([k[wi][:2] for _, k, _ in rows]); c = np.array([k[wi][2] for _, k, _ in rows])
            d = np.linalg.norm(np.diff(w, axis=0), axis=1) / np.maximum(np.diff(t), 1e-3) / max(body, 1)
            d[(c[1:] < .3) | (c[:-1] < .3)] = 0
            sp.append(d)
        s = np.maximum(sp[0], sp[1])
        s = np.convolve(s, np.ones(3) / 3, mode="same")
        thr = 1.5  # body heights per second
        peaks, last = 0, -9
        for i in range(1, len(s) - 1):
            if s[i] > thr and s[i] >= s[i - 1] and s[i] >= s[i + 1] and t[i + 1] - last > 0.6:
                peaks += 1; last = t[i + 1]
        out.append({"track": int(tid), "frames": len(rows), "height_pct": round(100 * body / H), "visible_frac": round(float(vis), 2), "swing_peaks": peaks,
                    "peak_speed": round(float(s.max()), 2) if len(s) else 0.0})
    out.sort(key=lambda x: -x["swing_peaks"])
    return {"clip": Path(path).name, "fps": fps, "persons": out, "verdict": verdict(out)}


def verdict(persons):
    for p in persons:
        if p["height_pct"] >= 25 and p["swing_peaks"] >= 3 and p["visible_frac"] >= 0.6:
            return "usable"
    return "reject"


if __name__ == "__main__":
    r = analyse(sys.argv[1])
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(r, indent=1))
    print(json.dumps(r))
