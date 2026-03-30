"""Split-screen coach vs user keypoint comparison analysis.

Draws skeletons and wrist/elbow trajectories on both sides,
matching the main pipeline's visual style (line=1px, radius=3px, trail=2px fade).
"""
import sys, os, json
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pose_estimator import PoseEstimator
from config.keypoints import (
    KEYPOINT_NAMES, SKELETON_CONNECTIONS, CONNECTION_COLORS,
    KEYPOINT_TO_PART, KEYPOINT_COLORS, FACE_KEYPOINTS,
)

KP = KEYPOINT_NAMES

# Visual params matching main pipeline
LINE_THICKNESS = 1
POINT_RADIUS = 3
TRAIL_THICKNESS = 2
CONF_THRESHOLD = 0.5

# Trail colors
COLOR_WRIST = (0, 255, 255)    # yellow
COLOR_ELBOW = (200, 200, 0)    # cyan


def _conf_ok(conf, idx, thr=0.3):
    return float(conf[idx]) >= thr


def _vec_angle(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
    return float(np.degrees(np.arccos(cos_a)))


def compute_angles(kp, conf):
    if kp is None or conf is None:
        return {}
    angles = {}
    pairs = [
        ("elbow_R", "right_shoulder", "right_elbow", "right_wrist"),
        ("elbow_L", "left_shoulder", "left_elbow", "left_wrist"),
        ("knee_R", "right_hip", "right_knee", "right_ankle"),
        ("knee_L", "left_hip", "left_knee", "left_ankle"),
    ]
    for name, a, v, b in pairs:
        if all(_conf_ok(conf, KP[k]) for k in [a, v, b]):
            v1 = kp[KP[a]] - kp[KP[v]]
            v2 = kp[KP[b]] - kp[KP[v]]
            angles[name] = _vec_angle(v1, v2)

    if all(_conf_ok(conf, KP[k]) for k in ["left_shoulder", "right_shoulder"]):
        sv = kp[KP["right_shoulder"]] - kp[KP["left_shoulder"]]
        h = np.array([1.0, 0.0])
        angles["shoulder_line"] = float(np.degrees(np.arctan2(h[0]*sv[1]-h[1]*sv[0], np.dot(h, sv))))

    if all(_conf_ok(conf, KP[k]) for k in ["left_hip", "right_hip"]):
        hv = kp[KP["right_hip"]] - kp[KP["left_hip"]]
        h = np.array([1.0, 0.0])
        angles["hip_line"] = float(np.degrees(np.arctan2(h[0]*hv[1]-h[1]*hv[0], np.dot(h, hv))))

    if all(_conf_ok(conf, KP[k]) for k in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]):
        sh = (kp[KP["left_shoulder"]] + kp[KP["right_shoulder"]]) / 2
        hp = (kp[KP["left_hip"]] + kp[KP["right_hip"]]) / 2
        angles["spine_tilt"] = _vec_angle(sh - hp, np.array([0.0, -1.0]))

    if "shoulder_line" in angles and "hip_line" in angles:
        angles["hip_shoulder_sep"] = abs(angles["shoulder_line"] - angles["hip_line"])

    if (_conf_ok(conf, KP["right_wrist"]) and _conf_ok(conf, KP["right_shoulder"])
            and _conf_ok(conf, KP["right_hip"])):
        tl = np.linalg.norm(kp[KP["right_shoulder"]] - kp[KP["right_hip"]])
        if tl > 10:
            angles["wrist_height_norm"] = float((kp[KP["right_shoulder"]][1] - kp[KP["right_wrist"]][1]) / tl)

    return angles


def draw_skeleton_direct(frame, kp, conf, x_offset=0):
    """Draw skeleton directly on frame with x offset (no copy)."""
    for idx, (s, e) in enumerate(SKELETON_CONNECTIONS):
        if conf[s] < CONF_THRESHOLD or conf[e] < CONF_THRESHOLD:
            continue
        sp = (int(kp[s][0]) + x_offset, int(kp[s][1]))
        ep = (int(kp[e][0]) + x_offset, int(kp[e][1]))
        if sp == (x_offset, 0) or ep == (x_offset, 0):
            continue
        cv2.line(frame, sp, ep, CONNECTION_COLORS[idx], LINE_THICKNESS, cv2.LINE_AA)

    for idx, (pt, c) in enumerate(zip(kp, conf)):
        if idx in FACE_KEYPOINTS:
            continue
        if c < CONF_THRESHOLD:
            continue
        center = (int(pt[0]) + x_offset, int(pt[1]))
        if center == (x_offset, 0):
            continue
        part = KEYPOINT_TO_PART.get(idx, "torso")
        cv2.circle(frame, center, POINT_RADIUS, KEYPOINT_COLORS[part], -1, cv2.LINE_AA)


def draw_trail_direct(frame, trail_pts, color, x_offset=0):
    """Draw trajectory trail directly on frame with x offset and fade."""
    if len(trail_pts) < 2:
        return
    n = len(trail_pts)
    for i in range(1, n):
        alpha = 0.4 + 0.6 * (i / n)
        c = tuple(int(v * alpha) for v in color)
        thick = max(1, int(TRAIL_THICKNESS * alpha))
        pt1 = (trail_pts[i-1][0] + x_offset, trail_pts[i-1][1])
        pt2 = (trail_pts[i][0] + x_offset, trail_pts[i][1])
        cv2.line(frame, pt1, pt2, c, thick, cv2.LINE_AA)
    # Current position dot
    last = trail_pts[-1]
    cv2.circle(frame, (last[0] + x_offset, last[1]), TRAIL_THICKNESS + 2, color, -1, cv2.LINE_AA)


def select_person(result, expected_x_center):
    if result["num_persons"] == 0:
        return None, None
    if result["num_persons"] == 1:
        return result["persons"][0]["keypoints"], result["persons"][0]["confidence"]
    best_idx, best_dist = 0, float("inf")
    for i, p in enumerate(result["persons"]):
        cx = (p["bbox"][0] + p["bbox"][2]) / 2
        if abs(cx - expected_x_center) < best_dist:
            best_dist = abs(cx - expected_x_center)
            best_idx = i
    return result["persons"][best_idx]["keypoints"], result["persons"][best_idx]["confidence"]


def update_trail(trail, kp, conf, joint_idx, max_trail=30):
    """Append to trail list, prune old entries."""
    if kp is not None and _conf_ok(conf, joint_idx):
        pt = (int(kp[joint_idx][0]), int(kp[joint_idx][1]))
        if trail and np.sqrt((pt[0]-trail[-1][0])**2 + (pt[1]-trail[-1][1])**2) < 3:
            return  # skip if too close
        trail.append(pt)
    if len(trail) > max_trail:
        trail.pop(0)


def main():
    import sys as _sys
    video_path = _sys.argv[1] if len(_sys.argv) > 1 else "/Users/qsy/Desktop/tennis/videos/606addd62628acf5f2a19fdec79662b0.mp4"
    output_dir = _sys.argv[2] if len(_sys.argv) > 2 else "/Users/qsy/Desktop/tennis/reports/2026-03-25/comparison"
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_x = W // 2

    print(f"Video: {W}x{H}, {fps}fps, {total_frames} frames")
    print(f"Split at x={mid_x}: left=coach, right=user")

    estimator = PoseEstimator()

    # Trail storage
    wrist_idx = KP["right_wrist"]
    elbow_idx = KP["right_elbow"]
    coach_wrist_trail = []
    coach_elbow_trail = []
    user_wrist_trail = []
    user_elbow_trail = []

    # Per-frame storage
    coach_angles_all = []
    user_angles_all = []
    coach_kps_all = []
    user_kps_all = []
    frames_raw_all = []

    # Output video
    out_path = os.path.join(output_dir, "comparison_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_raw_all.append(frame.copy())

        left_half = frame[:, :mid_x].copy()
        right_half = frame[:, mid_x:].copy()

        res_coach = estimator.predict(left_half)
        res_user = estimator.predict(right_half)

        coach_kp, coach_conf = select_person(res_coach, mid_x // 2)
        user_kp, user_conf = select_person(res_user, mid_x // 2)

        coach_angles = compute_angles(coach_kp, coach_conf)
        user_angles = compute_angles(user_kp, user_conf)

        coach_angles_all.append(coach_angles)
        user_angles_all.append(user_angles)
        coach_kps_all.append((coach_kp, coach_conf))
        user_kps_all.append((user_kp, user_conf))

        # Update trails
        update_trail(coach_wrist_trail, coach_kp, coach_conf, wrist_idx)
        update_trail(coach_elbow_trail, coach_kp, coach_conf, elbow_idx)
        update_trail(user_wrist_trail, user_kp, user_conf, wrist_idx)
        update_trail(user_elbow_trail, user_kp, user_conf, elbow_idx)

        # Draw on frame (in-place, no copies)
        vis = frame.copy()

        # Coach: skeleton + trails on left half (x_offset=0)
        if coach_kp is not None:
            draw_skeleton_direct(vis, coach_kp, coach_conf, x_offset=0)
        draw_trail_direct(vis, list(coach_elbow_trail), COLOR_ELBOW, x_offset=0)
        draw_trail_direct(vis, list(coach_wrist_trail), COLOR_WRIST, x_offset=0)

        # User: skeleton + trails on right half (x_offset=mid_x)
        if user_kp is not None:
            draw_skeleton_direct(vis, user_kp, user_conf, x_offset=mid_x)
        draw_trail_direct(vis, list(user_elbow_trail), COLOR_ELBOW, x_offset=mid_x)
        draw_trail_direct(vis, list(user_wrist_trail), COLOR_WRIST, x_offset=mid_x)

        # Divider
        cv2.line(vis, (mid_x, 0), (mid_x, H), (255, 255, 255), 1)

        # Labels + angles
        cv2.putText(vis, "Coach", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, "User", (mid_x+10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1, cv2.LINE_AA)
        y = 34
        for key, label in [("elbow_R", "Elbow"), ("knee_R", "Knee")]:
            cv_s = f"{coach_angles[key]:.0f}" if key in coach_angles and coach_angles[key] is not None else "-"
            uv_s = f"{user_angles[key]:.0f}" if key in user_angles and user_angles[key] is not None else "-"
            cv2.putText(vis, f"{label}:{cv_s}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(vis, f"{label}:{uv_s}", (mid_x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 100), 1, cv2.LINE_AA)
            y += 13

        writer.write(vis)
        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"  Frame {frame_idx}/{total_frames}")

    cap.release()
    writer.release()
    print(f"\nAnnotated video saved: {out_path}")

    # ── 6 keyframes grid ────────────────────────────────────────────
    key_indices = np.linspace(0, total_frames - 1, 6, dtype=int)

    grid_frames = []
    for ki in key_indices:
        frame = frames_raw_all[ki].copy()
        ckp, cconf = coach_kps_all[ki]
        ukp, uconf = user_kps_all[ki]

        if ckp is not None:
            draw_skeleton_direct(frame, ckp, cconf, x_offset=0)
        if ukp is not None:
            draw_skeleton_direct(frame, ukp, uconf, x_offset=mid_x)

        # Cumulative trails up to this frame
        for side_data, x_off in [
            ((coach_kps_all, elbow_idx, COLOR_ELBOW), 0),
            ((coach_kps_all, wrist_idx, COLOR_WRIST), 0),
            ((user_kps_all, elbow_idx, COLOR_ELBOW), mid_x),
            ((user_kps_all, wrist_idx, COLOR_WRIST), mid_x),
        ]:
            kps_list, joint_idx, color = side_data
            pts = []
            for i in range(ki + 1):
                kp, conf = kps_list[i]
                if kp is not None and _conf_ok(conf, joint_idx):
                    pts.append((int(kp[joint_idx][0]), int(kp[joint_idx][1])))
            if len(pts) >= 2:
                n = len(pts)
                for i in range(1, n):
                    alpha = 0.4 + 0.6 * (i / n)
                    c = tuple(int(v * alpha) for v in color)
                    pt1 = (pts[i-1][0] + x_off, pts[i-1][1])
                    pt2 = (pts[i][0] + x_off, pts[i][1])
                    cv2.line(frame, pt1, pt2, c, 2, cv2.LINE_AA)

        cv2.line(frame, (mid_x, 0), (mid_x, H), (255, 255, 255), 1)

        # Angles
        ca = coach_angles_all[ki]
        ua = user_angles_all[ki]
        y = 15
        for key, label in [("elbow_R", "Elbow"), ("knee_R", "Knee"), ("spine_tilt", "Spine")]:
            cv_s = f"{ca[key]:.0f}" if key in ca and ca[key] is not None else "-"
            uv_s = f"{ua[key]:.0f}" if key in ua and ua[key] is not None else "-"
            cv2.putText(frame, f"{label}:{cv_s}", (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{label}:{uv_s}", (mid_x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 255, 100), 1, cv2.LINE_AA)
            y += 11

        time_s = ki / fps
        cv2.putText(frame, f"F{ki} ({time_s:.2f}s)", (mid_x-55, H-6),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)
        grid_frames.append(frame)

    row1 = np.hstack(grid_frames[:3])
    row2 = np.hstack(grid_frames[3:])
    grid = np.vstack([row1, row2])
    grid_path = os.path.join(output_dir, "comparison_keyframes.jpg")
    cv2.imwrite(grid_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Keyframe grid saved: {grid_path}")

    # ── Report ──────────────────────────────────────────────────────
    angle_keys = ["elbow_R", "elbow_L", "knee_R", "knee_L", "shoulder_line", "hip_line",
                  "spine_tilt", "hip_shoulder_sep", "wrist_height_norm"]
    angle_labels = {
        "elbow_R": "右肘角度", "elbow_L": "左肘角度",
        "knee_R": "右膝角度", "knee_L": "左膝角度",
        "shoulder_line": "肩线角度", "hip_line": "髋线角度",
        "spine_tilt": "脊柱倾斜", "hip_shoulder_sep": "髋肩分离角",
        "wrist_height_norm": "手腕高度(归一化)",
    }

    lines = ["# 教练 vs 用户 关键点对比分析\n"]
    lines.append(f"> 视频: 606addd62628acf5f2a19fdec79662b0.mp4")
    lines.append(f"> 分析日期: 2026-03-26\n")
    lines.append("## 全程角度统计对比\n")
    lines.append("| 指标 | 教练 (均值±标准差) | 用户 (均值±标准差) | 差异 | 说明 |")
    lines.append("|------|-------------------|-------------------|------|------|")

    for key in angle_keys:
        label = angle_labels[key]
        cv_list = [a[key] for a in coach_angles_all if key in a and a[key] is not None]
        uv_list = [a[key] for a in user_angles_all if key in a and a[key] is not None]
        if cv_list and uv_list:
            cm, cs = np.mean(cv_list), np.std(cv_list)
            um, us = np.mean(uv_list), np.std(uv_list)
            diff = um - cm
            note = ""
            if key == "elbow_R":
                note = "越小=弯曲越多" if diff < -5 else ("越大=更伸展" if diff > 5 else "接近")
            elif key in ("knee_R", "knee_L"):
                note = "用户膝盖弯更多" if diff < -5 else ("用户腿更直" if diff > 5 else "接近")
            elif key == "spine_tilt":
                note = "用户前倾更多" if diff > 3 else ("用户更直立" if diff < -3 else "接近")
            elif key == "hip_shoulder_sep":
                note = "用户分离更大" if diff > 5 else ("用户分离不足" if diff < -5 else "接近")
            elif key == "wrist_height_norm":
                note = "正值=手腕高于肩"
            lines.append(f"| {label} | {cm:.1f} ± {cs:.1f} | {um:.1f} ± {us:.1f} | {diff:+.1f} | {note} |")
        else:
            c_s = f"{np.mean(cv_list):.1f}" if cv_list else "无数据"
            u_s = f"{np.mean(uv_list):.1f}" if uv_list else "无数据"
            lines.append(f"| {label} | {c_s} | {u_s} | — | 数据不足 |")

    lines.append("\n## 逐帧对比 (6个关键帧)\n")
    for i, ki in enumerate(key_indices):
        t = ki / fps
        ca = coach_angles_all[ki]
        ua = user_angles_all[ki]
        lines.append(f"### 帧 {i+1} (F{ki}, {t:.2f}s)\n")
        lines.append("| 指标 | 教练 | 用户 | 差值 |")
        lines.append("|------|------|------|------|")
        for key in ["elbow_R", "knee_R", "shoulder_line", "spine_tilt", "hip_shoulder_sep", "wrist_height_norm"]:
            label = angle_labels[key]
            cv_val = ca.get(key)
            uv_val = ua.get(key)
            c_s = f"{cv_val:.1f}" if cv_val is not None else "—"
            u_s = f"{uv_val:.1f}" if uv_val is not None else "—"
            d_s = f"{uv_val - cv_val:+.1f}" if cv_val is not None and uv_val is not None else "—"
            lines.append(f"| {label} | {c_s} | {u_s} | {d_s} |")
        lines.append("")

    summary_path = os.path.join(output_dir, "comparison_report.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report saved: {summary_path}")

    raw = {"coach_angles": coach_angles_all, "user_angles": user_angles_all,
           "key_frame_indices": key_indices.tolist(), "fps": fps, "total_frames": total_frames}
    json_path = os.path.join(output_dir, "comparison_data.json")
    with open(json_path, "w") as f:
        json.dump(raw, f, indent=2, default=str)
    print(f"Raw data saved: {json_path}")


if __name__ == "__main__":
    main()
