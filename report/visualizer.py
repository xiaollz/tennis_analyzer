"""可视化工具模块。

提供：
    - ``SkeletonDrawer``: 在视频帧上绘制 COCO-17 骨骼。
    - ``TrajectoryDrawer``: 绘制任意关节的轨迹拖尾（支持消失时间）。
    - ``ChartGenerator``: 生成 Matplotlib 分析图表。
"""

from __future__ import annotations

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from config.keypoints import (
    SKELETON_CONNECTIONS,
    CONNECTION_COLORS,
    KEYPOINT_TO_PART,
    KEYPOINT_COLORS,
    FACE_KEYPOINTS,
    KEYPOINT_NAMES,
)
from evaluation.kpi import KPIResult

# ---------- 中文字体支持 ----------
_CN_FONT_PATH = None
for p in [
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
]:
    if Path(p).exists():
        _CN_FONT_PATH = p
        break

if _CN_FONT_PATH:
    font_manager.fontManager.addfont(_CN_FONT_PATH)
    _cn_font_name = font_manager.FontProperties(fname=_CN_FONT_PATH).get_name()
    plt.rcParams["font.family"] = _cn_font_name
    plt.rcParams["axes.unicode_minus"] = False

# 关节名称中文映射
JOINT_CN = {
    "right_wrist": "右手腕", "left_wrist": "左手腕",
    "right_elbow": "右肘", "left_elbow": "左肘",
    "right_shoulder": "右肩", "left_shoulder": "左肩",
    "right_hip": "右髋", "left_hip": "左髋",
    "right_knee": "右膝", "left_knee": "左膝",
    "right_ankle": "右踝", "left_ankle": "左踝",
    "nose": "鼻子",
}


# =====================================================================
# Skeleton drawing
# =====================================================================

class SkeletonDrawer:
    """在 BGR 帧上绘制 COCO-17 姿态骨骼。"""

    def __init__(
        self,
        line_thickness: int = 1,
        point_radius: int = 3,
        confidence_threshold: float = 0.5,
        draw_face: bool = False,
    ):
        self.line_thickness = line_thickness
        self.point_radius = point_radius
        self.confidence_threshold = confidence_threshold
        self.draw_face = draw_face

    def draw(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        color_override: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        frame = frame.copy()
        self._draw_connections(frame, keypoints, confidence, color_override)
        self._draw_keypoints(frame, keypoints, confidence, color_override)
        return frame

    def _draw_connections(self, frame, keypoints, confidence, color_override):
        for idx, (s, e) in enumerate(SKELETON_CONNECTIONS):
            if confidence[s] < self.confidence_threshold or confidence[e] < self.confidence_threshold:
                continue
            sp = tuple(keypoints[s].astype(int))
            ep = tuple(keypoints[e].astype(int))
            if sp == (0, 0) or ep == (0, 0):
                continue
            color = color_override or CONNECTION_COLORS[idx]
            cv2.line(frame, sp, ep, color, self.line_thickness, cv2.LINE_AA)

    def _draw_keypoints(self, frame, keypoints, confidence, color_override):
        for idx, (pt, conf) in enumerate(zip(keypoints, confidence)):
            if not self.draw_face and idx in FACE_KEYPOINTS:
                continue
            if conf < self.confidence_threshold:
                continue
            center = tuple(pt.astype(int))
            if center == (0, 0):
                continue
            part = KEYPOINT_TO_PART.get(idx, "torso")
            color = color_override or KEYPOINT_COLORS[part]
            cv2.circle(frame, center, self.point_radius, color, -1, cv2.LINE_AA)


# =====================================================================
# Trajectory drawing (any joint) — 带消失时间
# =====================================================================

class TrajectoryDrawer:
    """绘制关节轨迹拖尾，支持消失时间（max_trail 帧后自动消失）。

    默认只追踪 1 个关节（持拍手腕），最多保留 30 帧轨迹。
    """

    def __init__(
        self,
        joint: str | int = "right_wrist",
        max_trail: int = 30,
        color: Tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2,
        min_distance: int = 3,
        fade: bool = True,
    ):
        if isinstance(joint, str):
            self.joint_idx = KEYPOINT_NAMES[joint]
            self.joint_name = joint
        else:
            self.joint_idx = joint
            self.joint_name = {v: k for k, v in KEYPOINT_NAMES.items()}.get(joint, f"kp_{joint}")

        self.max_trail = max_trail
        self.color = color
        self.thickness = thickness
        self.min_distance = min_distance
        self.fade = fade
        # 存储 (x, y, frame_idx) 以支持按帧消失
        self.trail: List[Tuple[int, int, int]] = []
        self._current_frame = 0

    def update(self, keypoints: np.ndarray, confidence: np.ndarray, frame_idx: int = -1):
        """记录当前帧的关节位置。"""
        self._current_frame = frame_idx if frame_idx >= 0 else self._current_frame + 1

        if confidence[self.joint_idx] < 0.3:
            return
        pos = tuple(keypoints[self.joint_idx].astype(int))
        if self.trail:
            last = self.trail[-1]
            dist = np.sqrt((pos[0] - last[0]) ** 2 + (pos[1] - last[1]) ** 2)
            if dist < self.min_distance:
                return
        self.trail.append((pos[0], pos[1], self._current_frame))
        # 按帧数消失
        self._prune()

    def _prune(self):
        """移除超过 max_trail 帧的旧轨迹点。"""
        cutoff = self._current_frame - self.max_trail
        while self.trail and self.trail[0][2] < cutoff:
            self.trail.pop(0)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """在帧上绘制累积的轨迹。"""
        if len(self.trail) < 2:
            return frame
        frame = frame.copy()
        n = len(self.trail)
        for i in range(1, n):
            if self.fade:
                alpha = i / n
                color = tuple(int(c * alpha) for c in self.color)
                thick = max(1, int(self.thickness * alpha))
            else:
                color = self.color
                thick = self.thickness
            pt1 = (self.trail[i - 1][0], self.trail[i - 1][1])
            pt2 = (self.trail[i][0], self.trail[i][1])
            cv2.line(frame, pt1, pt2, color, thick, cv2.LINE_AA)
        # 当前位置画点
        last = self.trail[-1]
        cv2.circle(frame, (last[0], last[1]), self.thickness + 2, self.color, -1, cv2.LINE_AA)
        return frame

    def clear(self):
        self.trail.clear()
        self._current_frame = 0


# =====================================================================
# Chart generation for reports — 全中文
# =====================================================================

class ChartGenerator:
    """生成 Matplotlib 分析图表。"""

    PHASE_COLORS = {
        "preparation": "#4CAF50",
        "loading": "#8BC34A",
        "kinetic_chain": "#FF9800",
        "contact": "#F44336",
        "extension": "#2196F3",
        "balance": "#9C27B0",
        # 单反阶段颜色
        "ohb_preparation": "#4CAF50",
        "ohb_backswing": "#8BC34A",
        "ohb_kinetic_chain": "#FF9800",
        "ohb_contact": "#F44336",
        "ohb_extension": "#2196F3",
        "ohb_balance": "#9C27B0",
    }

    # 正手阶段标签
    PHASE_LABELS = {
        "preparation": "准备\n& 转体",
        "loading": "蓄力\n& 落拍",
        "kinetic_chain": "动力链\n& 挥拍",
        "contact": "击球点",
        "extension": "延伸\n& 随挥",
        "balance": "平衡\n& 恢复",
        # 单反阶段标签
        "ohb_preparation": "准备\n& 侧身",
        "ohb_backswing": "引拍\n& L形",
        "ohb_kinetic_chain": "动力链\n& 前挥",
        "ohb_contact": "击球点\n& 伸展",
        "ohb_extension": "ATA收拍\n& 保持侧身",
        "ohb_balance": "平衡\n& 恢复",
    }

    @staticmethod
    def radar_chart(
        phase_scores: Dict[str, float],
        output_path: str,
        title: str = "各阶段评分雷达图",
        swing_idx: Optional[int] = None,
        phase_labels: Optional[Dict[str, str]] = None,
    ) -> str:
        """生成阶段评分雷达图。自动检测正手/单反阶段。"""
        labels = []
        values = []
        # 自动检测阶段类型：如果 phase_scores 中包含 ohb_ 前缀则是单反
        all_phases = list(phase_scores.keys())
        if any(p.startswith("ohb_") for p in all_phases):
            phase_order = ["ohb_preparation", "ohb_backswing", "ohb_kinetic_chain",
                           "ohb_contact", "ohb_extension", "ohb_balance"]
        else:
            phase_order = ["preparation", "kinetic_chain", "contact", "extension", "balance"]
        for phase in phase_order:
            if phase in phase_scores:
                labels.append(ChartGenerator.PHASE_LABELS.get(phase, phase))
                values.append(phase_scores[phase])

        if not labels:
            return ""

        n = len(labels)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        values_plot = values + [values[0]]
        angles_plot = angles + [angles[0]]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.fill(angles_plot, values_plot, alpha=0.25, color="#2196F3")
        ax.plot(angles_plot, values_plot, "o-", linewidth=2, color="#2196F3")

        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=9, color="gray")

        display_title = title
        if swing_idx is not None:
            display_title = f"第 {swing_idx + 1} 次击球 — {title}"
        ax.set_title(display_title, fontsize=14, fontweight="bold", pad=20)

        for angle, val in zip(angles, values):
            ax.annotate(
                f"{val:.0f}",
                xy=(angle, val),
                fontsize=12, fontweight="bold", ha="center", va="bottom",
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return output_path

    @staticmethod
    def kpi_bar_chart(
        kpi_results: List[KPIResult],
        output_path: str,
        title: str = "KPI 评分详情",
        swing_idx: Optional[int] = None,
    ) -> str:
        """生成 KPI 水平条形图。"""
        valid = [k for k in kpi_results if k.rating not in ("n/a", "无数据")]
        if not valid:
            return ""

        valid.sort(key=lambda k: k.score, reverse=True)
        names = [f"{k.kpi_id} {k.name}" for k in valid]
        scores = [k.score for k in valid]
        colors = [ChartGenerator.PHASE_COLORS.get(k.phase, "#666") for k in valid]

        fig, ax = plt.subplots(figsize=(10, max(4, len(valid) * 0.5)))
        bars = ax.barh(range(len(valid)), scores, color=colors, edgecolor="white", height=0.7)
        ax.set_yticks(range(len(valid)))
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlim(0, 105)
        ax.set_xlabel("分数", fontsize=12)

        display_title = title
        if swing_idx is not None:
            display_title = f"第 {swing_idx + 1} 次击球 — {title}"
        ax.set_title(display_title, fontsize=14, fontweight="bold")
        ax.invert_yaxis()

        for bar, score in zip(bars, scores):
            ax.text(
                score + 1, bar.get_y() + bar.get_height() / 2,
                f"{score:.0f}", va="center", fontsize=10, fontweight="bold",
            )

        # 只显示实际使用的阶段
        used_phases = set(k.phase for k in valid)
        patches = [
            mpatches.Patch(color=c, label=ChartGenerator.PHASE_LABELS.get(p, p).replace("\n", " "))
            for p, c in ChartGenerator.PHASE_COLORS.items()
            if p in used_phases
        ]
        if patches:
            ax.legend(handles=patches, loc="lower right", fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return output_path

    @staticmethod
    def joint_trajectory_chart(
        positions: np.ndarray,
        frame_indices: List[int],
        joint_name: str,
        output_path: str,
        impact_frames: Optional[List[int]] = None,
    ) -> str:
        """绘制关节 x/y 轨迹随时间变化图。"""
        if len(positions) < 2:
            return ""

        cn_name = JOINT_CN.get(joint_name, joint_name.replace("_", " ").title())

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        frames = frame_indices[:len(positions)]
        ax1.plot(frames, positions[:, 0], "b-", linewidth=1.5, label="X (水平)")
        ax2.plot(frames, positions[:, 1], "r-", linewidth=1.5, label="Y (垂直)")

        if impact_frames:
            for imp_f in impact_frames:
                for ax in (ax1, ax2):
                    ax.axvline(imp_f, color="green", linestyle="--", linewidth=2, alpha=0.7)
            # 只添加一次 legend
            ax1.axvline(impact_frames[0], color="green", linestyle="--", linewidth=0, label="击球点")

        ax1.set_ylabel("X 位置 (px)", fontsize=11)
        ax2.set_ylabel("Y 位置 (px)", fontsize=11)
        ax2.set_xlabel("帧", fontsize=11)
        ax1.set_title(f"{cn_name} 轨迹", fontsize=13, fontweight="bold")
        ax1.legend(fontsize=9)
        ax2.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return output_path

    @staticmethod
    def speed_profile_chart(
        speeds: np.ndarray,
        frame_indices: List[int],
        joint_name: str,
        output_path: str,
        impact_frames: Optional[List[int]] = None,
    ) -> str:
        """绘制关节速度曲线。"""
        if len(speeds) < 2:
            return ""

        cn_name = JOINT_CN.get(joint_name, joint_name.replace("_", " ").title())

        fig, ax = plt.subplots(figsize=(12, 4))
        frames = frame_indices[:len(speeds)]
        ax.plot(frames, speeds, "b-", linewidth=1.5)
        ax.fill_between(frames, speeds, alpha=0.2, color="blue")

        if impact_frames:
            for i, imp_f in enumerate(impact_frames):
                label = "击球点" if i == 0 else None
                ax.axvline(imp_f, color="red", linestyle="--", linewidth=2, label=label, alpha=0.7)

        ax.set_xlabel("帧", fontsize=11)
        ax.set_ylabel("速度 (px/s)", fontsize=11)
        ax.set_title(f"{cn_name} 速度曲线", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return output_path

    @staticmethod
    def angle_timeline_chart(
        angles: List[float],
        frame_indices: List[int],
        angle_name: str,
        output_path: str,
        ideal_range: Optional[Tuple[float, float]] = None,
        impact_frames: Optional[List[int]] = None,
    ) -> str:
        """绘制角度随时间变化图。"""
        if len(angles) < 2:
            return ""

        fig, ax = plt.subplots(figsize=(12, 4))
        frames = frame_indices[:len(angles)]
        ax.plot(frames, angles, "b-", linewidth=1.5)

        if ideal_range is not None:
            ax.axhspan(ideal_range[0], ideal_range[1], alpha=0.15, color="green", label="理想范围")

        if impact_frames:
            for i, imp_f in enumerate(impact_frames):
                label = "击球点" if i == 0 else None
                ax.axvline(imp_f, color="red", linestyle="--", linewidth=2, label=label, alpha=0.7)

        ax.set_xlabel("帧", fontsize=11)
        ax.set_ylabel(f"{angle_name} (°)", fontsize=11)
        ax.set_title(f"{angle_name} 变化曲线", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return output_path

    @staticmethod
    def multi_swing_summary_chart(
        swing_scores: List[Tuple[int, float]],
        output_path: str,
    ) -> str:
        """绘制多次击球综合评分对比图。"""
        if not swing_scores:
            return ""

        fig, ax = plt.subplots(figsize=(10, 5))
        indices = list(range(1, len(swing_scores) + 1))
        scores = [s[1] for s in swing_scores]
        colors = []
        for sc in scores:
            if sc >= 70:
                colors.append("#4CAF50")
            elif sc >= 50:
                colors.append("#FF9800")
            else:
                colors.append("#F44336")

        bars = ax.bar(indices, scores, color=colors, edgecolor="white", width=0.6)
        ax.set_xticks(indices)
        ax.set_xticklabels([f"第{i}次" for i in indices], fontsize=11)
        ax.set_ylim(0, 105)
        ax.set_ylabel("综合评分", fontsize=12)
        ax.set_title("各次击球评分对比", fontsize=14, fontweight="bold")
        ax.axhline(y=70, color="green", linestyle="--", alpha=0.5, label="良好线 (70)")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2, score + 1,
                f"{score:.0f}", ha="center", fontsize=12, fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return output_path
