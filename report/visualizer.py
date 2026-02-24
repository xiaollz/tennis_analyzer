"""Visualisation utilities for the tennis analyser.

Provides:
    - ``SkeletonDrawer``: draw COCO-17 skeleton on video frames.
    - ``TrajectoryDrawer``: draw any joint's trajectory trail on frames.
    - ``ChartGenerator``: produce Matplotlib charts for the analysis report.
"""

from __future__ import annotations

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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


# =====================================================================
# Skeleton drawing
# =====================================================================

class SkeletonDrawer:
    """Draw COCO-17 pose skeleton on BGR frames."""

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
# Trajectory drawing (any joint)
# =====================================================================

class TrajectoryDrawer:
    """Draw a joint's trajectory trail on video frames.

    Supports tracking any COCO keypoint by name or index.
    """

    def __init__(
        self,
        joint: str | int = "right_wrist",
        max_trail: int = 120,
        color: Tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2,
        min_distance: int = 4,
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
        self.trail: List[Tuple[int, int]] = []

    def update(self, keypoints: np.ndarray, confidence: np.ndarray):
        """Record the joint position from the current frame."""
        if confidence[self.joint_idx] < 0.3:
            return
        pos = tuple(keypoints[self.joint_idx].astype(int))
        if self.trail:
            last = self.trail[-1]
            dist = np.sqrt((pos[0] - last[0]) ** 2 + (pos[1] - last[1]) ** 2)
            if dist < self.min_distance:
                return
        self.trail.append(pos)
        if len(self.trail) > self.max_trail:
            self.trail.pop(0)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw the accumulated trail on the frame."""
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
            cv2.line(frame, self.trail[i - 1], self.trail[i], color, thick, cv2.LINE_AA)
        # Draw current position as a dot
        cv2.circle(frame, self.trail[-1], self.thickness + 2, self.color, -1, cv2.LINE_AA)
        return frame

    def clear(self):
        self.trail.clear()


# =====================================================================
# Chart generation for reports
# =====================================================================

class ChartGenerator:
    """Generate Matplotlib charts for the evaluation report."""

    PHASE_COLORS = {
        "preparation": "#4CAF50",
        "loading": "#8BC34A",
        "kinetic_chain": "#FF9800",
        "contact": "#F44336",
        "extension": "#2196F3",
        "balance": "#9C27B0",
    }

    PHASE_LABELS = {
        "preparation": "Preparation\n& Unit Turn",
        "loading": "Loading\n& Lag",
        "kinetic_chain": "Kinetic\nChain",
        "contact": "Contact\nPoint",
        "extension": "Extension &\nFollow-Through",
        "balance": "Balance &\nRecovery",
    }

    @staticmethod
    def radar_chart(
        phase_scores: Dict[str, float],
        output_path: str,
        title: str = "Modern Forehand Phase Scores",
    ) -> str:
        """Generate a radar/spider chart of phase scores."""
        labels = []
        values = []
        colors = []
        for phase in ["preparation", "kinetic_chain", "contact", "extension", "balance"]:
            if phase in phase_scores:
                labels.append(ChartGenerator.PHASE_LABELS.get(phase, phase))
                values.append(phase_scores[phase])
                colors.append(ChartGenerator.PHASE_COLORS.get(phase, "#666"))

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
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

        # Add score labels
        for angle, val in zip(angles, values):
            ax.annotate(
                f"{val:.0f}",
                xy=(angle, val),
                fontsize=12,
                fontweight="bold",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return output_path

    @staticmethod
    def kpi_bar_chart(
        kpi_results: List[KPIResult],
        output_path: str,
        title: str = "KPI Scores",
    ) -> str:
        """Generate a horizontal bar chart of all KPI scores."""
        valid = [k for k in kpi_results if k.rating != "n/a"]
        if not valid:
            return ""

        valid.sort(key=lambda k: k.score, reverse=True)
        names = [f"{k.kpi_id} {k.name}" for k in valid]
        scores = [k.score for k in valid]
        colors = []
        for k in valid:
            c = ChartGenerator.PHASE_COLORS.get(k.phase, "#666")
            colors.append(c)

        fig, ax = plt.subplots(figsize=(10, max(4, len(valid) * 0.5)))
        bars = ax.barh(range(len(valid)), scores, color=colors, edgecolor="white", height=0.7)
        ax.set_yticks(range(len(valid)))
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlim(0, 105)
        ax.set_xlabel("Score", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.invert_yaxis()

        for bar, score in zip(bars, scores):
            ax.text(
                score + 1, bar.get_y() + bar.get_height() / 2,
                f"{score:.0f}", va="center", fontsize=10, fontweight="bold",
            )

        # Legend for phases
        patches = [
            mpatches.Patch(color=c, label=ChartGenerator.PHASE_LABELS.get(p, p).replace("\n", " "))
            for p, c in ChartGenerator.PHASE_COLORS.items()
        ]
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
        impact_frame: Optional[int] = None,
    ) -> str:
        """Plot a joint's x,y trajectory over time."""
        if len(positions) < 2:
            return ""

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        frames = frame_indices[:len(positions)]
        ax1.plot(frames, positions[:, 0], "b-", linewidth=1.5, label="X (horizontal)")
        ax2.plot(frames, positions[:, 1], "r-", linewidth=1.5, label="Y (vertical)")

        if impact_frame is not None:
            for ax in (ax1, ax2):
                ax.axvline(impact_frame, color="green", linestyle="--", linewidth=2, label="Impact")

        ax1.set_ylabel("X position (px)", fontsize=11)
        ax2.set_ylabel("Y position (px)", fontsize=11)
        ax2.set_xlabel("Frame", fontsize=11)
        ax1.set_title(f"{joint_name} Trajectory", fontsize=13, fontweight="bold")
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
        impact_frame: Optional[int] = None,
    ) -> str:
        """Plot a joint's speed profile over time."""
        if len(speeds) < 2:
            return ""

        fig, ax = plt.subplots(figsize=(12, 4))
        frames = frame_indices[:len(speeds)]
        ax.plot(frames, speeds, "b-", linewidth=1.5)
        ax.fill_between(frames, speeds, alpha=0.2, color="blue")

        if impact_frame is not None:
            ax.axvline(impact_frame, color="red", linestyle="--", linewidth=2, label="Impact")

        ax.set_xlabel("Frame", fontsize=11)
        ax.set_ylabel("Speed (px/s)", fontsize=11)
        ax.set_title(f"{joint_name} Speed Profile", fontsize=13, fontweight="bold")
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
        impact_frame: Optional[int] = None,
    ) -> str:
        """Plot an angle metric over time with optional ideal range."""
        if len(angles) < 2:
            return ""

        fig, ax = plt.subplots(figsize=(12, 4))
        frames = frame_indices[:len(angles)]
        ax.plot(frames, angles, "b-", linewidth=1.5)

        if ideal_range is not None:
            ax.axhspan(ideal_range[0], ideal_range[1], alpha=0.15, color="green", label="Ideal range")

        if impact_frame is not None:
            ax.axvline(impact_frame, color="red", linestyle="--", linewidth=2, label="Impact")

        ax.set_xlabel("Frame", fontsize=11)
        ax.set_ylabel(f"{angle_name} (°)", fontsize=11)
        ax.set_title(f"{angle_name} Over Time", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return output_path
