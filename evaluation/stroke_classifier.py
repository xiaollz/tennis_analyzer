"""挥拍类型自动分类器 — 正手 / 单反 / 双反。

通过分析击球前后的关键点运动模式，自动判断每次挥拍是正手还是反手。

判断逻辑（基于 2D 姿态估计）：
    1. **持拍手腕位置**：正手时持拍手腕在身体前方（同侧），
       反手时持拍手腕穿过身体到对侧。
    2. **肩部旋转方向**：正手和反手的肩部旋转方向相反。
    3. **手腕运动方向**：击球时手腕的水平运动方向不同。
    4. **非持拍手位置**：单反时非持拍手向后伸展平衡。

支持：
    - 自动分类（基于启发式规则）
    - 用户手动指定（覆盖自动分类）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from enum import Enum

import numpy as np

from config.keypoints import KEYPOINT_NAMES


class StrokeType(Enum):
    """挥拍类型枚举。"""
    FOREHAND = "forehand"
    ONE_HANDED_BACKHAND = "one_handed_backhand"
    TWO_HANDED_BACKHAND = "two_handed_backhand"
    UNKNOWN = "unknown"

    @property
    def cn_name(self) -> str:
        names = {
            "forehand": "正手",
            "one_handed_backhand": "单手反拍",
            "two_handed_backhand": "双手反拍",
            "unknown": "未知",
        }
        return names.get(self.value, "未知")


@dataclass
class StrokeClassification:
    """单次挥拍的分类结果。"""
    stroke_type: StrokeType
    confidence: float           # 0.0 - 1.0
    reasoning: str              # 判断依据说明
    wrist_cross_ratio: float    # 手腕穿越身体中线的比例
    shoulder_rotation_dir: str  # "cw" / "ccw" / "unknown"


class StrokeClassifier:
    """基于 2D 姿态估计的挥拍类型分类器。

    核心判断逻辑：
    1. 在击球前的准备阶段，检查持拍手腕相对于身体中线（肩中心）的位置。
       - 正手：手腕在持拍侧（右手持拍 → 手腕在右侧）
       - 反手：手腕穿越到非持拍侧
    2. 击球时手腕的运动方向：
       - 正手：从持拍侧向对侧挥动
       - 反手：从非持拍侧向持拍侧挥动
    3. 非持拍手在击球时的位置（区分单反/双反）：
       - 单反：非持拍手远离持拍手（向后伸展平衡）
       - 双反：非持拍手靠近持拍手（双手握拍）
    """

    def __init__(
        self,
        is_right_handed: bool = True,
        min_confidence: float = 0.3,
        prep_window_frames: int = 15,
    ):
        self.is_right_handed = is_right_handed
        self.min_conf = min_confidence
        self.prep_window = prep_window_frames

        # 关键点索引
        self.dom_wrist = KEYPOINT_NAMES["right_wrist" if is_right_handed else "left_wrist"]
        self.non_dom_wrist = KEYPOINT_NAMES["left_wrist" if is_right_handed else "right_wrist"]
        self.dom_shoulder = KEYPOINT_NAMES["right_shoulder" if is_right_handed else "left_shoulder"]
        self.non_dom_shoulder = KEYPOINT_NAMES["left_shoulder" if is_right_handed else "right_shoulder"]
        self.dom_elbow = KEYPOINT_NAMES["right_elbow" if is_right_handed else "left_elbow"]
        self.non_dom_elbow = KEYPOINT_NAMES["left_elbow" if is_right_handed else "right_elbow"]
        self.l_shoulder = KEYPOINT_NAMES["left_shoulder"]
        self.r_shoulder = KEYPOINT_NAMES["right_shoulder"]
        self.l_hip = KEYPOINT_NAMES["left_hip"]
        self.r_hip = KEYPOINT_NAMES["right_hip"]

    def classify_swing(
        self,
        keypoints_series: List[np.ndarray],
        confidence_series: List[np.ndarray],
        frame_indices: List[int],
        impact_frame: int,
        prep_start_frame: Optional[int] = None,
    ) -> StrokeClassification:
        """对单次挥拍进行分类。

        Parameters
        ----------
        keypoints_series : list of (17, 2) arrays
        confidence_series : list of (17,) arrays
        frame_indices : list of int
        impact_frame : int — 击球帧
        prep_start_frame : int, optional — 准备开始帧

        Returns
        -------
        StrokeClassification
        """
        f2p = {f: i for i, f in enumerate(frame_indices)}
        impact_pos = f2p.get(impact_frame)
        if impact_pos is None:
            return StrokeClassification(
                StrokeType.UNKNOWN, 0.0, "无法定位击球帧", 0.0, "unknown"
            )

        # 确定准备阶段范围
        if prep_start_frame is not None and prep_start_frame in f2p:
            prep_pos = f2p[prep_start_frame]
        else:
            prep_pos = max(0, impact_pos - self.prep_window)

        # ── 特征 1：准备阶段手腕相对身体中线的位置 ──────────────
        cross_count = 0
        total_count = 0
        for i in range(prep_pos, min(impact_pos + 1, len(keypoints_series))):
            kp, conf = keypoints_series[i], confidence_series[i]
            wrist_side = self._wrist_side_of_body(kp, conf)
            if wrist_side is not None:
                total_count += 1
                if wrist_side == "cross":
                    cross_count += 1

        wrist_cross_ratio = cross_count / max(total_count, 1)

        # ── 特征 2：击球时手腕运动方向 ────────────────────────
        swing_direction = self._swing_direction(
            keypoints_series, confidence_series, impact_pos
        )

        # ── 特征 3：非持拍手距离（区分单反/双反）────────────────
        hands_distance_norm = self._hands_distance_at_contact(
            keypoints_series, confidence_series, impact_pos
        )

        # ── 综合判断 ──────────────────────────────────────────
        # 判断正手 vs 反手
        is_backhand = False
        reasons = []

        # 手腕穿越身体中线 > 40% 的时间 → 反手
        if wrist_cross_ratio > 0.40:
            is_backhand = True
            reasons.append(f"准备阶段手腕穿越身体中线比例={wrist_cross_ratio:.0%}")

        # 挥拍方向：正手从外向内，反手从内向外
        if swing_direction == "to_dom_side":
            is_backhand = True
            reasons.append("挥拍方向朝持拍侧")
        elif swing_direction == "to_non_dom_side":
            is_backhand = False
            if wrist_cross_ratio <= 0.40:
                reasons.append("挥拍方向朝非持拍侧")

        # 肩部旋转方向
        shoulder_dir = self._shoulder_rotation_direction(
            keypoints_series, confidence_series, prep_pos, impact_pos
        )

        if is_backhand:
            # 区分单反 vs 双反
            if hands_distance_norm is not None and hands_distance_norm > 0.5:
                stroke_type = StrokeType.ONE_HANDED_BACKHAND
                reasons.append(f"击球时双手距离={hands_distance_norm:.2f}（远离→单反）")
            elif hands_distance_norm is not None and hands_distance_norm <= 0.3:
                stroke_type = StrokeType.TWO_HANDED_BACKHAND
                reasons.append(f"击球时双手距离={hands_distance_norm:.2f}（靠近→双反）")
            else:
                # 默认假设单反（因为用户说要学单反）
                stroke_type = StrokeType.ONE_HANDED_BACKHAND
                reasons.append("反手类型无法确定，默认单反")
        else:
            stroke_type = StrokeType.FOREHAND
            reasons.append("正手击球模式")

        # 计算置信度
        confidence = self._compute_confidence(
            wrist_cross_ratio, swing_direction, hands_distance_norm, is_backhand
        )

        return StrokeClassification(
            stroke_type=stroke_type,
            confidence=confidence,
            reasoning="；".join(reasons),
            wrist_cross_ratio=wrist_cross_ratio,
            shoulder_rotation_dir=shoulder_dir,
        )

    # ── 内部特征计算 ──────────────────────────────────────────────

    def _wrist_side_of_body(
        self, kp: np.ndarray, conf: np.ndarray
    ) -> Optional[str]:
        """判断持拍手腕在身体的哪一侧。

        Returns "same" (同侧) 或 "cross" (穿越到对侧) 或 None。
        """
        if conf[self.dom_wrist] < self.min_conf:
            return None
        if conf[self.l_shoulder] < self.min_conf or conf[self.r_shoulder] < self.min_conf:
            return None

        # 身体中线 = 肩中心 x 坐标
        body_center_x = (kp[self.l_shoulder][0] + kp[self.r_shoulder][0]) / 2.0
        wrist_x = kp[self.dom_wrist][0]

        if self.is_right_handed:
            # 右手持拍：手腕在右侧 = same，在左侧 = cross
            return "same" if wrist_x > body_center_x else "cross"
        else:
            # 左手持拍：手腕在左侧 = same，在右侧 = cross
            return "same" if wrist_x < body_center_x else "cross"

    def _swing_direction(
        self,
        kp_series: List[np.ndarray],
        conf_series: List[np.ndarray],
        impact_pos: int,
    ) -> str:
        """判断击球时手腕的水平运动方向。

        Returns "to_dom_side" / "to_non_dom_side" / "unknown"。
        """
        # 取击球前后各 3 帧的手腕 x 位移
        before_pos = max(0, impact_pos - 3)
        after_pos = min(len(kp_series) - 1, impact_pos + 3)

        if before_pos >= after_pos:
            return "unknown"

        before_x = []
        after_x = []
        for i in range(before_pos, impact_pos):
            if conf_series[i][self.dom_wrist] >= self.min_conf:
                before_x.append(float(kp_series[i][self.dom_wrist][0]))
        for i in range(impact_pos, after_pos + 1):
            if conf_series[i][self.dom_wrist] >= self.min_conf:
                after_x.append(float(kp_series[i][self.dom_wrist][0]))

        if not before_x or not after_x:
            return "unknown"

        dx = np.mean(after_x) - np.mean(before_x)

        if self.is_right_handed:
            # 右手持拍：dx > 0 = 向右 = to_dom_side（反手）
            #           dx < 0 = 向左 = to_non_dom_side（正手）
            if dx > 10:
                return "to_dom_side"
            elif dx < -10:
                return "to_non_dom_side"
        else:
            # 左手持拍：dx < 0 = 向左 = to_dom_side（反手）
            if dx < -10:
                return "to_dom_side"
            elif dx > 10:
                return "to_non_dom_side"

        return "unknown"

    def _hands_distance_at_contact(
        self,
        kp_series: List[np.ndarray],
        conf_series: List[np.ndarray],
        impact_pos: int,
    ) -> Optional[float]:
        """击球时两只手腕之间的距离（归一化为肩宽）。

        单反时距离大（非持拍手向后伸展），双反时距离小（双手握拍）。
        """
        # 取击球前后 ±2 帧的平均
        distances = []
        for i in range(max(0, impact_pos - 2), min(len(kp_series), impact_pos + 3)):
            kp, conf = kp_series[i], conf_series[i]
            if (conf[self.dom_wrist] < self.min_conf or
                conf[self.non_dom_wrist] < self.min_conf):
                continue
            if (conf[self.l_shoulder] < self.min_conf or
                conf[self.r_shoulder] < self.min_conf):
                continue

            hand_dist = float(np.linalg.norm(
                kp[self.dom_wrist] - kp[self.non_dom_wrist]
            ))
            shoulder_width = float(np.linalg.norm(
                kp[self.r_shoulder] - kp[self.l_shoulder]
            ))
            if shoulder_width > 10:
                distances.append(hand_dist / shoulder_width)

        return float(np.mean(distances)) if distances else None

    def _shoulder_rotation_direction(
        self,
        kp_series: List[np.ndarray],
        conf_series: List[np.ndarray],
        prep_pos: int,
        impact_pos: int,
    ) -> str:
        """判断肩部旋转方向。"""
        if prep_pos >= impact_pos:
            return "unknown"

        # 准备阶段的肩部角度
        prep_angles = []
        for i in range(prep_pos, min(prep_pos + 5, len(kp_series))):
            angle = self._shoulder_angle(kp_series[i], conf_series[i])
            if angle is not None:
                prep_angles.append(angle)

        # 击球时的肩部角度
        contact_angles = []
        for i in range(max(0, impact_pos - 2), min(impact_pos + 3, len(kp_series))):
            angle = self._shoulder_angle(kp_series[i], conf_series[i])
            if angle is not None:
                contact_angles.append(angle)

        if not prep_angles or not contact_angles:
            return "unknown"

        delta = np.mean(contact_angles) - np.mean(prep_angles)
        if delta > 5:
            return "ccw"  # counter-clockwise
        elif delta < -5:
            return "cw"   # clockwise
        return "unknown"

    def _shoulder_angle(self, kp: np.ndarray, conf: np.ndarray) -> Optional[float]:
        """计算肩部连线相对水平的角度。"""
        if conf[self.l_shoulder] < self.min_conf or conf[self.r_shoulder] < self.min_conf:
            return None
        dx = float(kp[self.r_shoulder][0] - kp[self.l_shoulder][0])
        dy = float(kp[self.r_shoulder][1] - kp[self.l_shoulder][1])
        return float(np.degrees(np.arctan2(dy, dx)))

    def _compute_confidence(
        self,
        wrist_cross_ratio: float,
        swing_direction: str,
        hands_distance: Optional[float],
        is_backhand: bool,
    ) -> float:
        """计算分类置信度。"""
        conf = 0.5  # 基础置信度

        if is_backhand:
            if wrist_cross_ratio > 0.6:
                conf += 0.2
            elif wrist_cross_ratio > 0.4:
                conf += 0.1
            if swing_direction == "to_dom_side":
                conf += 0.15
            if hands_distance is not None:
                if hands_distance > 0.6:
                    conf += 0.15  # 明确单反
                elif hands_distance < 0.3:
                    conf += 0.15  # 明确双反
        else:
            if wrist_cross_ratio < 0.2:
                conf += 0.2
            elif wrist_cross_ratio < 0.4:
                conf += 0.1
            if swing_direction == "to_non_dom_side":
                conf += 0.15

        return min(1.0, conf)

    # ── 批量分类 ──────────────────────────────────────────────────

    def classify_all_swings(
        self,
        keypoints_series: List[np.ndarray],
        confidence_series: List[np.ndarray],
        frame_indices: List[int],
        impact_frames: List[int],
        prep_start_frames: Optional[List[Optional[int]]] = None,
    ) -> List[StrokeClassification]:
        """对视频中所有击球进行分类。"""
        results = []
        for i, impact_frame in enumerate(impact_frames):
            prep_start = (
                prep_start_frames[i] if prep_start_frames and i < len(prep_start_frames)
                else None
            )
            result = self.classify_swing(
                keypoints_series, confidence_series, frame_indices,
                impact_frame, prep_start,
            )
            results.append(result)
        return results
