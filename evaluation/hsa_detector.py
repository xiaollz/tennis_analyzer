"""HSA (Horizontal Shoulder Adduction) Detector.

水平肩内收检测模块 —— 把 5/3 用户重大突破 (HSA = pectoral-humeral angle closing,
胸大肌驱动的核心发力机制) 量化为可从 2D pose 关键点计算的指标。

理论支撑：
- Sasaki et al. 2022 (Sensors): horizontal flexion ≈ 45-48% of forward RHS
- Kovacs & Ellenbecker review: HSA ~25% + ISR ~40% = ~65% of impact velocity
- FTT: closing the angle between upper arm and chest is the chest fire mechanism
- 详见 docs/research/hsa_biomechanics_deep_dive.md

测量原理（2D pose 局限下的最佳近似）：
HSA 几何 = 大臂向量 (shoulder→elbow) 相对躯干横线 (l_shoulder→r_shoulder) 的夹角。
- 引拍极限：HSA 角接近 90-110° (外展)
- 接触瞬间：HSA 角降到 50-80° (闭合中)
- 随挥末端：HSA 角 < 50° (跨过身体)

HSA 失败模式（来自 hsa_kb_audit + diagnosis_engine 模式）：
1. 无闭合 (no_closure): swing 全程 HSA 角差 < 20° → 大臂在画弧而非内收
2. 闭合过晚 (late_closure): 闭合主要发生在接触后 → 没驱动球
3. 闭合过早 (early_closure): 接触前角度已 < 50° → 肘卡身侧推球
4. 静态闭合 (static): 整个 swing HSA 角度变化 < 10° → 纯靠手臂扫
5. 跨胸幅度不足 (insufficient_cross_body): 随挥末端右腕未越过左肩 → HSA 未完成

输出指标（消费方：foundation_layer.py F7_HSA / diagnosis_engine.py / VLM cross-check）：
- hsa_angle_at_phase: dict[phase_name -> angle_deg]
- hsa_total_closure_deg: backswing peak → contact 的角度差 (越大越好)
- hsa_post_contact_closure_deg: contact → follow-through 的角度差
- hsa_closure_rate_deg_per_s: 接触前 100ms 的 HSA 角速度
- hsa_closure_pattern: 上述 5 种失败模式之一 / "healthy"
- cross_body_finish: 随挥末端右腕是否跨过左肩 (bool)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np

from config.keypoints import KEYPOINT_NAMES
from analysis.kinematic_calculator import _vec_angle, _conf_ok, _smooth_series


# =============================================================================
# 常量
# =============================================================================

# HSA 角度阈值（基于 hsa_biomechanics_deep_dive.md 表 2.3 + FTT 视频观察）
HSA_BACKSWING_PEAK_TYPICAL = 95.0   # 引拍顶点典型值（外展位）
HSA_CONTACT_HEALTHY_MAX = 80.0       # 接触瞬间健康上限
HSA_CONTACT_HEALTHY_MIN = 45.0       # 接触瞬间健康下限
HSA_FOLLOW_THROUGH_MAX = 50.0        # 随挥健康上限（应 < 50°）

# 总闭合幅度（backswing peak → contact）
HSA_TOTAL_CLOSURE_HEALTHY = 25.0     # 至少需要 25° 闭合才算 HSA 启动了
HSA_TOTAL_CLOSURE_STRONG = 40.0      # 40°+ 是好的 HSA

# 时间窗口（ms）
HSA_PEAK_VELOCITY_WINDOW_MS = 100.0  # 接触前 100ms 是峰值速度窗

# 置信度阈值
KEYPOINT_CONF_THRESHOLD = 0.3


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class HSAMetrics:
    """单次 swing 的 HSA 检测结果。"""

    # 各阶段角度（度）
    hsa_angle_at_unit_turn_peak: Optional[float] = None
    hsa_angle_at_forward_swing_start: Optional[float] = None
    hsa_angle_at_contact: Optional[float] = None
    hsa_angle_at_follow_through: Optional[float] = None

    # 闭合幅度
    hsa_total_closure_deg: Optional[float] = None       # peak → contact
    hsa_post_contact_closure_deg: Optional[float] = None  # contact → follow-through

    # 速度（接触前 100ms 平均）
    hsa_closure_rate_deg_per_s: Optional[float] = None

    # 跨胸完成
    cross_body_finish: Optional[bool] = None
    cross_body_finish_distance_norm: Optional[float] = None  # 右腕越过左肩的距离 / 肩宽

    # 失败模式（5 选 1 或 healthy）
    hsa_closure_pattern: str = "uncertain"
    pattern_evidence: List[str] = field(default_factory=list)

    # 整体健康分（0-100）
    hsa_health_score: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# =============================================================================
# 单帧几何
# =============================================================================

def hsa_angle_2d(
    keypoints: np.ndarray,
    confidence: np.ndarray,
    is_right_handed: bool = True,
) -> Optional[float]:
    """单帧 HSA 角度（度）。

    定义：大臂向量（持拍肩 → 持拍肘）与躯干横线（左肩 → 右肩）的夹角。

    几何含义：
    - 角度大（≈90°）= 大臂从躯干外展开（HSA 未发生）
    - 角度小（< 50°）= 大臂跨过胸前（HSA 已完成）

    返回 None 当关键点置信度不足。
    """
    side = "right" if is_right_handed else "left"
    other = "left" if is_right_handed else "right"

    sho_idx = KEYPOINT_NAMES[f"{side}_shoulder"]
    elb_idx = KEYPOINT_NAMES[f"{side}_elbow"]
    other_sho_idx = KEYPOINT_NAMES[f"{other}_shoulder"]

    if not (_conf_ok(confidence, sho_idx, KEYPOINT_CONF_THRESHOLD)
            and _conf_ok(confidence, elb_idx, KEYPOINT_CONF_THRESHOLD)
            and _conf_ok(confidence, other_sho_idx, KEYPOINT_CONF_THRESHOLD)):
        return None

    sho = keypoints[sho_idx][:2].astype(np.float64)
    elb = keypoints[elb_idx][:2].astype(np.float64)
    other_sho = keypoints[other_sho_idx][:2].astype(np.float64)

    # 大臂向量（持拍肩 → 持拍肘）
    humerus = elb - sho
    # 躯干横线（持拍肩 → 非持拍肩）—— 注意方向
    shoulder_line = other_sho - sho

    # 取小于 180° 的夹角
    return _vec_angle(humerus, shoulder_line)


def cross_body_finish_distance_2d(
    keypoints: np.ndarray,
    confidence: np.ndarray,
    is_right_handed: bool = True,
) -> Optional[float]:
    """随挥末端：持拍腕越过非持拍肩的归一化距离（用肩宽归一化）。

    > 0 = 已越过 (cross-body 完成)
    < 0 = 未越过 (HSA 未完成)
    """
    side = "right" if is_right_handed else "left"
    other = "left" if is_right_handed else "right"

    wri_idx = KEYPOINT_NAMES[f"{side}_wrist"]
    other_sho_idx = KEYPOINT_NAMES[f"{other}_shoulder"]
    sho_idx = KEYPOINT_NAMES[f"{side}_shoulder"]

    if not all(_conf_ok(confidence, i, KEYPOINT_CONF_THRESHOLD)
               for i in (wri_idx, other_sho_idx, sho_idx)):
        return None

    wri = keypoints[wri_idx][:2].astype(np.float64)
    other_sho = keypoints[other_sho_idx][:2].astype(np.float64)
    sho = keypoints[sho_idx][:2].astype(np.float64)

    shoulder_width = float(np.linalg.norm(other_sho - sho))
    if shoulder_width < 1.0:
        return None

    # 在持拍肩 → 非持拍肩方向上，腕的投影位置
    direction = (other_sho - sho) / shoulder_width
    wri_relative = wri - sho
    projection = float(np.dot(wri_relative, direction))

    # > shoulder_width 表示腕越过非持拍肩
    return (projection - shoulder_width) / shoulder_width


# =============================================================================
# 时间序列分析
# =============================================================================

def compute_hsa_trajectory(
    keypoints_seq: np.ndarray,        # (T, 17, 2 or 3)
    confidence_seq: np.ndarray,       # (T, 17)
    is_right_handed: bool = True,
) -> np.ndarray:
    """计算整段 swing 的 HSA 角度时间序列。

    返回 shape=(T,) 的数组，每帧一个角度（degree），无效帧为 np.nan。
    """
    T = len(keypoints_seq)
    angles = np.full(T, np.nan, dtype=np.float64)
    for t in range(T):
        a = hsa_angle_2d(keypoints_seq[t], confidence_seq[t], is_right_handed)
        if a is not None:
            angles[t] = a
    return angles


def compute_hsa_velocity(
    angles: np.ndarray,
    fps: float,
) -> np.ndarray:
    """HSA 角速度 (deg/s)。负值 = 闭合中（角度在减小）。

    采用平滑 + central difference。无效帧由 nan 表示。
    """
    if len(angles) < 3 or fps <= 0:
        return np.full(len(angles), np.nan)
    # 用 nan-safe 插值再平滑
    valid = np.isfinite(angles)
    if int(np.sum(valid)) < 3:
        return np.full(len(angles), np.nan)
    idx = np.arange(len(angles), dtype=np.float64)
    interp = np.interp(idx, idx[valid], angles[valid])
    smooth = _smooth_series(interp)
    velocity = np.gradient(smooth, 1.0 / fps)
    # 在原本 nan 的位置仍标 nan（避免误用）
    velocity[~valid] = np.nan
    return velocity


# =============================================================================
# 闭合模式分类
# =============================================================================

def classify_closure_pattern(
    metrics: HSAMetrics,
) -> Tuple[str, List[str]]:
    """根据已填的指标分类 HSA 闭合模式。

    返回 (pattern_name, evidence_list)。

    模式列表：
      - "no_closure"            HSA 全程基本不闭合（< 15°）
      - "static"                变化幅度极小（< 10°），靠手臂扫
      - "early_closure"         接触前角度已 < 45°
      - "late_closure"          闭合主要发生在接触后
      - "insufficient_cross_body" 随挥末端右腕未越过左肩
      - "healthy"               其它都健康
      - "uncertain"             数据缺失
    """
    evidence: List[str] = []

    # 数据完整性检查
    has_peak = metrics.hsa_angle_at_unit_turn_peak is not None
    has_contact = metrics.hsa_angle_at_contact is not None
    has_followthrough = metrics.hsa_angle_at_follow_through is not None

    if not (has_contact and (has_peak or has_followthrough)):
        return "uncertain", ["关键阶段角度缺失 (peak/contact/follow-through)"]

    total = metrics.hsa_total_closure_deg
    post = metrics.hsa_post_contact_closure_deg

    # no_closure: 接触瞬间角度仍 > 85°
    if has_contact and metrics.hsa_angle_at_contact > 85.0:
        evidence.append(f"接触瞬间 HSA 角 {metrics.hsa_angle_at_contact:.1f}° > 85°，HSA 未启动")
        return "no_closure", evidence

    # static: 总闭合 < 10°
    if total is not None and abs(total) < 10.0:
        evidence.append(f"总闭合幅度 {total:.1f}° < 10°，纯静态扫球")
        return "static", evidence

    # early_closure: 接触前已 < 45°
    if has_contact and metrics.hsa_angle_at_contact < 45.0:
        evidence.append(f"接触瞬间 HSA 角 {metrics.hsa_angle_at_contact:.1f}° < 45°，肘已贴身")
        return "early_closure", evidence

    # late_closure: 接触后闭合 > 接触前闭合的 1.5 倍
    if total is not None and post is not None and total > 5.0:
        if post > total * 1.5:
            evidence.append(f"接触后闭合 {post:.1f}° > 接触前闭合 {total:.1f}° × 1.5，HSA 主体在击球后")
            return "late_closure", evidence

    # insufficient_cross_body: 跨胸距离 < 0
    if metrics.cross_body_finish is False:
        evidence.append("随挥末端持拍腕未越过非持拍肩，HSA 未完成跨胸")
        return "insufficient_cross_body", evidence

    # 否则健康
    if total is not None:
        evidence.append(f"总闭合 {total:.1f}° 接触角 {metrics.hsa_angle_at_contact:.1f}°，HSA 健康")
    return "healthy", evidence


def compute_health_score(metrics: HSAMetrics) -> Optional[float]:
    """0-100 的 HSA 健康分。基于：
       - 总闭合幅度（25-50°范围内得满分）
       - 接触角度落在健康区间（45-80°）
       - 跨胸完成（True 加分）
       - pattern == healthy 加分
    """
    score = 50.0  # 基线
    components = 0

    # 总闭合幅度
    total = metrics.hsa_total_closure_deg
    if total is not None:
        if total >= HSA_TOTAL_CLOSURE_STRONG:
            score += 20
        elif total >= HSA_TOTAL_CLOSURE_HEALTHY:
            score += 10
        elif total < 10:
            score -= 25
        components += 1

    # 接触角度
    contact = metrics.hsa_angle_at_contact
    if contact is not None:
        if HSA_CONTACT_HEALTHY_MIN <= contact <= HSA_CONTACT_HEALTHY_MAX:
            score += 15
        elif contact > 90:
            score -= 20
        elif contact < 35:
            score -= 15
        components += 1

    # 跨胸完成
    if metrics.cross_body_finish is True:
        score += 10
    elif metrics.cross_body_finish is False:
        score -= 15

    # pattern 健康
    if metrics.hsa_closure_pattern == "healthy":
        score += 5
    elif metrics.hsa_closure_pattern in ("no_closure", "static"):
        score -= 25

    if components == 0:
        return None
    return float(np.clip(score, 0.0, 100.0))


# =============================================================================
# 主入口
# =============================================================================

def detect_hsa(
    keypoints_seq: np.ndarray,
    confidence_seq: np.ndarray,
    fps: float,
    phase_frames: Dict[str, int],
    is_right_handed: bool = True,
) -> HSAMetrics:
    """从 pose 时间序列检测 HSA 指标。

    参数:
        keypoints_seq:  shape (T, 17, 2 or 3)，COCO 17 关键点
        confidence_seq: shape (T, 17)
        fps:            视频帧率
        phase_frames:   dict 至少含 'unit_turn_peak', 'contact', 'follow_through'，
                        值为帧号。也可以含 'forward_swing_start'。
        is_right_handed: 是否右手球员

    返回 HSAMetrics 实例。
    """
    metrics = HSAMetrics()
    T = len(keypoints_seq)
    if T == 0:
        return metrics

    # 1) 计算整段角度时间序列
    angles = compute_hsa_trajectory(keypoints_seq, confidence_seq, is_right_handed)

    # 2) 各 phase 提取角度
    def _at_frame(name: str) -> Optional[float]:
        if name not in phase_frames:
            return None
        f = int(phase_frames[name])
        if f < 0 or f >= T:
            return None
        v = float(angles[f]) if np.isfinite(angles[f]) else None
        return v

    metrics.hsa_angle_at_unit_turn_peak = _at_frame("unit_turn_peak")
    metrics.hsa_angle_at_forward_swing_start = _at_frame("forward_swing_start")
    metrics.hsa_angle_at_contact = _at_frame("contact")
    metrics.hsa_angle_at_follow_through = _at_frame("follow_through")

    # 3) 闭合幅度
    if (metrics.hsa_angle_at_unit_turn_peak is not None
            and metrics.hsa_angle_at_contact is not None):
        metrics.hsa_total_closure_deg = (
            metrics.hsa_angle_at_unit_turn_peak - metrics.hsa_angle_at_contact
        )
    if (metrics.hsa_angle_at_contact is not None
            and metrics.hsa_angle_at_follow_through is not None):
        metrics.hsa_post_contact_closure_deg = (
            metrics.hsa_angle_at_contact - metrics.hsa_angle_at_follow_through
        )

    # 4) 接触前 100ms 平均闭合速度
    contact_f = phase_frames.get("contact")
    if contact_f is not None and fps > 0:
        window_frames = max(1, int(round(HSA_PEAK_VELOCITY_WINDOW_MS / 1000.0 * fps)))
        start = max(0, contact_f - window_frames)
        end = min(T, contact_f + 1)
        if end - start >= 2:
            velocity = compute_hsa_velocity(angles[start:end], fps)
            valid = velocity[np.isfinite(velocity)]
            if valid.size > 0:
                # 闭合中应为负值，取平均；正负保留含义
                metrics.hsa_closure_rate_deg_per_s = float(np.mean(valid))

    # 5) 跨胸完成
    ft_frame = phase_frames.get("follow_through")
    if ft_frame is not None and 0 <= ft_frame < T:
        cb_dist = cross_body_finish_distance_2d(
            keypoints_seq[ft_frame], confidence_seq[ft_frame], is_right_handed
        )
        if cb_dist is not None:
            metrics.cross_body_finish_distance_norm = cb_dist
            metrics.cross_body_finish = cb_dist > 0.0

    # 6) 模式分类 + 健康分
    pattern, evidence = classify_closure_pattern(metrics)
    metrics.hsa_closure_pattern = pattern
    metrics.pattern_evidence = evidence
    metrics.hsa_health_score = compute_health_score(metrics)

    return metrics


# =============================================================================
# CLI 自检
# =============================================================================

if __name__ == "__main__":
    # 合成一组数据自检：模拟一次健康的正手挥拍
    T, fps = 60, 60.0
    kpts = np.zeros((T, 17, 2), dtype=np.float64)
    conf = np.ones((T, 17), dtype=np.float64)

    # 固定骨架：左肩 (100, 100), 右肩 (200, 100)
    for t in range(T):
        kpts[t, KEYPOINT_NAMES["left_shoulder"]] = [100, 100]
        kpts[t, KEYPOINT_NAMES["right_shoulder"]] = [200, 100]
        # 大臂从外展 (95°) 到跨胸 (35°) 线性闭合
        progress = t / (T - 1)
        angle_deg = 95.0 - 60.0 * progress
        # 计算肘位置：从右肩出发，长度 80px，角度按 angle_deg 偏离躯干横线
        angle_rad = np.deg2rad(180.0 - angle_deg)  # 因为基准是 → 左肩方向
        elb_x = 200 + 80 * np.cos(angle_rad)
        elb_y = 100 + 80 * np.sin(angle_rad)
        kpts[t, KEYPOINT_NAMES["right_elbow"]] = [elb_x, elb_y]
        # 腕位置：超出肘 60px 同方向
        wri_x = 200 + 140 * np.cos(angle_rad)
        wri_y = 100 + 140 * np.sin(angle_rad)
        kpts[t, KEYPOINT_NAMES["right_wrist"]] = [wri_x, wri_y]

    phase_frames = {
        "unit_turn_peak": 0,
        "forward_swing_start": 15,
        "contact": 45,
        "follow_through": 59,
    }

    result = detect_hsa(kpts, conf, fps, phase_frames, is_right_handed=True)
    print("=== HSA Self-Test (synthetic healthy swing) ===")
    for k, v in result.to_dict().items():
        print(f"  {k}: {v}")
