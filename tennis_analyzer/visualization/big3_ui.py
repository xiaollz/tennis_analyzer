#!/usr/bin/env python3
"""
Big 3 UI Renderer
=================
Modern, proportionate UI overlay for Big 3 real-time analysis.
Designed to be clean, readable, and visually appealing.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
from enum import Enum

from ..analysis.big3_monitors import Status, ContactPointStatus, WeightTransferStatus, ContactZoneStatus


# Colors (BGR for OpenCV)
class Colors:
    """UI Color palette."""
    GOOD = (94, 197, 34)       # Green #22C55E
    OK = (8, 179, 234)         # Yellow #EAB308
    BAD = (68, 68, 239)        # Red #EF4444
    UNKNOWN = (128, 128, 128)  # Gray
    
    BG_DARK = (55, 41, 31)     # Dark background #1F2937
    BG_PANEL = (70, 55, 45)    # Panel background
    TEXT_WHITE = (255, 255, 255)
    TEXT_GRAY = (180, 180, 180)
    
    @staticmethod
    def from_status(status: Status) -> Tuple[int, int, int]:
        """Get color from status enum."""
        if status == Status.GOOD:
            return Colors.GOOD
        elif status == Status.OK:
            return Colors.OK
        elif status == Status.BAD:
            return Colors.BAD
        return Colors.UNKNOWN


class Big3UIRenderer:
    """
    Renders Big 3 analysis UI overlay on video frames.
    Shows results ONLY after impact detection.
    """
    
    def __init__(self, 
                 font_size: int = 16,
                 panel_alpha: float = 0.85):
        self.font_size = font_size
        self.panel_alpha = panel_alpha
        
        # Results to show (frozen after impact)
        self.frozen_results = None
        self.show_results = False
        
        # Try to load a font that supports Chinese
        self.font = self._load_font(font_size)
        self.font_small = self._load_font(font_size - 4)
        self.font_large = self._load_font(font_size + 4)
        
    def _load_font(self, size: int):
        """Load font with Chinese support."""
        font_paths = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
        for path in font_paths:
            try:
                return ImageFont.truetype(path, size)
            except:
                continue
        return ImageFont.load_default()
    
    def _draw_text_pil(self, frame: np.ndarray, text: str, 
                       position: Tuple[int, int], 
                       color: Tuple[int, int, int] = None,
                       font = None) -> np.ndarray:
        """Draw text using PIL for Chinese support."""
        if color is None:
            color = Colors.TEXT_WHITE
        if font is None:
            font = self.font
            
        # Convert BGR to RGB for PIL
        rgb_color = (color[2], color[1], color[0])
        
        # Convert frame to PIL
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        draw.text(position, text, font=font, fill=rgb_color)
        
        # Convert back to OpenCV
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _get_text_size_pil(self, text: str, font=None) -> Tuple[int, int]:
        """Measure text size using PIL (accurate for Chinese fonts)."""
        if font is None:
            font = self.font
        dummy_img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy_img)
        bbox = draw.textbbox((0, 0), text, font=font)
        return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])

    def _wrap_text(self, text: str, max_width: int, font=None) -> list[str]:
        """Greedy wrap by characters (works for Chinese, avoids overflow)."""
        if font is None:
            font = self.font
        if not text:
            return [""]
        max_width = max(1, int(max_width))

        lines: list[str] = []
        current = ""
        for ch in text:
            candidate = current + ch
            w, _ = self._get_text_size_pil(candidate, font=font)
            if w <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = ch
        if current:
            lines.append(current)
        return lines
    
    def _draw_rounded_rect(self, frame: np.ndarray, 
                           pt1: Tuple[int, int], pt2: Tuple[int, int],
                           color: Tuple[int, int, int], 
                           radius: int = 8,
                           alpha: float = 0.8) -> np.ndarray:
        """Draw a rounded rectangle with transparency."""
        overlay = frame.copy()
        
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Draw filled rounded rectangle
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        
        # Draw corners
        cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)
        
        # Blend
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    def draw_status_panel(self, frame: np.ndarray, 
                          statuses: Dict) -> np.ndarray:
        """
        Draw the Big 3 status panel on the left side.
        
        Args:
            frame: Video frame
            statuses: Dict with contact_point, weight_transfer, contact_zone
        """
        h, w = frame.shape[:2]

        margin = 15
        pad_x = 12
        pad_y = 10
        title_gap = 10
        line_gap = 8

        x1, y1 = margin, margin
        max_panel_w = int(min(w - 2 * margin, w * 0.60))
        max_text_w = max(80, max_panel_w - 2 * pad_x)

        title = "BIG 3 检测"

        # Build display lines (wrap to fit the panel).
        items = [
            ("触球点", statuses.get("contact_point")),
            ("重心", statuses.get("weight_transfer")),
            ("随挥", statuses.get("contact_zone")),
        ]

        display_lines: list[tuple[str, Tuple[int, int, int], ImageFont.ImageFont]] = []
        for label, status in items:
            if not status or status.status == Status.UNKNOWN:
                continue
            color = Colors.from_status(status.status)
            msg = status.message if hasattr(status, "message") else "未知"
            msg = msg.replace("✓", "").replace("✗", "").strip()

            line = f"{label}: {msg}"
            for sub in self._wrap_text(line, max_text_w, font=self.font):
                display_lines.append((sub, color, self.font))

        # Measure required panel size.
        title_w, title_h = self._get_text_size_pil(title, font=self.font_large)
        text_ws: list[int] = []
        text_hs: list[int] = []
        for line, _, fnt in display_lines:
            tw, th = self._get_text_size_pil(line, font=fnt)
            text_ws.append(tw)
            text_hs.append(th)

        content_w = max([title_w] + text_ws) if (text_ws or title_w) else title_w
        panel_w = int(min(max_panel_w, content_w + 2 * pad_x))

        lines_h = sum(text_hs) + max(0, len(text_hs) - 1) * line_gap
        panel_h = int(pad_y + title_h + title_gap + lines_h + pad_y)
        panel_h = min(panel_h, h - 2 * margin)

        x2, y2 = x1 + panel_w, y1 + panel_h

        # Draw background (shadow panel).
        frame = self._draw_rounded_rect(
            frame, (x1, y1), (x2, y2), Colors.BG_DARK, radius=10, alpha=self.panel_alpha
        )

        # Title.
        y_cursor = y1 + pad_y
        frame = self._draw_text_pil(
            frame, title, (x1 + pad_x, y_cursor), Colors.TEXT_WHITE, self.font_large
        )
        y_cursor += title_h + title_gap

        # Lines.
        for (line, color, fnt), th in zip(display_lines, text_hs):
            if y_cursor + th > y2 - pad_y:
                break
            frame = self._draw_text_pil(frame, line, (x1 + pad_x, y_cursor), color, fnt)
            y_cursor += th + line_gap

        return frame
    
    def _get_status_icon(self, status: Status) -> str:
        """Get emoji/icon for status."""
        if status == Status.GOOD:
            return "✓"
        elif status == Status.OK:
            return "△"
        elif status == Status.BAD:
            return "✗"
        return "?"
    
    def draw_contact_gauge(self, frame: np.ndarray, 
                           delta_px: float,
                           status: Status) -> np.ndarray:
        """
        Draw contact point gauge at the bottom.
        Shows position from -100px (late) to +100px (early).
        """
        h, w = frame.shape[:2]
        
        # Gauge dimensions
        gauge_h = 45
        padding = 20
        y1 = h - gauge_h - padding
        y2 = h - padding
        x1 = padding
        x2 = w - padding
        
        # Draw background
        frame = self._draw_rounded_rect(frame, (x1, y1), (x2, y2), 
                                         Colors.BG_DARK, radius=8, 
                                         alpha=self.panel_alpha)
        
        # Gauge bar dimensions
        bar_x1 = x1 + 80
        bar_x2 = x2 - 100
        bar_y = y1 + gauge_h // 2
        bar_h = 12
        bar_w = bar_x2 - bar_x1
        
        # Draw gradient bar (red -> yellow -> green)
        for i in range(bar_w):
            ratio = i / bar_w
            if ratio < 0.3:  # Red zone (late)
                color = Colors.BAD
            elif ratio < 0.5:  # Yellow zone (OK)
                color = Colors.OK
            else:  # Green zone (good)
                color = Colors.GOOD
            cv2.line(frame, (bar_x1 + i, bar_y - bar_h//2), 
                     (bar_x1 + i, bar_y + bar_h//2), color, 1)
        
        # Draw center line
        center_x = (bar_x1 + bar_x2) // 2
        cv2.line(frame, (center_x, bar_y - bar_h), (center_x, bar_y + bar_h), 
                 Colors.TEXT_WHITE, 2)
        
        # Draw marker (position based on delta)
        # Map delta to bar position: -100 to +100 -> bar_x1 to bar_x2
        clamped_delta = max(-100, min(100, delta_px))
        marker_ratio = (clamped_delta + 100) / 200
        marker_x = int(bar_x1 + marker_ratio * bar_w)
        
        marker_color = Colors.from_status(status)
        cv2.circle(frame, (marker_x, bar_y), 10, marker_color, -1)
        cv2.circle(frame, (marker_x, bar_y), 10, Colors.TEXT_WHITE, 2)
        
        # Labels
        frame = self._draw_text_pil(frame, "触球点", (x1 + 12, y1 + 10), 
                                    Colors.TEXT_GRAY, self.font_small)
        
        # Value label
        sign = "+" if delta_px >= 0 else ""
        value_text = f"{sign}{delta_px:.0f}px"
        frame = self._draw_text_pil(frame, value_text, (x2 - 80, y1 + 10), 
                                    marker_color, self.font)
        
        return frame
    
    def draw_phase_label(self, frame: np.ndarray, 
                         phase: str, 
                         frame_num: int,
                         total_frames: int = 0) -> np.ndarray:
        """Draw current phase and frame number at top right."""
        h, w = frame.shape[:2]
        
        # Panel dimensions
        panel_w = 180
        panel_h = 55
        padding = 15
        x1 = w - panel_w - padding
        y1 = padding
        x2 = w - padding
        y2 = y1 + panel_h
        
        # Draw background
        frame = self._draw_rounded_rect(frame, (x1, y1), (x2, y2), 
                                         Colors.BG_DARK, radius=8, 
                                         alpha=self.panel_alpha)
        
        # Phase label
        phase_display = self._format_phase(phase)
        frame = self._draw_text_pil(frame, f"阶段: {phase_display}", 
                                    (x1 + 10, y1 + 5), 
                                    Colors.TEXT_WHITE, self.font_small)
        
        # Frame number
        if total_frames > 0:
            frame_text = f"帧: {frame_num}/{total_frames}"
        else:
            frame_text = f"帧: {frame_num}"
        frame = self._draw_text_pil(frame, frame_text, 
                                    (x1 + 10, y1 + 28), 
                                    Colors.TEXT_GRAY, self.font_small)
        
        return frame
    
    def _format_phase(self, phase: str) -> str:
        """Format phase name for display."""
        phase_map = {
            "ready": "准备",
            "unit_turn": "转体",
            "backswing": "引拍",
            "forward_swing": "挥拍",
            "contact": "击球",
            "follow_through": "随挥",
            "recovery": "还原",
            "unknown": "未知",
        }
        return phase_map.get(phase.lower(), phase)
    
    def draw_impact_flash(self, frame: np.ndarray) -> np.ndarray:
        """Draw a brief flash effect when impact is detected."""
        h, w = frame.shape[:2]
        
        # Create yellow border flash
        border_size = 8
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, border_size), Colors.OK, -1)
        cv2.rectangle(overlay, (0, h - border_size), (w, h), Colors.OK, -1)
        cv2.rectangle(overlay, (0, 0), (border_size, h), Colors.OK, -1)
        cv2.rectangle(overlay, (w - border_size, 0), (w, h), Colors.OK, -1)
        
        return cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    
    def draw_feedback_message(self, frame: np.ndarray, 
                              message: str,
                              status: Status) -> np.ndarray:
        """Draw a feedback message at the bottom center."""
        h, w = frame.shape[:2]
        
        # Calculate text size (approximate)
        text_w = len(message) * 12
        text_h = 30
        
        x1 = (w - text_w) // 2 - 10
        y1 = h - 110
        x2 = x1 + text_w + 20
        y2 = y1 + text_h + 10
        
        color = Colors.from_status(status)
        
        # Draw background
        frame = self._draw_rounded_rect(frame, (x1, y1), (x2, y2), 
                                         color, radius=5, alpha=0.9)
        
        # Draw text
        frame = self._draw_text_pil(frame, message, 
                                    (x1 + 10, y1 + 5), 
                                    Colors.TEXT_WHITE, self.font)
        
        return frame
    
    def freeze_results(self, statuses: Dict):
        """Freeze results after impact detection."""
        self.frozen_results = statuses
        self.show_results = True
    
    def reset(self):
        """Reset for new swing."""
        self.frozen_results = None
        self.show_results = False
    
    def render_overlay(self, frame: np.ndarray, 
                       statuses: Dict,
                       phase: str = "unknown",
                       frame_num: int = 0,
                       total_frames: int = 0,
                       show_impact_flash: bool = False,
                       is_actual_impact: bool = False) -> np.ndarray:
        """
        Render Big 3 overlay - only shows results AFTER impact.
        Results are frozen on impact and stay until next swing.
        
        Args:
            frame: Video frame
            statuses: Dict from Big3MonitorSet.update()
            phase: Current stroke phase
            frame_num: Current frame number
            total_frames: Total frames in video
            show_impact_flash: Whether to show impact flash effect
            is_actual_impact: True only on the exact frame impact was detected
        """
        # Impact flash (visual effect only)
        if show_impact_flash:
            frame = self.draw_impact_flash(frame)
        
        # Phase label (top-right) - always show
        frame = self.draw_phase_label(frame, phase, frame_num, total_frames)
        
        # Status panel (top-left) - ONLY show frozen results after impact
        if self.show_results and self.frozen_results:
            frame = self.draw_status_panel(frame, self.frozen_results)
        
        return frame
