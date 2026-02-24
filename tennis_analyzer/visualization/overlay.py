"""Overlay rendering for metrics and info."""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont


class OverlayRenderer:
    """Render text overlays and metrics on frames with Chinese support."""

    def __init__(
        self,
        font_size: int = 20,
        text_color: Tuple[int, int, int] = (255, 255, 255),
        bg_color: Tuple[int, int, int] = (0, 0, 0),
        bg_alpha: float = 0.6,
    ):
        """
        Initialize overlay renderer.

        Args:
            font_size: Font size for text
            text_color: Text color (BGR for OpenCV, will convert to RGB for PIL)
            bg_color: Background color (BGR)
            bg_alpha: Background transparency
        """
        self.font_size = font_size
        self.text_color = text_color
        self.bg_color = bg_color
        self.bg_alpha = bg_alpha

        # Try to load a Chinese-compatible font
        self.pil_font = self._load_chinese_font(font_size)

        # Fallback OpenCV font settings
        self.cv_font = cv2.FONT_HERSHEY_SIMPLEX
        self.cv_font_scale = 0.6
        self.cv_font_thickness = 2

    def _load_chinese_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Load a font that supports Chinese characters."""
        # Common Chinese font paths on macOS
        font_paths = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        ]

        for path in font_paths:
            try:
                return ImageFont.truetype(path, size)
            except (IOError, OSError):
                continue

        # Fallback to default
        try:
            return ImageFont.truetype("Arial", size)
        except:
            return ImageFont.load_default()

    def _has_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                return True
        return False

    def _draw_text_pil(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int] = None,
    ) -> np.ndarray:
        """Draw text using PIL (supports Chinese)."""
        if color is None:
            color = self.text_color

        # Convert BGR to RGB for PIL
        rgb_color = (color[2], color[1], color[0])

        # Convert frame to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        # Draw text
        draw.text(position, text, font=self.pil_font, fill=rgb_color)

        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def _get_text_size_pil(self, text: str) -> Tuple[int, int]:
        """Get text size using PIL."""
        # Create a temporary image to measure text
        dummy_img = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(dummy_img)
        bbox = draw.textbbox((0, 0), text, font=self.pil_font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    def draw_text(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        with_background: bool = True,
        color: Tuple[int, int, int] = None,
    ) -> np.ndarray:
        """
        Draw text on frame (supports Chinese).

        Args:
            frame: BGR image
            text: Text to draw
            position: (x, y) position for text
            with_background: Whether to draw background box
            color: Text color (BGR), defaults to self.text_color

        Returns:
            Frame with text drawn
        """
        frame = frame.copy()
        if color is None:
            color = self.text_color

        # Get text size
        text_width, text_height = self._get_text_size_pil(text)

        if with_background:
            # Draw background rectangle
            padding = 8
            x, y = position
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (x - padding, y - padding),
                (x + text_width + padding, y + text_height + padding),
                self.bg_color,
                -1
            )
            frame = cv2.addWeighted(overlay, self.bg_alpha, frame, 1 - self.bg_alpha, 0)

        # Draw text using PIL
        frame = self._draw_text_pil(frame, text, position, color)

        return frame

    def draw_info_panel(
        self,
        frame: np.ndarray,
        info: Dict[str, str],
        position: str = "top-left",
        padding: int = 10,
    ) -> np.ndarray:
        """
        Draw an info panel with multiple lines.

        Args:
            frame: BGR image
            info: Dictionary of label: value pairs
            position: Panel position ('top-left', 'top-right', 'bottom-left', 'bottom-right')
            padding: Padding from frame edge

        Returns:
            Frame with info panel
        """
        frame = frame.copy()
        h, w = frame.shape[:2]

        # Calculate text dimensions
        lines = [f"{k}: {v}" for k, v in info.items()]
        line_sizes = [self._get_text_size_pil(line) for line in lines]
        line_heights = [size[1] for size in line_sizes]
        line_widths = [size[0] for size in line_sizes]

        line_spacing = 8
        total_height = sum(line_heights) + (len(lines) - 1) * line_spacing
        max_width = max(line_widths) if line_widths else 0

        # Calculate panel position
        panel_padding = 8
        if position == "top-left":
            x = padding
            y = padding
        elif position == "top-right":
            x = w - max_width - padding - panel_padding * 2
            y = padding
        elif position == "bottom-left":
            x = padding
            y = h - total_height - padding - panel_padding * 2
        else:  # bottom-right
            x = w - max_width - padding - panel_padding * 2
            y = h - total_height - padding - panel_padding * 2

        # Draw background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x - panel_padding, y - panel_padding),
            (x + max_width + panel_padding, y + total_height + panel_padding),
            self.bg_color,
            -1
        )
        frame = cv2.addWeighted(overlay, self.bg_alpha, frame, 1 - self.bg_alpha, 0)

        # Draw text lines
        current_y = y
        for line, lh in zip(lines, line_heights):
            frame = self._draw_text_pil(frame, line, (x, current_y))
            current_y += lh + line_spacing

        return frame

    def draw_metrics(
        self,
        frame: np.ndarray,
        metrics: Dict[str, float],
        position: str = "top-right",
    ) -> np.ndarray:
        """
        Draw metrics panel with formatted values.

        Args:
            frame: BGR image
            metrics: Dictionary of metric_name: value pairs
            position: Panel position

        Returns:
            Frame with metrics panel
        """
        formatted = {}
        for name, value in metrics.items():
            if isinstance(value, float):
                formatted[name] = f"{value:.1f}°"
            else:
                formatted[name] = str(value)

        return self.draw_info_panel(frame, formatted, position)

    def draw_feedback(
        self,
        frame: np.ndarray,
        text: str,
        color: Tuple[int, int, int],
        position: str = "bottom-center",
    ) -> np.ndarray:
        """
        Draw feedback message at specified position.

        Args:
            frame: BGR image
            text: Feedback text
            color: Text color (BGR)
            position: Position ('top-center', 'bottom-center')

        Returns:
            Frame with feedback
        """
        h, w = frame.shape[:2]
        text_width, text_height = self._get_text_size_pil(text)

        # Calculate position
        x = (w - text_width) // 2
        if position == "top-center":
            y = 20
        else:  # bottom-center
            y = h - text_height - 40

        return self.draw_text(frame, text, (x, y), with_background=True, color=color)

    def draw_feedback_list(
        self,
        frame: np.ndarray,
        feedbacks: list,
        position: str = "bottom-left",
    ) -> np.ndarray:
        """
        Draw multiple feedback messages in a column.

        Args:
            frame: BGR image
            feedbacks: List of (message, status) tuples where status is 'good' or 'warning'
            position: Position ('bottom-left', 'bottom-right')

        Returns:
            Frame with feedbacks
        """
        if not feedbacks:
            return frame

        h, w = frame.shape[:2]
        padding = 10
        line_spacing = 8

        # Colors
        color_good = (0, 255, 128)    # Green
        color_warning = (0, 0, 255)   # Red

        # Calculate total height
        line_heights = []
        line_widths = []
        for message, status in feedbacks:
            tw, th = self._get_text_size_pil(message)
            line_heights.append(th)
            line_widths.append(tw)

        total_height = sum(line_heights) + (len(feedbacks) - 1) * line_spacing
        max_width = max(line_widths) if line_widths else 0

        # Calculate starting position
        panel_padding = 8
        if position == "bottom-left":
            x = padding
            start_y = h - total_height - padding - panel_padding
        else:  # bottom-right
            x = w - max_width - padding - panel_padding * 2
            start_y = h - total_height - padding - panel_padding

        # Draw background
        frame = frame.copy()
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x - panel_padding, start_y - panel_padding),
            (x + max_width + panel_padding, start_y + total_height + panel_padding),
            self.bg_color,
            -1
        )
        frame = cv2.addWeighted(overlay, self.bg_alpha, frame, 1 - self.bg_alpha, 0)

        # Draw each feedback line
        current_y = start_y
        for (message, status), lh in zip(feedbacks, line_heights):
            color = color_good if status == "good" else color_warning
            frame = self._draw_text_pil(frame, message, (x, current_y), color)
            current_y += lh + line_spacing

        return frame
