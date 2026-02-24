#!/usr/bin/env python3
"""
Visual Report Generator
=======================
Creates annotated images and a visual report card for Big 3 analysis.
Shows exactly where the problems are with clear visual markers.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from big3_analyzer import Big3Analyzer, Big3Report, CheckpointResult

# Colors (BGR format for OpenCV)
GREEN = (0, 200, 0)
YELLOW = (0, 200, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# COCO Keypoint Indices
RIGHT_WRIST = 10
RIGHT_HIP = 12
RIGHT_ANKLE = 16


def get_score_color(score: str) -> Tuple[int, int, int]:
    """Get color based on score."""
    if score == "GOOD":
        return GREEN
    elif score == "OK":
        return YELLOW
    else:
        return RED


def get_score_emoji(score: str) -> str:
    """Get emoji based on score."""
    if score == "GOOD":
        return "GOOD"
    elif score == "OK":
        return "OK"
    else:
        return "BAD"


def draw_contact_point_annotation(img: np.ndarray, keypoints: np.ndarray, 
                                   confidence: np.ndarray, result: CheckpointResult) -> np.ndarray:
    """
    Draw contact point annotation showing wrist vs hip position.
    """
    annotated = img.copy()
    h, w = annotated.shape[:2]
    
    if confidence[RIGHT_WRIST] > 0.5 and confidence[RIGHT_HIP] > 0.5:
        wrist = keypoints[RIGHT_WRIST].astype(int)
        hip = keypoints[RIGHT_HIP].astype(int)
        
        color = get_score_color(result.score)
        
        # Draw vertical lines at wrist and hip
        cv2.line(annotated, (wrist[0], 0), (wrist[0], h), color, 2)
        cv2.line(annotated, (hip[0], 0), (hip[0], h), (255, 150, 150), 2)
        
        # Draw horizontal line connecting them
        cv2.line(annotated, (hip[0], wrist[1]), (wrist[0], wrist[1]), color, 3)
        
        # Draw circles at keypoints
        cv2.circle(annotated, tuple(wrist), 10, color, -1)
        cv2.circle(annotated, tuple(hip), 10, (255, 150, 150), -1)
        
        # Add delta annotation
        delta = result.value
        mid_x = (wrist[0] + hip[0]) // 2
        text = f"{'+' if delta >= 0 else ''}{delta:.0f}px"
        cv2.putText(annotated, text, (mid_x - 30, wrist[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Add labels
        cv2.putText(annotated, "WRIST", (wrist[0] - 30, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(annotated, "HIP", (hip[0] - 15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 150), 2)
    
    # Add result text at top
    cv2.rectangle(annotated, (0, h - 60), (w, h), BLACK, -1)
    cv2.putText(annotated, f"CONTACT POINT: {get_score_emoji(result.score)}", (10, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, get_score_color(result.score), 2)
    cv2.putText(annotated, result.message[2:] if result.message.startswith(('✅', '⚠️', '❌')) else result.message, 
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    
    return annotated


def draw_weight_transfer_annotation(img: np.ndarray, keypoints: np.ndarray,
                                     confidence: np.ndarray, result: CheckpointResult) -> np.ndarray:
    """
    Draw weight transfer annotation showing back foot position.
    """
    annotated = img.copy()
    h, w = annotated.shape[:2]
    
    if confidence[RIGHT_ANKLE] > 0.5 and confidence[RIGHT_HIP] > 0.5:
        ankle = keypoints[RIGHT_ANKLE].astype(int)
        hip = keypoints[RIGHT_HIP].astype(int)
        
        color = get_score_color(result.score)
        
        # Draw line from hip to ankle
        cv2.line(annotated, tuple(hip), tuple(ankle), color, 3)
        
        # Draw circles
        cv2.circle(annotated, tuple(ankle), 12, color, -1)
        cv2.circle(annotated, tuple(hip), 8, color, 2)
        
        # Add arrow showing rotation direction
        if result.score == "GOOD":
            arrow_end = (ankle[0] + 40, ankle[1] - 20)
            cv2.arrowedLine(annotated, tuple(ankle), arrow_end, GREEN, 3, tipLength=0.3)
            cv2.putText(annotated, "ROTATED!", (ankle[0] + 50, ankle[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
        elif result.score == "BAD":
            cv2.putText(annotated, "FLAT!", (ankle[0] + 10, ankle[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
    
    # Add result text at top
    cv2.rectangle(annotated, (0, h - 60), (w, h), BLACK, -1)
    cv2.putText(annotated, f"WEIGHT TRANSFER: {get_score_emoji(result.score)}", (10, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, get_score_color(result.score), 2)
    cv2.putText(annotated, result.message[2:] if result.message.startswith(('✅', '⚠️', '❌')) else result.message,
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    
    return annotated


def draw_contact_zone_annotation(img: np.ndarray, trajectory: List[np.ndarray],
                                  result: CheckpointResult) -> np.ndarray:
    """
    Draw contact zone annotation showing wrist trajectory after contact.
    """
    annotated = img.copy()
    h, w = annotated.shape[:2]
    
    color = get_score_color(result.score)
    
    # Draw trajectory
    if len(trajectory) >= 2:
        for i in range(len(trajectory) - 1):
            pt1 = tuple(trajectory[i].astype(int))
            pt2 = tuple(trajectory[i + 1].astype(int))
            cv2.line(annotated, pt1, pt2, color, 4)
        
        # Draw start and end points
        start = tuple(trajectory[0].astype(int))
        end = tuple(trajectory[-1].astype(int))
        cv2.circle(annotated, start, 10, WHITE, -1)
        cv2.circle(annotated, end, 10, color, -1)
        
        # Add arrow
        cv2.arrowedLine(annotated, start, end, color, 3, tipLength=0.2)
        
        # Add direction labels
        delta_x = end[0] - start[0]
        delta_y = start[1] - end[1]  # Inverted Y
        
        if delta_x > 0:
            cv2.putText(annotated, f"FWD +{delta_x}px", (end[0] + 10, end[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if delta_y > 0:
            cv2.putText(annotated, f"UP +{delta_y}px", (end[0] + 10, end[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add result text at top
    cv2.rectangle(annotated, (0, h - 60), (w, h), BLACK, -1)
    cv2.putText(annotated, f"CONTACT ZONE: {get_score_emoji(result.score)}", (10, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(annotated, result.message[2:] if result.message.startswith(('✅', '⚠️', '❌')) else result.message,
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    
    return annotated


def create_report_card(report: Big3Report, output_path: str, width: int = 800, height: int = 600):
    """
    Create a visual report card image showing all Big 3 results.
    """
    # Create blank canvas
    card = np.ones((height, width, 3), dtype=np.uint8) * 40  # Dark background
    
    # Title
    cv2.rectangle(card, (0, 0), (width, 80), (50, 50, 50), -1)
    cv2.putText(card, "BIG 3 TENNIS ANALYSIS", (width // 2 - 180, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 2)
    
    # Draw each checkpoint
    y_start = 100
    row_height = 150
    
    checkpoints = [
        ("1. CONTACT POINT", report.contact_point),
        ("2. WEIGHT TRANSFER", report.weight_transfer),
        ("3. CONTACT ZONE", report.contact_zone),
    ]
    
    for i, (title, cp) in enumerate(checkpoints):
        y = y_start + i * row_height
        
        if cp:
            color = get_score_color(cp.score)
            
            # Background bar
            cv2.rectangle(card, (20, y), (width - 20, y + 130), (60, 60, 60), -1)
            
            # Score indicator
            cv2.rectangle(card, (20, y), (120, y + 130), color, -1)
            cv2.putText(card, cp.score, (30, y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
            
            # Title and message
            cv2.putText(card, title, (140, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
            
            # Clean message (remove emoji)
            clean_msg = cp.message
            for emoji in ['✅', '⚠️', '❌']:
                clean_msg = clean_msg.replace(emoji, '').strip()
            
            cv2.putText(card, clean_msg[:60], (140, y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(card, f"Frame: {cp.frame_idx}", (140, y + 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    # Priority fix section (bottom)
    y_bottom = height - 80
    cv2.rectangle(card, (0, y_bottom), (width, height), (30, 100, 30), -1)
    cv2.putText(card, f"PRIORITY FIX: {report.priority_fix}", (20, y_bottom + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    cv2.putText(card, f"TIP: {report.tip}", (20, y_bottom + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
    
    # Save
    cv2.imwrite(output_path, card)
    print(f"📊 Report card saved: {output_path}")
    return output_path


def generate_visual_report(data_dir: str, output_dir: str = None):
    """
    Generate complete visual report with annotated frames and report card.
    """
    # Run analysis
    analyzer = Big3Analyzer(data_dir)
    report = analyzer.analyze()
    
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../reports'))
    os.makedirs(output_dir, exist_ok=True)
    
    # Find contact frame
    contact_idx = analyzer._find_contact_frame()
    contact_frame = analyzer.frames_data[contact_idx]
    
    print("\n🎨 Generating visual report...")
    
    # 1. Contact Point annotation
    if report.contact_point:
        img = contact_frame["image"]
        annotated = draw_contact_point_annotation(
            img, contact_frame["keypoints"], contact_frame["confidence"], report.contact_point
        )
        path = os.path.join(output_dir, f"1_contact_point_{report.contact_point.score}.png")
        cv2.imwrite(path, annotated)
        print(f"  ✅ Saved: {path}")
    
    # 2. Weight Transfer annotation
    if report.weight_transfer:
        img = contact_frame["image"]
        annotated = draw_weight_transfer_annotation(
            img, contact_frame["keypoints"], contact_frame["confidence"], report.weight_transfer
        )
        path = os.path.join(output_dir, f"2_weight_transfer_{report.weight_transfer.score}.png")
        cv2.imwrite(path, annotated)
        print(f"  ✅ Saved: {path}")
    
    # 3. Contact Zone annotation
    if report.contact_zone:
        # Get wrist trajectory
        post_frames = analyzer.frames_data[contact_idx:contact_idx + 5]
        trajectory = []
        for f in post_frames:
            if f["confidence"][RIGHT_WRIST] > 0.5:
                trajectory.append(f["keypoints"][RIGHT_WRIST])
        
        img = contact_frame["image"]
        annotated = draw_contact_zone_annotation(img, trajectory, report.contact_zone)
        path = os.path.join(output_dir, f"3_contact_zone_{report.contact_zone.score}.png")
        cv2.imwrite(path, annotated)
        print(f"  ✅ Saved: {path}")
    
    # 4. Report card
    card_path = os.path.join(output_dir, "big3_report_card.png")
    create_report_card(report, card_path)
    
    # 5. Save text report
    analyzer.print_report(report)
    analyzer.save_report(report, output_dir)
    
    print(f"\n✨ Visual report complete! Check {output_dir}/")
    return report


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate visual report for Big 3 analysis")
    parser.add_argument("data_dir", type=str, help="Directory containing extracted frames")
    parser.add_argument("--output", type=str, default=None, help="Output directory for reports")
    args = parser.parse_args()
    
    generate_visual_report(args.data_dir, args.output)


if __name__ == "__main__":
    main()
