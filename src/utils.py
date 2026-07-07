import math
import cv2
import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime

def calculate_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Calculates Euclidean distance between two points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def draw_bounding_box(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """Draws a bounding box with label on a frame."""
    x1, y1, x2, y2 = bbox
    frame_copy = frame.copy()
    
    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)
    
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.rectangle(frame_copy, (x1, y1 - label_size[1] - 5), 
                  (x1 + label_size[0], y1), color, -1)
    cv2.putText(frame_copy, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame_copy

def resize_frame(frame: np.ndarray, max_width: int = 1280, max_height: int = 720) -> np.ndarray:
    """Resizes a frame if it exceeds maximum dimensions."""
    h, w = frame.shape[:2]
    if w > max_width or h > max_height:
        scale = min(max_width / w, max_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h))
    return frame

def add_fps_counter(frame: np.ndarray, fps: float) -> np.ndarray:
    """Adds an FPS counter on the frame."""
    frame_copy = frame.copy()
    cv2.putText(
        frame_copy,
        f"FPS: {fps:.1f}",
        (frame_copy.shape[1] - 120, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )
    return frame_copy

def get_timestamp() -> str:
    """Returns a formatted timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]