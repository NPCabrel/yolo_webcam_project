from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import cv2

@dataclass
class DetectionConfig:
    """Configuration for YOLO object detection."""
    model_name: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    device: str = "cpu"
    image_size: int = 640

@dataclass
class AlarmConfig:
    """Configuration for social distancing alarm."""
    min_distance_px: int = 150
    alarm_text: str = "ACHTUNG: ABSTAND ZU GERING!"
    ok_text: str = "Abstand ok"
    line_color: Tuple[int, int, int] = (0, 0, 255)  # Red (BGR)
    ok_color: Tuple[int, int, int] = (0, 255, 0)    # Green (BGR)

@dataclass
class ScreenshotConfig:
    """Configuration for screenshot capture."""
    enabled: bool = True
    trigger_classes: Tuple[str, ...] = ("cell phone", "handy", "laptop", "airpots", "Uhr", "Ring", "Kette")
    output_dir: str = "screenshots"
    format: str = "jpg"
    quality: int = 95

@dataclass
class WebcamConfig:
    """Configuration for webcam stream."""
    camera_id: int = 0
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 30

@dataclass
class UIConfig:
    """Configuration for user interface."""
    window_name: str = "YOLOv8 - Security System"
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.7
    font_thickness: int = 2

# Global configuration instances
detection_config = DetectionConfig()
alarm_config = AlarmConfig()
screenshot_config = ScreenshotConfig()
webcam_config = WebcamConfig()
ui_config = UIConfig()