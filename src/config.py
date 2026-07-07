# src/config.py
from dataclasses import dataclass
from typing import List, Tuple
import cv2

@dataclass
class DetectionConfig:
    model_name: str = "yolov8n.pt"          # Modèle YOLO
    confidence_threshold: float = 0.5        # Seuil de confiance
    device: str = "cpu"                     # "cuda" si GPU dispo

@dataclass
class AlarmConfig:
    min_distance_px: int = 150              # Seuil d'alarme
    alarm_text: str = "ACHTUNG: ABSTAND ZU GERING!"

@dataclass
class ScreenshotConfig:
    enabled: bool = True
    trigger_classes: List[str] = ("cell phone", "handy")  # Objets déclencheurs
    output_dir: str = "screenshots"
    format: str = "jpg"

@dataclass
class UIConfig:
    window_name: str = "YOLOv8 - Security System"
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    colors: dict = None  # Défini après import cv2
    
# (dans main.py, on fera : from src.config import DetectionConfig, ...)