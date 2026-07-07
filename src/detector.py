# src/detector.py
import numpy as np
import cv2
from ultralytics import YOLO
from src.config import DetectionConfig
from typing import List, Tuple, Any

class YOLODetector:
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.model = YOLO(config.model_name)
        self.class_names = self.model.names

    def detect(self, frame: np.ndarray) -> Tuple[List[dict], np.ndarray]:
        """
        Detect objects in a frame and return:
        - list of detections (bbox, class, confidence)
        - annotated image
        """
        results = self.model(frame, conf=self.config.confidence_threshold)
        annotated = results[0].plot()
        detections = self._parse_results(results[0])
        return detections, annotated

    def _parse_results(self, result) -> List[dict]:
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "class": self.class_names[cls],
                    "class_id": cls,
                    "confidence": conf,
                    "center": ((x1+x2)//2, (y1+y2)//2)
                })
        return detections