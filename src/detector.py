import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from ultralytics import YOLO
from src.config import DetectionConfig

class YOLODetector:
    """YOLOv8 object detector with result parsing."""
    
    def __init__(self, config: DetectionConfig = None):
        self.config = config or DetectionConfig()
        self.model: Optional[YOLO] = None
        self.class_names: Dict[int, str] = {}
        self.is_loaded: bool = False
        
    def load_model(self) -> bool:
        """Loads the YOLO model."""
        try:
            print(f" Loading model {self.config.model_name}...")
            self.model = YOLO(self.config.model_name)
            self.class_names = self.model.names
            self.is_loaded = True
            print(f"Model loaded ({len(self.class_names)} classes)")
            return True
        except Exception as e:
            print(f" Error loading model: {e}")
            return False
    
    def detect(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Detects objects in a frame.
        
        Returns:
            - List of detections (bbox, class, confidence, center)
            - Annotated frame
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if frame is None or frame.size == 0:
            return [], frame
        
        results = self.model(
            frame, 
            conf=self.config.confidence_threshold,
            imgsz=self.config.image_size,
            device=self.config.device
        )
        
        annotated_frame = results[0].plot() if results else frame.copy()
        detections = self._parse_results(results[0]) if results else []
        
        return detections, annotated_frame
    
    def _parse_results(self, result) -> List[Dict[str, Any]]:
        """Parses YOLO results into structured dictionaries."""
        detections = []
        
        if result.boxes is None:
            return detections
        
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = self.class_names.get(cls_id, "unknown")
            
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "class_id": cls_id,
                "class": class_name,
                "confidence": confidence,
                "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                "width": x2 - x1,
                "height": y2 - y1
            })
        
        return detections
    
    def get_objects_by_class(self, detections: List[Dict], class_name: str) -> List[Dict]:
        """Filters detections by class name."""
        return [d for d in detections if d["class"] == class_name]
    
    def get_persons(self, detections: List[Dict]) -> List[Dict]:
        """Filters detections to get only persons."""
        return self.get_objects_by_class(detections, "person")