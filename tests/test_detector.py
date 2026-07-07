import pytest
import numpy as np
from src.detector import YOLODetector
from src.config import DetectionConfig

def test_detector_initialization():
    detector = YOLODetector()
    assert detector.is_loaded == False
    assert detector.model is None

def test_detector_load_model():
    detector = YOLODetector(DetectionConfig(model_name="yolov8n.pt"))
    assert detector.load_model() == True
    assert detector.is_loaded == True
    assert len(detector.class_names) > 0

def test_detector_parse_empty():
    detector = YOLODetector()
    class MockResult:
        boxes = None
    result = MockResult()
    detections = detector._parse_results(result)
    assert detections == []