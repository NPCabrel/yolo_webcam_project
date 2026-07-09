#!/usr/bin/env python3
"""
YOLO Webcam Security System - Main Entry Point
Real-time object detection with social distancing alarm.
"""

import cv2
import sys
import signal
from typing import Optional

from src.config import (
    detection_config, alarm_config, screenshot_config, 
    webcam_config, ui_config
)
from src.webcam_stream import WebcamStream
from src.detector import YOLODetector
from src.alarm_system import DistanceAlarm
from src.screenshot_manager import ScreenshotManager
from src.utils import add_fps_counter, resize_frame


class YOLOSecuritySystem:
    """Main application orchestrating all components."""
    
    def __init__(self):
        self.detector = YOLODetector(detection_config)
        self.alarm = DistanceAlarm(alarm_config)
        self.screenshot_mgr = ScreenshotManager(screenshot_config)
        self.stream = WebcamStream(webcam_config)
        
        self.is_running: bool = False
        self.frame_count: int = 0
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handles clean shutdown (CTRL+C)."""
        print("\n🛑 Shutdown requested...")
        self.stop()
        sys.exit(0)
    
    def run(self) -> int:
        """Runs the main application."""
        print("=" * 60)
        print("🔒 YOLO Webcam Security System")
        print("=" * 60)
        
        if not self.detector.load_model():
            print(" Model loading failed. Stopping.")
            return 1
        
        if not self.stream.open():
            print(" Webcam opening failed. Stopping.")
            return 1
        
        self.is_running = True
        print("\n🎥 Webcam active. Press 'q' or CTRL+C to stop.")
        print("-" * 60)
        
        while self.is_running:
            ret, frame = self.stream.read()
            if not ret:
                print("No more frames. Stopping.")
                break
            
            self.frame_count += 1
            
            detections, annotated = self.detector.detect(frame)
            persons = self.detector.get_persons(detections)
            
            alarm_triggered, annotated, _ = self.alarm.check_alarm(persons, annotated)
            
            if alarm_triggered:
                self.screenshot_mgr.capture_if_triggered(detections, annotated)
            
            annotated = add_fps_counter(annotated, self.stream.get_fps())
            annotated = resize_frame(annotated)
            
            cv2.imshow(ui_config.window_name, annotated)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🛑 Stop requested by user.")
                break
        
        self.stop()
        return 0
    
    def stop(self):
        """Cleans up resources."""
        self.is_running = False
        self.stream.release()
        cv2.destroyAllWindows()
        print("\n Statistics:")
        print(f"   - Frames processed: {self.frame_count}")
        print(f"   - Screenshots: {self.screenshot_mgr.get_count()}")
        print(f"   - Alarms triggered: {self.alarm.get_statistics()['alarm_trigger_count']}")
        print("Program ended.")


def main():
    system = YOLOSecuritySystem()
    sys.exit(system.run())


if __name__ == "__main__":
    main()