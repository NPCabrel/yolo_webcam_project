import os
import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
from src.config import ScreenshotConfig

class ScreenshotManager:
    """Manages automatic screenshots when trigger objects are detected."""
    
    def __init__(self, config: ScreenshotConfig = None):
        self.config = config or ScreenshotConfig()
        self.screenshot_count: int = 0
        self._ensure_output_dir()
        
    def _ensure_output_dir(self):
        """Creates the output directory if it doesn't exist."""
        if self.config.enabled and not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)
            print(f"📁 Directory created: {self.config.output_dir}")
    
    def capture_if_triggered(
        self, 
        detections: List[Dict[str, Any]], 
        frame: np.ndarray
    ) -> Optional[str]:
        """
        Captures an image if a trigger object is detected.
        
        Returns:
            - Saved file path, or None
        """
        if not self.config.enabled:
            return None
        
        if frame is None or frame.size == 0:
            return None
        
        detected_classes = [d["class"].lower() for d in detections]
        trigger_found = False
        trigger_name = None
        
        for trigger in self.config.trigger_classes:
            if trigger.lower() in detected_classes:
                trigger_found = True
                trigger_name = trigger
                break
        
        if not trigger_found:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{trigger_name}_{timestamp}.{self.config.format}"
        filepath = os.path.join(self.config.output_dir, filename)
        
        try:
            cv2.imwrite(filepath, frame, [
                cv2.IMWRITE_JPEG_QUALITY, self.config.quality
            ])
            self.screenshot_count += 1
            print(f"📸 Screenshot saved: {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving screenshot: {e}")
            return None
    
    def get_count(self) -> int:
        """Returns the number of screenshots taken."""
        return self.screenshot_count
    
    def get_latest_files(self, n: int = 5) -> List[str]:
        """Returns the n most recent files."""
        if not os.path.exists(self.config.output_dir):
            return []
        
        files = [os.path.join(self.config.output_dir, f) 
                 for f in os.listdir(self.config.output_dir)
                 if f.endswith(f".{self.config.format}")]
        files.sort(key=os.path.getctime, reverse=True)
        return files[:n]