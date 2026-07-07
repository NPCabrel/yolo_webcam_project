import cv2
import time
from typing import Optional, Generator, Tuple
import numpy as np
from src.config import WebcamConfig

class WebcamStream:
    """Handler for webcam video stream with error management."""
    
    def __init__(self, config: WebcamConfig = None):
        self.config = config or WebcamConfig()
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened: bool = False
        self.frame_count: int = 0
        self.fps: float = 0.0
        self._last_time: float = time.time()
        
    def open(self) -> bool:
        """Opens the webcam and returns True if successful."""
        try:
            self.cap = cv2.VideoCapture(self.config.camera_id)
            if not self.cap.isOpened():
                print(f"Error: Cannot open webcam {self.config.camera_id}")
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            self.is_opened = True
            print(f" Webcam {self.config.camera_id} opened successfully")
            print(f"   Resolution: {self.get_width()}x{self.get_height()}")
            return True
            
        except Exception as e:
            print(f" Error opening webcam: {e}")
            return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads a frame. Returns (success, frame)."""
        if not self.is_opened or self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        if not ret:
            print(" Warning: Frame not read (end of stream?)")
            return False, None
        
        self.frame_count += 1
        
        if self.frame_count % 30 == 0:
            current_time = time.time()
            elapsed = current_time - self._last_time
            self.fps = 30 / elapsed if elapsed > 0 else 0
            self._last_time = current_time
        
        return True, frame
    
    def __iter__(self) -> Generator[np.ndarray, None, None]:
        """Allows iteration with 'for frame in stream'."""
        if not self.is_opened and not self.open():
                return
        
        while True:
            ret, frame = self.read()
            if not ret:
                break
            yield frame
    
    def release(self):
        """Releases the webcam."""
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            print(" Webcam released")
    
    def get_width(self) -> int:
        if self.cap is not None:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        return self.config.frame_width
    
    def get_height(self) -> int:
        if self.cap is not None:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return self.config.frame_height
    
    def get_fps(self) -> float:
        return self.fps
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()