import math
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
from src.config import AlarmConfig

class DistanceAlarm:
    """Social distancing alarm system."""
    
    def __init__(self, config: AlarmConfig = None):
        self.config = config or AlarmConfig()
        self.alarm_counter: int = 0
        self.total_frames_alarm: int = 0
        
    def check_alarm(
        self, 
        persons: List[Dict[str, Any]], 
        frame: np.ndarray
    ) -> Tuple[bool, np.ndarray, List[Tuple[int, int]]]:
        """
        Checks if persons are too close to each other.
        
        Returns:
            - alarm_triggered (bool)
            - annotated frame
            - list of person centers
        """
        frame_copy = frame.copy() if frame is not None else None
        if frame_copy is None:
            return False, frame, []
        
        person_centers = [p["center"] for p in persons]
        
        if len(person_centers) < 2:
            return False, frame_copy, person_centers
        
        alarm = False
        
        for i in range(len(person_centers)):
            for j in range(i + 1, len(person_centers)):
                p1 = person_centers[i]
                p2 = person_centers[j]
                dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                
                if dist < self.config.min_distance_px:
                    alarm = True
                    cv2.line(frame_copy, p1, p2, self.config.line_color, 2)
                    mid_x = (p1[0] + p2[0]) // 2
                    mid_y = (p1[1] + p2[1]) // 2
                    cv2.putText(
                        frame_copy, 
                        f"{int(dist)}px", 
                        (mid_x, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        self.config.line_color, 
                        1
                    )
        
        if alarm:
            self.alarm_counter += 1
            self.total_frames_alarm += 1
            status_text = self.config.alarm_text
            status_color = self.config.line_color
        else:
            status_text = self.config.ok_text
            status_color = self.config.ok_color
        
        cv2.putText(
            frame_copy, 
            status_text, 
            (10, frame_copy.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            status_color, 
            2
        )
        
        cv2.putText(
            frame_copy,
            f"👤 Personen: {len(person_centers)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        
        return alarm, frame_copy, person_centers
    
    def get_statistics(self) -> Dict[str, int]:
        """Returns alarm statistics."""
        return {
            "total_alarm_frames": self.total_frames_alarm,
            "alarm_trigger_count": self.alarm_counter
        }
    
    def reset_statistics(self):
        """Resets alarm statistics."""
        self.alarm_counter = 0
        self.total_frames_alarm = 0