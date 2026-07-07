# src/alarm_system.py
import numpy as np
import math
import cv2
from src.config import AlarmConfig
from typing import List, Tuple, Optional

class DistanceAlarm:
    def __init__(self, config: AlarmConfig):
        self.config = config

    def check_alarm(self, person_centers: List[Tuple[int, int]], frame: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Vérifie si deux personnes sont trop proches.
        Retourne (alarm_triggered, frame_annotated)
        """
        alarm = False
        if len(person_centers) < 2:
            return False, frame

        for i in range(len(person_centers)):
            for j in range(i+1, len(person_centers)):
                p1, p2 = person_centers[i], person_centers[j]
                dist = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
                if dist < self.config.min_distance_px:
                    alarm = True
                    cv2.line(frame, p1, p2, (0,0,255), 2)
                    # Afficher la distance
                    mid_x, mid_y = (p1[0]+p2[0])//2, (p1[1]+p2[1])//2
                    cv2.putText(frame, f"{int(dist)}px", (mid_x, mid_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        return alarm, frame