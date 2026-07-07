# src/main.py
import cv2
from src.config import DetectionConfig, AlarmConfig, ScreenshotConfig, UIConfig
from src.detector import YOLODetector
from src.alarm_system import DistanceAlarm
from src.screenshot_manager import ScreenshotManager
from src.webcam_stream import WebcamStream

def main():
    # 1. Charger les configurations
    det_config = DetectionConfig()
    alarm_config = AlarmConfig()
    screen_config = ScreenshotConfig()
    
    # 2. Initialiser les composants
    detector = YOLODetector(det_config)
    alarm = DistanceAlarm(alarm_config)
    screenshot = ScreenshotManager(screen_config)
    stream = WebcamStream(0)  # ID de la webcam
    
    # 3. Boucle principale
    for frame in stream:
        detections, annotated = detector.detect(frame)
        persons = [d["center"] for d in detections if d["class"] == "person"]
        
        # Alarme distance
        triggered, annotated = alarm.check_alarm(persons, annotated)
        print("ALARME !" if triggered else "Tout va bien")
        
        # Screenshot conditionnel
        screenshot.capture_if_triggered(detections, frame)
        
        cv2.imshow("Security System", annotated)
        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == "__main__":
    main()