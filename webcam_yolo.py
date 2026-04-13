import cv2
from ultralytics import YOLO
from datetime import datetime
import os
import math

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Fehler: Kamera konnte nicht geöffnet werden.")
    exit()

screenshot_folder = "screenshots"
if not os.path.exists(screenshot_folder):
    os.makedirs(screenshot_folder)

print("Webcam läuft. Drücke 'q' zum Beenden.")
print("Features: Personen zählen | Abstandsalarm | Screenshots bei 'handy' oder 'cell phone'")

def calculate_distance(point1, point2):
    return math.hypot(point1[0] - point2[0], point1[1] - point2[1])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    boxes = results[0].boxes
    person_centers = []
    detected_objects = []

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls[0])
            class_name = model.names[cls]
            detected_objects.append(class_name)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            if class_name == "person":
                person_centers.append((center_x, center_y))

    # Personen zählen
    person_count = len(person_centers)
    cv2.putText(annotated_frame, f"Personen: {person_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Abstandsalarm
    min_distance = 150
    alarm_active = False
    if person_count >= 2:
        for i in range(len(person_centers)):
            for j in range(i+1, len(person_centers)):
                dist = calculate_distance(person_centers[i], person_centers[j])
                if dist < min_distance:
                    alarm_active = True
                    cv2.line(annotated_frame, person_centers[i], person_centers[j], (0, 0, 255), 2)
                    cv2.putText(annotated_frame, f"{int(dist)}px",
                                ((person_centers[i][0] + person_centers[j][0]) // 2,
                                 (person_centers[i][1] + person_centers[j][1]) // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    if alarm_active:
        cv2.putText(annotated_frame, "ACHTUNG: ABSTAND ZU GERING!", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(annotated_frame, "Abstand ok", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Screenshot bei bestimmten Objekten
    trigger_objects = ["cell phone", "handy"]  # "handy" ist nicht in COCO, aber "cell phone" schon
    screenshot_taken = False
    for obj in trigger_objects:
        if obj in detected_objects:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{screenshot_folder}/{obj}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Screenshot gespeichert: {filename}")
            screenshot_taken = True
            break
    if screenshot_taken:
        cv2.putText(annotated_frame, "SCREENSHOT!", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    cv2.imshow("YOLOv8 Webcam - Personen, Abstand, Screenshot", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Programm beendet.")