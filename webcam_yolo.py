import cv2
from ultralytics import YOLO
from datetime import datetime
import os
import math

# 1. load the pretrained YOLOv8-model (the smallest version "yolov8n.pt")
#    The data will be automatic downloaded in the first start

model = YOLO("yolov8n.pt")

# 2. open webcam ( 0 = first camera; in case there is many, try 1, 2, ... )
cap = cv2.VideoCapture(0)

# 3. verify if the camera is okay
if not cap.isOpened():
    print("Error: Camera cannot be opened.")
    exit()

print("Webcam run. tippe 'q' on the image/object-section to close.")

# 4. folder for Screenshots
screenshot_folder = "screenshots"
if not os.path.exists(screenshot_folder):
    os.makedirs(screenshot_folder)
    
print("Webcam runs. tippe 'q' to close ")
print("Features: count person | distance alarm | screenshots by 'Handy' or 'cell phone'")


# Help function: distance between 2 person
def calculate_distance(point1, point2):
    return math.hypot(point1[0] - point2[0], point1[1] - point2[1])


# 4. Endless loop: a frame after another
while True:
    #read Frame from camera
    ret, frame = cap.read()
    if not ret:
        print("Error: has no Frame")
        break
    
    # Compile YOLO-Object detection on the actual frame
    # results has the detected object
    
    results = model(frame)
    
    
    # Results on the Frame label (Bounding Boxes, Labels, Confidence)
    # 'plot()' returns a new object with label
    
    annotated_frame = results[0].plot()
    
    
    
    # extract data from results
    boxes = results[0].boxes
    person_centers = []
    detected_objects = []
    
    if boxes is not None:
        for box in boxes:
            # coordiate of box
            x1, y1, x2, y2 = map(int box.xyxy[0].tolist())
            cls = int(box.cls[0])
            class_name = model.names[cls]
            confidence = float(box.conf[0])
            detected_objects.append(class_name)
            
            
            # calculate middle point
            
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # if there's someone, store the middle point
            if class_name == "person":
                person_centers.append((center_x, center_y))
    # ------- Feature A : count person
    person_count = len(person_centers)
    
    #Show text on the image
    cv2.putText(annotated_frame, f"Person: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # -------- Feature B : alam distance
    min_distance = 150  #pixel
    alarm_active = False
    
    if person_count >= 2:
        # verify alle couple of person
        for i in range(len(person_centers)):
            for j in range(i+1, len(person_centers)):
                dist = calculate_distance(person_centers[i], person_centers[j])
                if dist < min_distance:
                    alarm_active = True
                    #show the red line between the 2 person
                    cv2.line(annotated_frame, person_centers[i], person_centers[j], (0, 0, 255), 2)
                    # writeback the line distance
                    cv2.putText(annotated_frame, f"{int(dist)}px",
                                ((person_centers[i][0] + person_centers[j][0]) // 2,
                                 (person_centers[i][1] + person_centers[j][1]) // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0, 5, (0, 0, 255), 1)
    if alarm_active:
        cv2.putText(annotated_frame, "Attention: Distance too small!", (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(annotated_frame, "distance okay", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
    
    # ------ Feature C: Screenshot for detected objects
    # define, by which object, we have a screenshot
    trigger_objects = ["cell phone", "handy"]
    screenshot_taken = False
    for obj in trigger_objects:
        if obj in detected_objects:
            # Screenshot speichern mit Zeitstempel
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{screenshot_folder}/{obj}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Screenshot gespeichert: {filename}")
            screenshot_taken = True
            break
    #final image
    cv2.imshow("YOLOv8 Webcam - Person, distance, Screenshot", annotated_frame)
    
    #end with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
#Arrangement
cap.release()
cv2.dsetroyAllWindows()
print("Program end.")
    

# 5. Arrange: Camera free and close Wndow/section

cap.release()
cv2.destroyAllWindows()
print("Program end.")