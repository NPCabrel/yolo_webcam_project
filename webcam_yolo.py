import cv2
from ultralytics import YOLO

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


# 4. Endless loop: a frame after another
while true:
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
    
    # Show the final object/image
    
    cv2.imshow("YOLOv8 Webcam", annotated_frame)
    
    # Wait to Tipping: Since 'q' is tipped, close/breakdown.
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    

# 5. Arrange: Camera free and close Wndow/section

cap.release()
cv2.destroyAllWindows()
print("Program end.")