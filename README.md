# yolo_webcam_project
Real‑Time Object Detection on a Webcam with YOLOv8 and Custom Filter

Step 1: prepare project folder and python environment
Step 2: installation of essential python packets

 #####  Goal is to install 3 Libraries: 
  - ultralytics : has YOLOv8n (AI model for object/image detection)
  - openCV-python : for webcam and object 
  - and NumPy : for operations
Step 3 : first script webcam + yolov8n
Step 4 : self additional logic
  - Option A: count person
  shows top left the number of person present
  - Option B: calculate distance between 2 person + alarm
  if two person are too closed, then write red "distance too small"
  - Option C: Screenshot by detected object
  always store a screenshot when an object is detected.
