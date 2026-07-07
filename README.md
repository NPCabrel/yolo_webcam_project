# YOLO-Webcam Security System

**Real-time object detection with YOLOv8 + OpenCV – designed for industrial surveillance and social distancing monitoring.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-red)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



##  Features

- **Real-time detection** at **30 FPS** on standard hardware (CPU only)
- **People counting** with live display
- **Social distancing alarm** – visual alert when two persons are too close (threshold configurable)
- **Smart screenshot** – automatically saves images when specific objects (e.g., `cell phone`) are detected
- **Modular architecture** – easy to extend with new models or filters
- **Fully documented** codebase with unit tests


## Architecture

Have a look under asset/yolo-architecture

---

## Quick Start

### Prerequisites
- Python 3.9+
- Webcam

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/NPCabrel/yolo_webcam_project.git
cd yolo_webcam_project

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python src/main.py
