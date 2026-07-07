from setuptools import setup, find_packages

setup(
    name="yolo-webcam-security",
    version="1.0.0",
    description="Real-time object detection with YOLOv8 and social distancing alarm",
    author="Pascal C. Nague",
    author_email="naguepascal5@gmail.com",
    url="https://github.com/NPCabrel/yolo_webcam_project",
    packages=find_packages(),
    install_requires=[
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "yolo-security=src.main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)