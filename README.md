# YOLOv8 Football Player and Ball Tracker

## Introduction

Welcome to the YOLOv8 Football Player and Ball Tracker project! As the developer, my goal is to provide a comprehensive solution for detecting and tracking players, referees, and footballs in videos using the powerful YOLOv8 AI object detection model. Here's what you can expect from this project:

- **Object Detection with YOLOv8**:
  Utilizing YOLOv8, one of the best AI object detection models available, to accurately detect players, referees, and footballs in video footage.

- **Model Training for Performance Improvement**:
  Training the YOLOv5 model to enhance its performance and ensure accurate detection and tracking results.

- **Team Assignment based on Jersey Colors**:
  Implementing Kmeans for pixel segmentation and clustering to assign players to teams based on the colors of their jersey.

- **Ball Acquisition Percentage Measurement**:
  Utilizing team assignments to measure a team's ball acquisition percentage during a match.

- **Optical Flow for Camera Movement Measurement**:
  Using optical flow to measure camera movement between frames, providing precise data on player movement.

- **Perspective Transformation for Depth Representation**:
  Implementing perspective transformation to represent the scene's depth and perspective, enabling measurements of player speed and distance in meters rather than pixels.

- **Speed and Distance Covered Calculation**:
  Calculating a player's speed and the distance covered during gameplay.

This project covers various advanced concepts and addresses real-world problems in football match analysis. Whether you're a beginner or an experienced machine learning engineer, you'll find this project both informative and practical. Let's dive in and explore the exciting world of football tracking with AI!

![Screenshot](img/project_screenshot.png)

## Technology Used

The following modules are used in this project:

- YOLOv8: AI object detection model
- Kmeans: Pixel segmentation and clustering to detect t-shirt color
- Optical Flow: Measure camera movement
- Perspective Transformation: Represent scene depth and perspective
- Speed and distance calculation per player

## Requirements

To run this project, you need to have the following requirements installed:

- Python 3.x
- ultralytics
- supervision
- OpenCV
- NumPy
- Matplotlib
- Pandas

## Getting started

To run this demo project, create an virtual environment and install the src package:

1. create .env files with the following secret keys:

```bash
ROBOFLOW_API_KEY=
```

```bash
# install src package in development mode to install setup.py
pip install -e .

# install dependencies in requirements.txt file
pip install -r requirements.txt
```

Run the main program:

```bash
python src/main.py
```

## Challanges and Recommendation

- Camera movement: Unable to plot view perspective consistently. Better if I can get football tactical video with wide angle view and fixed camera.

## Future Improvement

- [x] Move code into src folder
- [x] Setup logging
- [x] Add config.yaml for all global variables and configuration variables
- [ ] Draw and display player movement as top view grid map.
