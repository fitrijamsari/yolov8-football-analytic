import torch
from ultralytics import YOLO

print(torch.backends.mps.is_available())

# Load a model
model = YOLO("../models/best.pt")

video_path = "../input_videos/World-Cup-2022-Spain-Costa-Rica.mp4"

results = model(source=video_path, show=True, conf=0.1, save=True, device="mps")
