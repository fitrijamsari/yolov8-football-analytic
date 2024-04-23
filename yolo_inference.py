from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")

results = model.predict("input_videos/bundesliga_video.mp4", save=True)

print(results[0])
print("============================================================")

for box in results[0].boxes:
    print(box)
