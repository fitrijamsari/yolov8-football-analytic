import cv2


def read_video(video_path: str) -> list:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    return frames


def save_video(output_video_frames: list, output_video_path: str):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        24.0,
        (output_video_frames[0].shape[1], output_video_frames[0].shape[0]),
    )
    for frame in output_video_frames:
        out.write(frame)
    out.release()
