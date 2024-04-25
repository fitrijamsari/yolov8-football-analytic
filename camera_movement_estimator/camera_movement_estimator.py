import os
import pickle
import sys

import cv2
import numpy as np

sys.path.append("../")
from utils import measure_euclidean_distance, measure_xy_distance


class CameraMovementEstimator:
    def __init__(self, frame):

        self.minimum_distance = 5

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1  # top side
        mask_features[:, 900:1050] = 1  # bottom side

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features,
        )

    def add_adjust_position_to_tracks(
        self,
        tracks: list,
        camera_movement_per_frame: list,
    ):
        for cls_name, cls_track in tracks.items():
            for frame_num, frame_info in enumerate(cls_track):
                for track_id, track_bbox in frame_info.items():
                    position = track_bbox["position"]
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (
                        position[0] - camera_movement[0],
                        position[1] + camera_movement[1],
                    )
                    tracks[cls_name][frame_num][track_id][
                        "position_adjusted"
                    ] = position_adjusted

    def get_camera_movement(
        self, frames: list, read_from_stub: bool = False, stub_path: str = None
    ) -> list:
        # Read the stub
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            print(f"Loading camera_movement stub from {stub_path}")
            with open(stub_path, "rb") as f:
                camera_movement = pickle.load(f)
            return camera_movement

        # Estimate the camera movement
        camera_movement = [[0, 0]] * len(frames)

        prev_grayscale = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        prev_features = cv2.goodFeaturesToTrack(prev_grayscale, **self.features)

        for frame_num in range(1, len(frames)):
            frame_grayscale = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)

            # Calculate optical flow using Lucas-Kanade method
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(
                prev_grayscale, frame_grayscale, prev_features, None, **self.lk_params
            )

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            for i, (new, prev) in enumerate(zip(new_features, prev_features)):
                new_features_point = new.ravel()  # flatten multi-dimension array
                prev_features_point = prev.ravel()

                distance = measure_euclidean_distance(
                    new_features_point, prev_features_point
                )

                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(
                        prev_features_point, new_features_point
                    )

            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                prev_features = cv2.goodFeaturesToTrack(
                    frame_grayscale, **self.features
                )

            prev_grayscale = frame_grayscale.copy()

        # save as pickle file for easy test and load
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def draw_camera_movement(self, frames: list, camera_movement_per_frame: list):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 80), (0, 255, 0), cv2.FILLED)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(
                frame,
                f"Camera Movement X : {x_movement:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                thickness=3,
            )

            frame = cv2.putText(
                frame,
                f"Camera Movement Y : {y_movement:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                thickness=3,
            )

            output_frames.append(frame)

        return output_frames
