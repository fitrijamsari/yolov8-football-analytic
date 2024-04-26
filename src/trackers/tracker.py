import os
import pickle
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.append("../")
import supervision as sv
from ultralytics import YOLO

from utils import get_bbox_width, get_bottom_middle_bbox, get_center_of_bbox


class Tracker:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks: dict) -> dict:
        for cls_name, cls_track in tracks.items():
            for frame_num, frame_info in enumerate(cls_track):
                for track_id, track_bbox in frame_info.items():
                    bbox = track_bbox["bbox"]
                    if cls_name == "ball":
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_bottom_middle_bbox(bbox)  # foot position
                    # save position in tracks
                    tracks[cls_name][frame_num][track_id]["position"] = position

    def interpolate_ball_positions(self, ball_positions: list) -> list:
        """
        Interpolate the ball positions in a list of dictionaries.

        Args:
            ball_positions (list): A list of dictionaries containing the ball positions. Each dictionary has the following structure:
                {
                    <ball_cls_id>: {
                        "bbox": [x1, y1, x2, y2]
                    }
                }

        Returns:
            list: A list of dictionaries containing the interpolated ball positions. Each dictionary has the same structure as the input.

        This function takes a list of dictionaries representing the ball positions and interpolates the missing values. It converts the list of dictionaries into a pandas DataFrame for interpolation, performs the interpolation using the `interpolate()` method, and then converts the DataFrame back into a list of dictionaries.

        The function assumes that the ball positions are represented as a list of dictionaries, where each dictionary contains the ball class ID as the key and a dictionary with the "bbox" key and its corresponding value as the value. The "bbox" value is a list representing the coordinates of the bounding box of the ball.

        The function returns a list of dictionaries with the interpolated ball positions. Each dictionary has the same structure as the input, but the "bbox" value is a list representing the interpolated bounding box coordinates of the ball.
        """

        ball_cls_id = 1
        ball_positions = [
            x.get(ball_cls_id, {}).get("bbox", []) for x in ball_positions
        ]
        # convert to pd dataframe for interpolation
        df_ball_positions = pd.DataFrame(
            ball_positions, columns=["x1", "y1", "x2", "y2"]
        )

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        # convert back to list of dict format
        ball_positions = [
            {ball_cls_id: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()
        ]

        return ball_positions

    def detect_frames(self, frames: list) -> list:
        """
        Perform object detection on a list of frames by skipping frames e.g 20 frames.

        Args:
            self: The Tracker object.
            frames (list): A list of frames to perform object detection on.

        Returns:
            list: A list of detections from the object detection process.
        """
        batch_size = 20
        conf = 0.1
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            detections_batch = self.model.predict(source=batch, conf=conf, device="mps")
            detections += detections_batch

        return detections

    # Since we have goalkeeper which is blinking due to inaccurate detection, i want to change goalkeeper to be players and then track them.
    # So instead of using build in ultralytics trakcer, i will use supervision ByteTracker to track after .predict() for flexible modification
    def get_object_tracks(
        self, frames: list, read_from_stub: bool = False, stub_path: str = None
    ) -> dict:
        """
        Generate the tracks of objects in each frame of the given frames.

        Parameters:
            frames (list): A list of frames.

        Returns:
            dict: A dictionary containing the tracks of objects in each frame. The dictionary has the following structure:
                {
                    "players": List[Dict[int, Dict[str, List[int]]]],
                    "referees": List[Dict[int, Dict[str, List[int]]]],
                    "ball": List[Dict[int, Dict[str, List[int]]]]
                }
                Each element in the "players", "referees", and "ball" lists represents a frame, and each frame is represented as a dictionary where the keys are track IDs and the values are dictionaries containing the bounding box coordinates of the objects.
        """

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            print(f"Loading object_tracks stub from {stub_path}")
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {"players": [], "referees": [], "ball": []}

        for frame_num, detection in enumerate(detections):
            cls_name = detection.names  # {0:person, 1:goalkeeper ...}
            cls_name_inverse = {value: key for key, value in cls_name.items()}
            # print(cls_name)

            # convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            # print(detection_supervision)

            # convert goalkeepar to player object
            for object_idx, class_id in enumerate(detection_supervision.class_id):
                if cls_name[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_idx] = cls_name_inverse[
                        "player"
                    ]

            # Track objects
            detection_with_tracks = self.tracker.update_with_detections(
                detection_supervision
            )
            # print(detection_with_tracks)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_name_inverse["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_name_inverse["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_name_inverse["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        # save as pickle file for easy test and load
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(
        self, frame: list, bbox: list, color: tuple, track_id: int = None
    ) -> list:
        y2 = int(bbox[3])

        x_center, _ = get_center_of_bbox(bbox)
        bbox_width = get_bbox_width(bbox)

        # draw the ellipse
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(bbox_width), int(0.35 * bbox_width)),
            angle=0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        rectangle_width, rectangle_height = 40, 20
        x1_rectangle = x_center - rectangle_width // 2
        x2_rectangle = x_center + rectangle_width // 2
        y1_rectangle = (y2 - rectangle_height // 2) + 15
        y2_rectangle = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                pt1=(int(x1_rectangle), int(y1_rectangle)),
                pt2=(int(x2_rectangle), int(y2_rectangle)),
                color=color,
                thickness=cv2.FILLED,
            )
            x1_text = x1_rectangle + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                str(track_id),
                (int(x1_text), int(y1_rectangle + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                thickness=2,
            )

        return frame

    def draw_triangle(
        self,
        frame: list,
        bbox: list,
        color: tuple,
    ) -> list:
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array(
            [
                [x, y],
                [x - 10, y - 20],
                [x + 10, y - 20],
            ]
        )

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(
        self, frame: list, frame_nums: int, team_ball_control: list
    ) -> list:
        # Draw semi transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (0, 255, 0), cv2.FILLED)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[: frame_nums + 1]
        # Get the number of times each team has the ball
        team_1_num_frames = team_ball_control_till_frame[
            team_ball_control_till_frame == 1
        ].shape[0]
        team_2_num_frames = team_ball_control_till_frame[
            team_ball_control_till_frame == 2
        ].shape[0]

        team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)

        cv2.putText(
            frame,
            f"Team 1 Ball Control: {team_1*100:.2f}%",
            (1400, 900),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            thickness=2,
        )
        cv2.putText(
            frame,
            f"Team 2 Ball Control: {team_2*100:.2f}%",
            (1400, 950),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            thickness=2,
        )

        return frame

    def draw_annotations(
        self, video_frames: list, tracks: dict, team_ball_control: list
    ) -> list:
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # Draw players
            for track_id, player in player_dict.items():
                bbox = player["bbox"]
                color = player.get("team_color", (255, 255, 255))
                frame = self.draw_ellipse(frame, bbox, color, track_id)

                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame, bbox, (0, 0, 255))

            # Draw referees
            for _, referee in referee_dict.items():
                bbox = referee["bbox"]
                color = (0, 255, 255)
                frame = self.draw_ellipse(frame, bbox, color)

            # Draw ball
            for _, ball in ball_dict.items():
                bbox = ball["bbox"]
                color = (0, 255, 0)
                frame = self.draw_triangle(frame, bbox, color)

            # Draw team ball control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames
