import sys

import cv2

sys.path.append("../")
from utils import get_bottom_middle_bbox, measure_euclidean_distance


class SpeedDistanceEstimator:
    def __init__(self):
        self.frame_window = 5  # calculate speed and distance for 5 frames
        self.frame_rate = 24

    def add_speed_and_distance_to_tracks(self, tracks: dict) -> None:
        """
        Adds speed and distance information to the tracks of objects over multiple frames.

        Parameters:
            tracks (dict): A dictionary containing the tracks of objects. The dictionary has the following structure:
                {
                    "class_name1": List[Dict[str, Any]],
                    "class_name2": List[Dict[str, Any]],
                    ...
                }
                Each element in the "class_nameX" lists represents a frame, and each frame is represented as a dictionary where the keys are track IDs and the values are dictionaries containing information about the object's track.

        Returns:
            dict: A dictionary containing the updated tracks of objects with speed and distance information added. The dictionary has the following structure:
                {
                    "class_name1": List[Dict[str, Any]],
                    "class_name2": List[Dict[str, Any]],
                    ...
                }
                Each element in the "class_nameX" lists represents a frame, and each frame is represented as a dictionary where the keys are track IDs and the values are dictionaries containing updated track information with speed and distance added.
        """

        total_distance = {}

        for cls_name, cls_tracks in tracks.items():
            if cls_name == "ball" or cls_name == "referees":
                continue
            number_of_frames = len(cls_tracks)

            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                for track_id, _ in cls_tracks[frame_num].items():
                    # check if track_id still exists in the last frame, we want the track_id still exist in the 1st batch frame and the last batch frame
                    if track_id not in cls_tracks[last_frame]:
                        continue

                    start_position = cls_tracks[frame_num][track_id][
                        "position_transformed"
                    ]
                    end_position = cls_tracks[last_frame][track_id][
                        "position_transformed"
                    ]

                    if start_position is None or end_position is None:
                        continue

                    distance_covered = measure_euclidean_distance(
                        start_position, end_position
                    )
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    speed_meter_per_second = distance_covered / time_elapsed
                    speed_kmh = speed_meter_per_second * 3.6

                    if cls_name not in total_distance:
                        total_distance[cls_name] = {}

                    if track_id not in total_distance[cls_name]:
                        total_distance[cls_name][track_id] = 0

                    total_distance[cls_name][track_id] += distance_covered

                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in cls_tracks[frame_num_batch]:
                            continue
                        tracks[cls_name][frame_num_batch][track_id]["speed"] = speed_kmh
                        tracks[cls_name][frame_num_batch][track_id]["distance"] = (
                            total_distance[cls_name][track_id]
                        )

    def draw_speed_and_distance(self, frames: list, tracks: list) -> list:
        # output_frames = []
        # for frame_num, frame in enumerate(frames):
        #     for object, object_tracks in tracks.items():
        #         if object == "ball" or object == "referees":
        #             continue
        #         for _, track_info in object_tracks[frame_num].items():
        #             if "speed" in track_info:
        #                 speed = track_info.get("speed", None)
        #                 distance = track_info.get("distance", None)
        #                 if speed is None or distance is None:
        #                     continue

        #                 bbox = track_info["bbox"]
        #                 position = get_bottom_middle_bbox(bbox)
        #                 position = list(position)
        #                 position[1] += 40

        #                 position = tuple(map(int, position))
        #                 cv2.putText(
        #                     frame,
        #                     f"{speed:.2f} km/h",
        #                     position,
        #                     cv2.FONT_HERSHEY_SIMPLEX,
        #                     0.5,
        #                     (0, 0, 0),
        #                     2,
        #                 )
        #                 cv2.putText(
        #                     frame,
        #                     f"{distance:.2f} m",
        #                     (position[0], position[1] + 20),
        #                     cv2.FONT_HERSHEY_SIMPLEX,
        #                     0.5,
        #                     (0, 0, 0),
        #                     2,
        #                 )
        #     output_frames.append(frame)

        # return output_frames

        output_frames = []

        for frame_num, frame in enumerate(frames):
            for cls_name, cls_tracks in tracks.items():
                if cls_name == "ball" or cls_name == "referees":
                    continue
                for _, track_bbox in cls_tracks[frame_num].items():
                    if "speed" in track_bbox:
                        speed = track_bbox.get("speed", None)
                        distance = track_bbox.get("distance", None)
                        if speed is None and distance is None:
                            continue

                        bbox = track_bbox["bbox"]
                        position = get_bottom_middle_bbox(bbox)
                        # add buffer for the position to make sure to draw below the player bbox
                        position = list(position)
                        position[1] += 40

                        position = tuple(map(int, position))
                        cv2.putText(
                            frame,
                            f"{speed:.2f} km/h",
                            position,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 0),
                            thickness=2,
                        )
                        cv2.putText(
                            frame,
                            f"{distance:.2f} m",
                            (
                                position[0],
                                position[1] + 20,
                            ),  # add buffer for the position to make sure to draw below the player bboxposition[0],
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 0),
                            thickness=2,
                        )
            output_frames.append(frame)
        return output_frames
