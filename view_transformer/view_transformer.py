import cv2
import numpy as np


class ViewTransformer:
    def __init__(self):
        field_width = 68
        field_length = 23.82

        # TODO: Create script to get pixel value from images or videos by clicking
        self.pixel_vertices = np.array(
            [[110, 1035], [265, 275], [910, 260], [1640, 915]]
        )  # bottom left, top left, top right, bottom right

        self.target_vertices = np.array(
            [[0, field_width], [0, 0], [field_length, 0], [field_length, field_width]]
        )

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(
            self.pixel_vertices, self.target_vertices
        )

    def transform_point(self, point: list) -> list:
        p = (int(point[0]), int(point[1]))
        # check if point (player bottom middle) inside of the field
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None

        # reshape point to the format that support for persepective transform
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        transformed_point = cv2.perspectiveTransform(
            reshaped_point, self.perspective_transformer
        )

        return transformed_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks: dict) -> dict:
        for cls_name, cls_track in tracks.items():
            for frame_num, frame_info in enumerate(cls_track):
                for track_id, track_bbox in frame_info.items():
                    position = track_bbox["position_adjusted"]
                    position = np.array(position)
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    tracks[cls_name][frame_num][track_id][
                        "position_transformed"
                    ] = position_transformed  # position_transformed is not in meter projection coordinates
