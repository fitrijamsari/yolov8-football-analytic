import sys

sys.path.append("../")
from utils import get_center_of_bbox, measure_euclidean_distance


class PlayerBallAssigner:
    def __init__(self):
        self.max_player_ball_distance = 70

    def assign_ball_to_player(self, players: dict, ball_bbox: list):
        ball_position = get_center_of_bbox(ball_bbox)

        minimum_distance = 9999
        assigned_player_id = -1

        for player_id, player in players.items():
            player_bbox = player["bbox"]

            distance_left = measure_euclidean_distance(
                (player_bbox[0], player_bbox[-1]), ball_position
            )
            distance_right = measure_euclidean_distance(
                (player_bbox[2], player_bbox[-1]), ball_position
            )
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance:
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player_id = player_id

        return assigned_player_id
