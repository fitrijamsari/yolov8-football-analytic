import logging

import numpy as np

from camera_movement_estimator import CameraMovementEstimator
from player_ball_assigner import PlayerBallAssigner
from speed_distance_estimator import SpeedDistanceEstimator
from src.logging_conf import setup_logging
from team_assigner import TeamAssigner
from trackers import Tracker
from utils import read_video, save_video
from view_transformer import ViewTransformer


def main():
    # Setup Logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Read Video
    logger.info("Reading video")
    video_frames = read_video("../input_videos/bundesliga_video.mp4")

    # Initialize Tracker
    logger.info("Initializing tracker")
    tracker = Tracker("../models/best.pt")

    tracks = tracker.get_object_tracks(
        video_frames, read_from_stub=True, stub_path="stubs/track_stubs.pkl"
    )

    # Get object positions
    logger.info("Get object positions")
    tracker.add_position_to_tracks(tracks)

    # Camera Movement Estimator
    logger.info("Running Camera Movement Estimator")
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames, read_from_stub=True, stub_path="stubs/camera_movement_stubs.pkl"
    )

    camera_movement_estimator.add_adjust_position_to_tracks(
        tracks, camera_movement_per_frame
    )

    # View Transformer
    logger.info("Running Perspective Transformation")
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    logger.info("Interpolating ball positions")
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and Distance Estimator
    logger.info("Running Speed and Distance Estimator")
    speed_distance_estimator = SpeedDistanceEstimator()
    speed_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    logger.info("Running Assigning player teams")
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

    # Loop for each frame and assign team for each players
    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team_id = team_assigner.get_player_team(
                video_frames[frame_num], track["bbox"], player_id
            )
            tracks["players"][frame_num][player_id]["team"] = team_id
            tracks["players"][frame_num][player_id]["team_color"] = (
                team_assigner.team_colors[team_id]
            )

    # Assign Ball Acquisition
    logger.info("Running Assigning ball acquisition")
    player_assigner = PlayerBallAssigner()
    team_ball_control = []  # which team has the ball in each frame

    for frame_num, player_track in enumerate(tracks["players"]):
        ball_cls_id = 1
        ball_bbox = tracks["ball"][frame_num][ball_cls_id]["bbox"]
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(
                tracks["players"][frame_num][assigned_player]["team"]
            )
        else:
            team_ball_control.append(team_ball_control[-1])  # appoint the last team

    team_ball_control = np.array(team_ball_control)

    # Draw Output
    ## Draw object tracks
    logger.info("Drawing object tracks")
    output_video_frames = tracker.draw_annotations(
        video_frames, tracks, team_ball_control
    )

    ## Draw camera movement
    logger.info("Drawing camera movement")
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, camera_movement_per_frame
    )

    ## Draw speed and distance
    logger.info("Drawing speed and distance")
    output_video_frames = speed_distance_estimator.draw_speed_and_distance(
        output_video_frames, tracks
    )

    # Save Video
    logger.info("Saving video")
    save_video(output_video_frames, "../output_videos/bundesliga_video_output.mp4")

    logger.info("Programm Finished")


if __name__ == "__main__":
    main()
