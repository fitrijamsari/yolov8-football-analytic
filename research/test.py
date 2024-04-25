tracks = {
    "players": [
        {
            "1": {"bbox": [10, 20, 30, 40]},
            "2": {"bbox": [50, 60, 70, 80]},
            "3": {"bbox": [90, 100, 110, 120]},
        },
        {
            "1": {"bbox": [15, 25, 35, 45]},
            "2": {"bbox": [55, 65, 75, 85]},
            "3": {"bbox": [95, 105, 115, 125]},
        },
        {
            "1": {"bbox": [19, 29, 39, 49]},
            "2": {"bbox": [59, 69, 79, 89]},
            "3": {"bbox": [99, 109, 119, 129]},
        },
    ],
    "referees": [
        {
            "4": {"bbox": [10, 20, 30, 40]},
            "5": {"bbox": [50, 60, 70, 80]},
            "6": {"bbox": [90, 100, 110, 120]},
        },
        {
            "4": {"bbox": [15, 25, 35, 45]},
            "5": {"bbox": [55, 65, 75, 85]},
            "6": {"bbox": [95, 105, 115, 125]},
        },
        {
            "4": {"bbox": [19, 29, 39, 49]},
            "5": {"bbox": [59, 69, 79, 89]},
            "6": {"bbox": [99, 109, 119, 129]},
        },
    ],
    "ball": [
        {"1": {"bbox": [10, 20, 30, 40]}},
        {"1": {"bbox": [15, 25, 35, 45]}},
        {"1": {"bbox": [19, 29, 39, 49]}},
    ],
}


def main() -> None:
    # for cls_name, cls_track in tracks.items():
    #     for frame_num, object_bbox in enumerate(cls_track):
    #         for track_id, track_bbox in object_bbox.items():
    #             print(cls_name, frame_num, track_id, track_bbox)

    for cls_name, cls_tracks in tracks.items():
        if cls_name == "ball" or cls_name == "referees":
            continue
        number_of_frames = len(cls_tracks)
        print(number_of_frames)

        for frame_num in range(0, number_of_frames, 5):
            last_frame = min(frame_num + 5, number_of_frames)

            print(frame_num, last_frame)


if __name__ == "__main__":
    main()
