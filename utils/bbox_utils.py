def get_center_of_bbox(bbox: list) -> int:
    """
    Get the center of the bounding box.

    Parameters:
        bbox (list): The bounding box coordinates in the format [xmin, ymin, xmax, ymax].

    Returns:
        list: The center of the bounding box in the format int(x), int(y).

    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox: list) -> int:
    """
    Get the width of the bounding box.

    Parameters:
        bbox (list): The bounding box coordinates in the format [xmin, ymin, xmax, ymax].

    Returns:
        int: The width of the bounding box.

    """
    x1, y1, x2, y2 = bbox
    return x2 - x1


def measure_euclidean_distance(bbox1: list, bbox2: list) -> int:
    """
    A function that calculates the Euclidean distance between two points represented by bounding box coordinates.

    Parameters:
        bbox1 (list): The bounding box coordinates of the first point in the format [xmin, ymin, xmax, ymax].
        bbox2 (list): The bounding box coordinates of the second point in the format [xmin, ymin, xmax, ymax].

    Returns:
        int: The Euclidean distance between the two points.
    """

    return ((bbox1[0] - bbox2[0]) ** 2 + (bbox1[1] - bbox2[1]) ** 2) ** 0.5


def measure_xy_distance(bbox1: list, bbox2: list) -> int:
    """
    A function that calculates the absolute differences in x and y coordinates between two bounding boxes.

    Parameters:
        bbox1 (list): The bounding box coordinates of the first point in the format [xmin, ymin, xmax, ymax].
        bbox2 (list): The bounding box coordinates of the second point in the format [xmin, ymin, xmax, ymax].

    Returns:
        int: The absolute differences in x and y coordinates between the two points.
    """

    return abs(bbox1[0] - bbox2[0]), abs(bbox1[1] - bbox2[1])


def get_bottom_middle_bbox(bbox: list) -> list:
    """
    Get the bottom middle coordinates of a bounding box.

    Parameters:
        bbox (list): The bounding box coordinates in the format [x1, y1, x2, y2].

    Returns:
        list: The bottom middle coordinates of the bounding box in the format [int(x), int(y)].
    """

    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)
