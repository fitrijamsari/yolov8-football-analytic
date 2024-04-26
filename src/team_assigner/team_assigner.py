from sklearn.cluster import KMeans


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}  # {1: [R, G, B], 2: [R, G, B]}
        self.player_team_dict = {}  # {player_id: team_id}

    def get_cluster_model(self, image):
        # Reshape image into 2D array
        image_2d = image.reshape(-1, 3)

        # perform k-means clustering with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame: list, bbox: list) -> list:
        cropped_image = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]

        top_half_image = cropped_image[
            0 : int(cropped_image.shape[0] / 2), 0 : int(cropped_image.shape[1])
        ]

        # Get the cluster model
        kmeans = self.get_cluster_model(top_half_image)

        # get the cluster labels for each pixel
        labels = kmeans.labels_

        # reshape the labels into the original image shape
        cluster_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # get the class on each edge corners
        corner_clusters = [
            cluster_image[0, 0],
            cluster_image[0, -1],
            cluster_image[-1, 0],
            cluster_image[-1, -1],
        ]
        non_player_clusters = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_clusters
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame: list, player_detections: dict) -> list:
        player_colors = []

        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        # cluster team colors (2 colors)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame: list, player_bbox: list, player_id: list):
        # check if player has already been assigned a team
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        else:
            player_color = self.get_player_color(frame, player_bbox)
            team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]  # 0 or 1
            team_id += 1  # team_id = 1 or 2

            # temperary solution hard coded assign team id
            if player_id in [98, 89, 115]:
                team_id = 2  # white team 2, green team 1

            self.player_team_dict[player_id] = team_id

            return team_id
