import os
import json

def Classes(root):

    matches = os.listdir(root)
    match_files = [os.path.join(root, match_file) for match_file in matches]
    jerseys = []
    for path in match_files:
        json_dir, video_dir = sorted(os.listdir(path), key=lambda x: (x))
        json_dir, video_dir = os.path.join(path, json_dir), os.path.join(path, video_dir)
        json_file = open(json_dir, "r")
        annotations = json.load(json_file)["annotations"]
        players = [player for player in annotations if player["category_id"] == 4]
        jersey_numbers = [jersey_number["attributes"]["jersey_number"] for jersey_number in players]
        jerseys.extend(jersey_numbers)
    return sorted(list(set(jerseys)), key= lambda x: int(x), reverse=False)



classes = Classes(root="/home/acer/Documents/Code/Excercise/football-20230818T081210Z-001/football")
