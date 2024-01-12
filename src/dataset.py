import cv2
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.v2 import Compose, Resize, ToTensor, ToPILImage

def collate_fn(batch):
    items = list(zip(*batch))
    images, labels = items
    final_images = torch.cat(images, dim=0)
    final_labels = []
    for l in labels:
        final_labels.extend(l)
    return final_images, torch.FloatTensor(final_labels)


class FootballDataset(Dataset):
    def __init__(self, root, transform=None, train=None):
        if train:
            paths = os.path.join(root, "train")

        else:
            paths = os.path.join(root, "test")

        self.matches = os.listdir(paths)
        self.match_files = [os.path.join(paths, match_file) for match_file in self.matches]
        self.transform = transform

        self.from_id = 0
        self.to_id = 0

        self.video_select = {}
        for path in self.match_files:
            json_dir, video_dir = sorted(os.listdir(path), key=lambda x: (x))
            json_dir, video_dir = os.path.join(path, json_dir), os.path.join(path, video_dir)
            with open(json_dir, "r") as json_file:
                json_data = json.load(json_file)
            self.to_id += len(json_data["images"])
            self.video_select[path] = [self.from_id + 1, self.to_id]
            self.from_id = self.to_id

    def __len__(self):
        return self.to_id

    def __getitem__(self, idx):
        for key, value in self.video_select.items():
            if value[0] <= idx + 1 <= value[1]:
                idx = idx - value[0] + 1
                select_path = key
        json_dir, video_dir = sorted(os.listdir(select_path), key=lambda x: (x))
        json_dir, video_dir = os.path.join(select_path, json_dir), os.path.join(select_path, video_dir)
        json_file = open(json_dir, "r")
        annotations = json.load(json_file)["annotations"]

        cap = cv2.VideoCapture(video_dir)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        flag, frame = cap.read()
        if not flag:
            print(f"Failed to read frame {idx} from video {video_dir}")
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        annotations = [anno for anno in annotations if anno["image_id"] == idx + 1 and anno["category_id"] == 4]
        box = [annotation["bbox"] for annotation in annotations]
        cropped_images = [frame[int(y):int(y + h), int(x):int(x + w)] for [x, y, w, h] in box]
        if self.transform:
            cropped_images = torch.stack([self.transform(image) for image in cropped_images])
        jerseys = [int(annotation["attributes"]["jersey_number"]) for annotation in annotations]
        return cropped_images, jerseys



