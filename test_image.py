import numpy as np
import torch
import torch.nn as nn
from models import CNN
import argparse
import cv2
from classes import classes


def get_args():
    parser = argparse.ArgumentParser(description="Train an CNN model")
    parser.add_argument("--image_path", "-i", type=str, default="/home/acer/Pictures/r1149798_1296x729_16-9.jpg")
    parser.add_argument("--image_size", "-s", type=int, default=224)
    parser.add_argument("--checkpoint_path", "-m", type=str, default="football_football_transfer_learning_checkpoints/best.pt")
    args = parser.parse_args()
    return args


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(num_classes=20).to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    image = cv2.imread(args.image_path)
    # Preprocess image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.image_size, args.image_size))
    image = np.transpose(image, (2, 0, 1))/255.
    # image = np.expand_dims(image, axis=0)
    image = image[None, :, :, :]
    image = torch.from_numpy(image).float()
    image = image.to(device)
    softmax = nn.Softmax()
    with torch.no_grad():
        output = model(image)
    predicted_class = classes[torch.argmax(output)]
    print("The image is about {} with probability of {}".format(predicted_class))




if __name__ == '__main__':
    args = get_args()
    test(args)
