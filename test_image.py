import numpy as np
import torch
import torch.nn as nn
from models import CNN
import argparse
import cv2
classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# This classes just to illustrate that we have 20 players

def get_args():
    parser = argparse.ArgumentParser(description="Train an CNN model")
    parser.add_argument("--image_path", "-i", type=str, default="/home/acer/Pictures/Cristiano-Ronaldo-back-vs-Atletico-Madrid.jpg")
    parser.add_argument("--image_size", "-s", type=int, default=224)
    parser.add_argument("--checkpoint_path", "-m", type=str, default="football_football_transfer_learning_checkpoints/best.pt")
    args = parser.parse_args()
    return args


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(num_classes=20).to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    image = cv2.imread(args.image_path)
    # Preprocess image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.image_size, args.image_size))
    image = np.transpose(image, (2, 0, 1))/255.
    image = image[None, :, :, :]
    image = torch.from_numpy(image).float()
    image = image.to(device)
    softmax = nn.Softmax()
    with torch.no_grad():
        output = model(image)
    predicted_class = classes[torch.argmax(output)]
    print("The prediction result of model is {}".format(predicted_class))




if __name__ == '__main__':
    args = get_args()
    test(args)
