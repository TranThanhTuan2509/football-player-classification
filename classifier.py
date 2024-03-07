import torch.nn as nn
import torch
import torchvision.models as models

class PlayerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50()
        del self.backbone.fc
        # 3 = number of digits: 0 -> 1 digit. 1 -> 2 digits. 2 -> unknown
        self.backbone.fc1 = nn.Linear(2048, 3)
        # 11 = possibilities for unit digit
        # 0 ~ (10). 1 ~ (1, 11). 2 ~ (2, 12), ... 9 ~ (9, 19). 10 ~ unknown
        self.backbone.fc2 = nn.Linear(2048, 11)
        # 1 = number of colors. 0 = white. 1 = black
        self.backbone.fc3 = nn.Linear(2048, 2)  # 2 = number of colors

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        out1 = self.backbone.fc1(x)
        out2 = self.backbone.fc2(x)
        out3 = self.backbone.fc3(x)

        return out1, out2, out3
