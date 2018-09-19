class DeepStitch(nn.Module):
    def __init__(self, feature_model=None):
        super(DeepStitch, self).__init__()
        if feature_model == None:
            self.feature_layer = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        else:
            self.feature_layer = nn.Sequential(*list(feature_model.children())[:-2])
        self.match_layer = nn.Sequential(
            nn.Conv2d(1024, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.output_num = [8, 4, 2, 1]
        self.adapMaxPool = nn.AdaptiveMaxPool2d(8)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


resnet = models.resnet34()
deepStitchModel = DeepStitch(resnet)

import torch.nn as nn
import torch
class DeepStitch(nn.Module):
    def __init__(self):
        super(DeepStitch, self).__init__()
        self.feature_layer2 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.feature_layer3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.match_layer1 = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.match_layer2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.match_layer3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.adapMaxPool = nn.AdaptiveMaxPool2d(8)
        self.fc1 = nn.Linear(2048, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, xA, xB):
        match_1 = self.match_layer1(torch.cat([xA, xB], 1))

        feature_1A = self.feature_layer2(xA)
        feature_1B = self.feature_layer2(xB)
        match_2 = self.match_layer2(torch.cat([feature_1A, feature_1B], 1))

        feature_2A = self.feature_layer2(feature_1A)
        feature_2B = self.feature_layer2(feature_1B)
        match_3 = self.match_layer2(torch.cat([feature_2A, feature_2B], 1))

        match = match_1 + match_2 + match_3
        adpaMP = self.adapMaxPool(match)

        out = self.fc1(adpaMP)
