import torch.nn as nn


class FruitCNNDropout(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Extract features from images
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Classify images with dropout
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 12 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x