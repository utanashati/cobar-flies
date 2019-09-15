"""model.py

Neural network to train on the labeled data.
"""

import sys
import torch
from torch import nn

class Net(nn.Module):
    def __init__(self, batch_size=32, auxiliary=True):
        super(Net, self).__init__()
        self.batch_size = batch_size
        self.auxiliary = auxiliary

        # CNN 2
        self.features = nn.Sequential(                             # 1 x 512 x 96
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),   # 8 x 512 x 96
            nn.MaxPool2d(kernel_size=2, stride=2),                 # 8 x 256 x 48

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # 16 x 128 x 24

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=4),                 # 32 x 32 x 6

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),       # 64 x 8 x 3

            nn.Conv2d(64, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(3, 8), stride=(3, 8))        # 512 x 1 x 1
        )
        
        """
        # CNN 1
        self.features = nn.Sequential(                  # 1 x 512 x 96
            nn.Conv2d(1, 8, kernel_size=4, stride=4),   # 8 x 128 x 24
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),      # 16 x 64 x 12

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),      # 32 x 32 x 6

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),      # 64 x 8 x 3

            nn.Conv2d(64, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(3, 8), stride=(3, 8))      # 512 x 1 x 1
        )
        """

        if self.auxiliary:
            self.classifier_1 = nn.Sequential(
                nn.Linear(512, 512),
                nn.Dropout(),
                nn.ReLU(inplace=True),
                nn.Linear(512, 2),
            )

            self.classifier_2 = nn.Sequential(
                nn.Linear(512, 512),
                nn.Dropout(),
                nn.ReLU(inplace=True),
                nn.Linear(512, 2),
            )

        self.classifier_final = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4),
        )

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def forward(self, x):
        x = self.features(x).view(self.batch_size, -1)
        if self.auxiliary:
            x_1 = self.classifier_1(x)
            x_2 = self.classifier_2(x)
            x_fin = self.classifier_final(x)
            return x_1, x_2, x_fin
        else:
            x_fin = self.classifier_final(x)
            return x_fin

    def _update_progress(self, progress, avg_loss, val_loss):
        length = 20
        status = ""
        try:
            progress = float(progress)
        except TypeError:
            progress = 0
            status = "Error: progress must be numeric\r\n"

        if progress < 0:
            progress = 0
            status = "Error: progress must be >= 0\r\n"
        if progress >= 1:
            progress = 1
            status = "Fin\n"

        block = int(round(length * progress))
        text = \
            "\rPercent: [{}] {:3.0f}% " \
            "TL:{:1.5f} VL:{:1.5f} {}".format(
                "#" * block + "-" * (length - block),
                round(progress * 100, 2),
                avg_loss, val_loss, status
            )
        sys.stdout.write(text)
        sys.stdout.flush()
