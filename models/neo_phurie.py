import torch

class NeoPHURIE(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.convolutions = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 5),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(3, 2),

            torch.nn.Conv2d(32, 64, 3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(3, 2),

            torch.nn.Conv2d(64, 64, 3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(3, 2),

            torch.nn.Conv2d(64, 64, 3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(3, 1),

            torch.nn.Conv2d(64, 128, 3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(3, 2),

            torch.nn.Conv2d(128, 128, 3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(3, 2)
            )

        self.linears = torch.nn.Sequential(
            torch.nn.Linear(128 * 3 * 3, 512),
            torch.nn.ReLU(),

            torch.nn.Linear(512, 64),
            torch.nn.ReLU(),

            torch.nn.Linear(64, 1))

    def forward(self, x):

        for op in self.convolutions:
            x = op(x)

        x = torch.flatten(x, 1)

        for op in self.linears:
            x = op(x)

        return x

    def predict_from_embedding(self, x):
        for op in self.linears:
            x = op(x)

        return x
