import torch

class OraclePHURIE(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(2 * 1152 + 2, 8192),
            torch.nn.ReLU(),
            
            torch.nn.Linear(8192, 4096),
            torch.nn.ReLU(),
            
            torch.nn.Linear(4096, 2048),
            torch.nn.ReLU(),
            
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            
            torch.nn.Linear(1024, 1153))

    def forward(self, x):
        for op in self.layers:
            x = op(x)

        return x