import torch
import torch.nn as nn

class ConvNet(nn.Module):
     def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.flatten = nn.flatten()


    def forward(self, x) -> torch.Tensor:
        out = self.layer1(x)
        out = self.layer2(out)
        ### This doesn't rehsape to [1, w * h * ch]
        out = self.drop_out(out)
        out = self.flatten(out)
        return out


