import torch
import torch.nn as nn

class ConvNet(nn.Module):
     def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(5, 25, kernel_size=3, padding=1),
            nn.ReLU())
            #nn.MaxPool1d(kernel_size=3, stride=1))
        self.layer2 = nn.Sequential(
                nn.Conv1d(25, 5, kernel_size=1),
                nn.ReLU())
        self.drop_out = nn.Dropout()


     def forward(self, x):
         out = self.layer1(x)
         out = self.layer2(out)
         out = self.drop_out(out)
         return out


