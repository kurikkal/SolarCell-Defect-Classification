import torch
class ResBlock(torch.nn.Module):

    def __init__(self,in_channels, out_channels, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.conv1 = torch.nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride = self.stride, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(self.out_channels)
        self.relu = torch.nn.LeakyReLU()

        self.conv2 = torch.nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(self.out_channels)

        self.downsample = torch.nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride = self.stride)
        self.bn_downsample = torch.nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        inp = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        inp = self.downsample(inp)
        inp = self.bn_downsample(inp)

        out += inp
        out = self.relu(out)

        return out



class ResNet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3,64,kernel_size=7,stride=2)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.LeakyReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2)

        self.resblock1 = ResBlock(64, 64, 1)
        self.resblock2 = ResBlock(64, 128, 2)
        self.resblock3 = ResBlock(128, 256, 2)
        self.resblock4 = ResBlock(256, 512, 2)

        self.global_avgpool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(512, 2)
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)

        x = self.global_avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x
