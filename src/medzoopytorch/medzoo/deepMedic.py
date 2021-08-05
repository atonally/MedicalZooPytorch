import torch.nn as nn
import torch
from torchsummary import summary

from medzoopytorch.medzoo.BaseModelClass import BaseModel

# adapt from https://github.com/Kamnitsask/deepmedic
class DeepMedic(BaseModel):
    
    def __init__(self, in_channels, n_classes, n1=70, n2=80, n3=90, m=250, up=True):
        super(DeepMedic, self).__init__()
        #n1, n2, n3 = 30, 40, 50
        self.in_channels = in_channels
        self.n_classes = n_classes
        
        n = 3*n3

        self.branch1 = nn.Sequential(
                conv3x3(self.in_channels, n1),
                conv3x3(n1, n1),
                ResBlock(n1, n2),
                ResBlock(n2, n2),
                ResBlock(n2, n3))

        self.branch2 = nn.Sequential(
                conv3x3(self.in_channels, n1),
                conv3x3(n1, n1),
                conv3x3(n1, n2),
                conv3x3(n2, n2),
                conv3x3(n2, n2),
                conv3x3(n2, n2),
                conv3x3(n2, n3),
                conv3x3(n3, n3))

        self.branch3 = nn.Sequential(
                conv3x3(self.in_channels, n1),
                conv3x3(n1, n1),
                conv3x3(n1, n2),
                conv3x3(n2, n2),
                conv3x3(n2, n2),
                conv3x3(n2, n2),
                conv3x3(n2, n3),
                conv3x3(n3, n3))

        self.up3 = nn.Upsample(scale_factor=3,
                mode='trilinear', align_corners=False) if up else repeat
        self.up5 = nn.Upsample(scale_factor=5,
                mode='trilinear', align_corners=False) if up else repeat

        self.fc = nn.Sequential(
                nn.Dropout3d(p=0.1),
                conv3x3(n, m, 1),
                nn.Dropout3d(p=0.5),
                conv3x3(m, m, 1),
                nn.Dropout3d(p=0.5),
                nn.Conv3d(m, self.n_classes, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        
        x1, x2, x3 = inputs
        x1 = self.branch1(x1)
        x2 = self.branch2(x2)
        x3 = self.branch3(x3)
        x2 = self.up3(x2)
        x3 = self.up5(x3)
        x = torch.cat([x1, x2, x3], 1)
        x = self.fc(x)
        return x


    def test(self,device='cpu'):

        input_tensor = torch.rand(1, 2, 32, 32, 32)
        ideal_out = torch.rand(1, self.n_classes, 32, 32, 32)
        out = self.forward(input_tensor)
        assert ideal_out.shape == out.shape
        summary(self.to(torch.device(device)), (2, 32, 32, 32),device='cpu')
        # import torchsummaryX
        # torchsummaryX.summary(self, input_tensor.to(device))
        print("DeepMedic test is complete")

    
   
    
class ResBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(ResBlock, self).__init__()
        self.inplanes = inplanes
        self.conv1 = nn.Conv3d(inplanes, planes, 3, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, 3, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        x = x[:, :, 2:-2, 2:-2, 2:-2]
        y[:, :self.inplanes] += x
        y = self.relu(y)
        return y

def conv3x3(inplanes, planes, ksize=3):
    return nn.Sequential(
            nn.Conv3d(inplanes, planes, ksize, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True))

def repeat(x, n=3):
    # nc333
    b, c, h, w, t = x.shape
    x = x.unsqueeze(5).unsqueeze(4).unsqueeze(3)
    x = x.repeat(1, 1, 1, n, 1, n, 1, n)
    return x.view(b, c, n*h, n*w, n*t)
