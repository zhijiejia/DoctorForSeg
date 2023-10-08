

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.init import kaiming_normal_, constant_

def conv(in_planes, out_planes, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU()
    )

def deconv(in_planes, out_planes, scale_factor=2):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=1, bias=True),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(),
        nn.UpsamplingBilinear2d(scale_factor=scale_factor),
    )

def predict_mask(in_planes, channel=9):
    return nn.Conv2d(in_planes, channel, kernel_size=3, stride=1, padding=1, bias=True)

class SpixelNet(nn.Module):

    def __init__(self, batchNorm=True):
        super(SpixelNet,self).__init__()

        self.batchNorm = batchNorm
        self.assign_ch = 9
        
        self.conv2 = conv(2048, 1024) 
        self.deconv2 = deconv(2048, 1024)

        self.conv1 = conv(1024, 512) 
        self.deconv1 = deconv(1024, 512)     

        self.conv0 = conv(512, 256) 
        self.deconv0 = deconv(512, 256)    

        self.conv = conv(256, 64) 

        self.pred_mask = predict_mask(64, self.assign_ch)
        self.softmax = nn.Softmax(1)

    def forward(self, xs):

        x = self.deconv2(xs[-1])
        x = torch.cat([F.interpolate(xs[-2], size=x.shape[2:], mode='bilinear'), x], dim=1)
        x = self.conv2(x)
        
        x = self.deconv1(x)
        x = torch.cat([F.interpolate(xs[-3], size=x.shape[2:], mode='bilinear'), x], dim=1)
        x = self.conv1(x)
        
        x = self.deconv0(x)
        x = torch.cat([F.interpolate(xs[-4], size=x.shape[2:], mode='bilinear'), x], dim=1)
        x = self.conv0(x)

        x = self.conv(x)
        prob = self.pred_mask(x)
        return self.softmax(prob)
        
