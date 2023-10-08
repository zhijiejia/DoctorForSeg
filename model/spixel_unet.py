from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from model.resnet import resnet50
from .utils import predict_prob, conv_block, up_conv


class Encoder(nn.Module):
    def __init__(self, in_ch):
        super(Encoder, self).__init__()

        filters = [64, 256, 512, 1024, 2048]
        self.encoder = resnet50(pretrained=True)

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)

        return {'e1': e1, 'e2': e2, 'e3': e3, 'e4': e4, 'e5': e5}


class Decoder(nn.Module):
    def __init__(self, out_ch, softmax=False, inter_predictions=False):
        super(Decoder, self).__init__()
        self.softmax = softmax

        filters = [64, 256, 512, 1024, 2048]
        self.inter_predictions = inter_predictions
        if inter_predictions:
            self.pre5 = predict_prob(filters[3], out_ch)
            self.pre4 = predict_prob(filters[2], out_ch)
            self.pre3 = predict_prob(filters[1], out_ch)
            self.pre2 = predict_prob(filters[0], out_ch)

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0], up=False)
        self.Up_conv2 = conv_block(filters[0] * 2, filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, e, h, w):
        e1, e2, e3, e4, e5 = e['e1'], e['e2'], e['e3'], e['e4'], e['e5']

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        p5 = self.pre5(d5) if self.inter_predictions else None

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        p4 = self.pre4(d4) if self.inter_predictions else None

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        p3 = self.pre3(d3) if self.inter_predictions else None

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        p2 = self.pre2(d2) if self.inter_predictions else None

        out = self.Conv(d2)
        out = F.interpolate(out, (h, w), mode='bilinear')

        if self.softmax:
            out = nn.Softmax(1)(out)

        return {'out': out, 'p2': p2, 'p3': p3, 'p4': p4, 'p5': p5} if self.inter_predictions else {'out': out,
                                                                                                    'd2': d2, 'd3': d3,
                                                                                                    'd4': d4, 'd5': d5}


class SpixelUnet(nn.Module):
    def __init__(self, in_ch=3, num_classes=2, out_ch_sp=9):
        super(SpixelUnet, self).__init__()
        self.encoder = Encoder(in_ch)
        self.aux = nn.Conv2d(2048, num_classes, kernel_size=3, padding=1, bias=False)
        self.seg_decoder = Decoder(num_classes)
        self.sp_decoder = Decoder(out_ch_sp, softmax=True, inter_predictions=True)

    def forward(self, x):
        _, _, h, w = x.shape
        encoded = self.encoder(x)
        
        res_aux = self.aux(encoded[-1])
        res_aux = F.interpolate(res_aux, (h, w), mode='bilinear')

        res_seg = self.seg_decoder(encoded, h, w)
        res_sp = self.sp_decoder(encoded, h, w)

        return res_seg['out'], res_aux, 0, res_sp['out']



