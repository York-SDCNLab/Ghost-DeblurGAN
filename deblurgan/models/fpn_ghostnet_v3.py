import math

import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
#import logging
#from mobilenet_v2 import MobileNetV2
#from pretrainedmodels import inceptionresnetv2
#import fpn_mobilenet as mn
import numpy as np
import cv2
#from pypapi import events, papi_high as high
#from thop import profile, clever_format
#from pthflops import count_ops
#from flops.flop_count import flop_count
#from flops.compute_flops import warmup,measure_time, fmt_res
#import tqdm
from thop import profile, clever_format

class HINet(nn.Module):

    def __init__(self, in_ch):

        super(HINet, self).__init__()
        self.instance_norm= nn.InstanceNorm2d(in_ch -in_ch//2, affine= True)

    def forward(self, x):

        channels= x.shape[1]
        channels_i= channels -channels//2
        x_i= x[:,:channels_i,:,:]
        x_r= x[:,channels_i:,:,:]
        x_i= self.instance_norm(x_i)

        return torch.cat([x_i, x_r], dim=1)


class GhostModule(nn.Module):
    """class that replaces Conv2D layers"""
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True,  bias= False):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=bias),
            #nn.InstanceNorm2d(init_channels),
            #nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=bias),
            #nn.InstanceNorm2d(new_channels),
            #nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        #print(f'\n\n{x.shape}\n\n')
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        #print(f'\n\n{x1.shape}\n\n')
        #print(f'\n\n{x2.shape}\n\n')
        out = torch.cat([x1,x2], dim=1)
        #print(f'\n\n{out.shape}\n\n')
        return out[:,:self.oup,:,:]

def count_ghostmodule(m: GhostModule, x: (torch.Tensor,), y: torch.Tensor):

    x = x[0]
    conv1= m.primary_conv[0]
    conv2- m.cheap_operation[0]

    kernel_ops_1= torch.zeros(conv1.weight.size()[2:]).numel()  # Kw x Kh
    bias_ops_1 = 1 if conv1.bias is not None else 0
    kernel_ops_2 = torch.zeros(conv2.weight.size()[2:]).numel()  # Kw x Kh
    bias_ops_2 = 1 if conv2.bias is not None else 0

    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    total_ops_1 = y.nelement() * (conv1.in_channels // conv1.groups * kernel_ops_1 + bias_ops_1)
    y_0= y-2
    x_0= x-2
    total_ops_2 = y_0.nelement() * (conv2.in_channels // conv2.groups * kernel_ops_2 + bias_ops_2)

    m.total_ops += torch.DoubleTensor([int(total_ops_1+total_ops_2)])


class FPNHead(nn.Module):

    """"this is the FPNHead class common to all backbones"""
    def __init__(self, num_in, num_mid, num_out):
        super(FPNHead, self).__init__()

        self.block0 = GhostModule(num_in, num_mid, kernel_size=3, bias= False) #nn.Conv2d(num_in, num_mid, kernel_size= 3,padding=1, bias= False)
        self.block1 = GhostModule(num_mid, num_out, kernel_size=3, bias= False)#nn.Conv2d(num_mid, num_out, kernel_size= 3,padding=1,  bias= False)

    def forward(self, x):
        x = nn.functional.relu(self.block0(x), inplace=True)
        x = nn.functional.relu(self.block1(x), inplace=True)
        return x
class FPNGhostNetv3(nn.Module):

    def __init__(self, norm_layer, output_ch=3, num_filters= 64, num_filters_fpn= 128, pretrained=True):
        super(FPNGhostNetv3, self).__init__()
        #print("\n\n GhostNET constructed \n\n")

        # Feature Pyramid Network (FPN) with four feature maps of resolutions
        # 1/4, 1/8, 1/16, 1/32 and `num_filters` filters for all feature maps.

        self.fpn = FPN(num_filters=num_filters_fpn, norm_layer=norm_layer, pretrained=pretrained)

        # The segmentation heads on top of the FPN

        self.head1 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head2 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head3 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head4 = FPNHead(num_filters_fpn, num_filters, num_filters)

        self.smooth = nn.Sequential(
            GhostModule(4 * num_filters, num_filters, kernel_size=3, bias=True),
            # nn.Conv2d(4*num_filters, num_filters,padding=1, kernel_size=3),
            norm_layer(num_filters),
            nn.ReLU(),
        )

        self.smooth2 = nn.Sequential(
            GhostModule(num_filters, num_filters // 2, kernel_size=3, bias=True),
            # nn.Conv2d(num_filters, num_filters //2,padding=1, kernel_size= 3),
            norm_layer(num_filters // 2),
            nn.ReLU(),
        )
        # self.pixel_shuffle= nn.PixelShuffle(2)

        self.final = GhostModule(num_filters // 2, output_ch, kernel_size=3,
                                 bias=True)  # nn.Conv2d(num_filters // 2, output_ch,padding=1, kernel_size= 3)

    def unfreeze(self):
        self.fpn.unfreeze()

    def forward(self, x):
        #print(x.shape)
        map0, map1, map2, map3, map4 = self.fpn(x)

        map4 = nn.functional.interpolate(self.head4(map4), scale_factor=4, mode="nearest")
        map3 = nn.functional.interpolate(self.head3(map3), scale_factor=4, mode="nearest")
        map2 = nn.functional.interpolate(self.head2(map2), scale_factor=2, mode="nearest")
        map1 = self.head1(map1) #nn.functional.interpolate(self.head1(map1), scale_factor=1, mode="nearest")

        """for i in range(1,5):
            eval(f'print(map{i}.shape)')"""

        smoothed = self.smooth(torch.cat([map4, map3, map2, map1], dim=1))
        smoothed = nn.functional.interpolate(smoothed, scale_factor=4, mode="nearest")
        #print(smoothed.shape)
        #print(map0.shape)
        smoothed = self.smooth2(smoothed + map0)
        smoothed = nn.functional.interpolate(smoothed, scale_factor=2, mode="nearest") #self.pixel_shuffle(smoothed)#

        final = self.final(smoothed)
        res = torch.tanh(final) + x

        return torch.clamp(res, min=-1, max=1)

class FPN(nn.Module):

    def __init__(self, norm_layer, num_filters= 128, pretrained=True):
        """Creates an `FPN` instance for feature extraction.
        Args:
          num_filters: the number of filters in each output pyramid level
          pretrained: use ImageNet pre-trained backbone feature extractor
        """
        print(f'num filters are: {num_filters}')
        super(FPN, self).__init__()
        model = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=pretrained)
        indices = ["blocks[3][0].se.avg_pool", "blocks[4][0].se.avg_pool", "blocks[6][3].se.avg_pool",
                   "blocks[6][4].se.avg_pool", "blocks[7][0].se.avg_pool", "blocks[8][1].se.avg_pool",
                   "blocks[8][3].se.avg_pool"]
        for idx in indices:
            exec(f"model.{idx}= nn.Sequential()")

        #model.train()
        self.features= model.blocks
        """model.blocks has 10 elements"""
        #print(len(self.blocks))
        self.enc0 = nn.Sequential(model.conv_stem, model.bn1, model.act1) # nn.Sequential(*self.features[0:2])
        self.enc1 = nn.Sequential(*self.features[0:4])
        self.enc2 = nn.Sequential(*self.features[4:7])
        self.enc3 = nn.Sequential(*self.features[7:10])
        self.enc4 = nn.Sequential(nn.ReLU())

        self.td1 = nn.Sequential(GhostModule(num_filters, num_filters, kernel_size=3, bias=True),
                                 # nn.Conv2d(num_filters, num_filters, kernel_size= 3,padding=1, bias= True),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.td2 = nn.Sequential(GhostModule(num_filters, num_filters, kernel_size=3, bias=True),
                                 # nn.Conv2d(num_filters, num_filters, kernel_size= 3, padding=1,bias= True),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.td3 = nn.Sequential(GhostModule(num_filters, num_filters, kernel_size=3, bias=True),
                                 # nn.Conv2d(num_filters, num_filters, kernel_size= 3, padding=1,bias= True),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.pad = nn.ReflectionPad2d(1)

        self.lateral4 = nn.Conv2d(960, num_filters, kernel_size=1, bias=False) #GhostModule(960, num_filters, kernel_size=1, bias= False)#
        self.lateral3 = nn.Conv2d(960, num_filters, kernel_size=1, bias=False) #GhostModule(960, num_filters, kernel_size=1, bias= False)#
        self.lateral2 = nn.Conv2d(112, num_filters, kernel_size=1, bias=False) #GhostModule(112, num_filters, kernel_size=1, bias= False)#
        self.lateral1 = nn.Conv2d(40, num_filters, kernel_size=1, bias=False) #GhostModule(40, num_filters, kernel_size=1, bias= False)#
        self.lateral0 = nn.Conv2d(16, num_filters // 2, kernel_size=1, bias=False) #GhostModule(16, num_filters//2, kernel_size=1, bias= False)#

        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, x):

        # Bottom-up pathway, from ResNet
        #print(x.shape)
        enc0 = self.enc0(x)

        enc1 = self.enc1(enc0)  # 256

        enc2 = self.enc2(enc1)  # 512

        enc3 = self.enc3(enc2)  # 1024

        enc4 = self.enc4(enc3)  # 2048
        """for i in range(5):
            eval(f'print(enc{i}.shape)')"""


        # Lateral connections

        lateral4 = self.lateral4(enc4)
        lateral3 = self.lateral3(enc3)
        lateral2 = self.lateral2(enc2)
        lateral1 = self.lateral1(enc1)
        lateral0 = self.lateral0(enc0)

        # Top-down pathway
        map4 = lateral4

        map3 = self.td1(lateral3 + map4)
        map2 = self.td2(lateral2 + nn.functional.interpolate(map3, scale_factor=2, mode="nearest"))
        """for i in range(2,5):
            print(f'map{i}')
            eval(f'print(map{i}.shape)')"""
        map1 = self.td3(lateral1 + nn.functional.interpolate(map2, scale_factor=2, mode="nearest"))
        return lateral0, map1, map2, map3, map4





if __name__== "__main__":

    import time

    #x= x.double()
    #TIME_I= time.time()
    #t_run, flops= main(arg, time)
    #TIME_F= time.time()
    #return TIME_F-TIME_I
    x= torch.rand([1,3,736, 1312 ])
    model =  FPNGhostNetv3(HINet)
    TIME_I= time.time()

    output= model(x)
    t_run= time.time()- TIME_I
    print(f'time taken: {t_run}')


    inp= x.squeeze().detach().numpy().transpose([2,1,0])
    outp= output.squeeze().detach().numpy().transpose([2,1,0])

    #cv2.imshow("input", inp)
    #cv2.imshow("output", outp)
    #cv2.waitKey(0)
    from torchstat import stat

    stat(model, (3, 736, 1312))
    """macs, params = profile(model, inputs=(inputs_,),
                           custom_ops={GhostModule: count_ghostmodule})
    macs, params = clever_format([macs, params], "%.3f")
    print(f'macs: {macs}\n params:{params}')"""
    #count_ops(model, arg)
    #output= model(arg)
    #output= output.squeeze().detach().numpy().transpose([2,1,0])
    #cv2.imshow("img",output)
    #cv2.waitKey(0)
    #time, flops= main(arg, time)
    #macs, params = profile(model, inputs=(arg,))
    #macs, params = clever_format([macs, params], "%.3f")
    #print(f'MACS of ghostnet v2: {macs}')
    #print(f'PARAMS ghostnetv2: {params}')
    """results= {}
    device= torch.device("cuda")
    model_name= "ghostnetv3"
    model.to(device)
    with torch.no_grad():
        tmp = []
        tmp2 = []
        for img in tqdm.tqdm(inputs_):
            inputs = img.to(device)
            res = flop_count(model, (inputs,))
            t = measure_time(model, inputs)
            tmp.append(sum(res.values()))
            tmp2.append(t)

    results[model_name] = {'flops': fmt_res(np.array(tmp)), 'time': fmt_res(np.array(tmp2))}
    print('=============================')
    print('')
    for r in results:
        print(r)
        for k, v in results[r].items():
            print(' ', k, ':', v)"""






