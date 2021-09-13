import math
import torch
import torch.nn as nn
import timm
import numpy as np
import cv2


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
            
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=bias),
            
        )

    def forward(self, x):
        
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]



class FPNHead(nn.Module):

    """"this is the FPNHead class common to all backbones"""
    def __init__(self, num_in, num_mid, num_out):
        super(FPNHead, self).__init__()

        self.block0 = GhostModule(num_in, num_mid, kernel_size=3, bias= False) 
        self.block1 = GhostModule(num_mid, num_out, kernel_size=3, bias= False)

    def forward(self, x):
        x = nn.functional.relu(self.block0(x), inplace=True)
        x = nn.functional.relu(self.block1(x), inplace=True)
        return x

class FPNGhostNet(nn.Module):

    def __init__(self, norm_layer, output_ch=3, num_filters= 64, num_filters_fpn= 128, pretrained=True):
        super(FPNGhostNet, self).__init__()
        

        self.fpn = FPN(num_filters=num_filters_fpn, norm_layer=norm_layer, pretrained=pretrained)

        # The segmentation heads on top of the FPN

        self.head1 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head2 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head3 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head4 = FPNHead(num_filters_fpn, num_filters, num_filters)

        self.smooth = nn.Sequential(
            GhostModule(4*num_filters, num_filters, kernel_size=3, bias= True),
            norm_layer(num_filters),
            nn.ReLU(),
        )

        self.smooth2 = nn.Sequential(
            GhostModule(num_filters, num_filters//2, kernel_size=3, bias= True),
            norm_layer(num_filters // 2),
            nn.ReLU(),
        )
        

        self.final =  GhostModule(num_filters//2, output_ch, kernel_size=3, bias= True)

    def unfreeze(self):
        self.fpn.unfreeze()

    def forward(self, x):
        
        map0, map1, map2, map3, map4 = self.fpn(x)

        map4 = nn.functional.interpolate(self.head4(map4), scale_factor= 8, mode="nearest")
        map3 = nn.functional.interpolate(self.head3(map3), scale_factor=4, mode="nearest")
        map2 = nn.functional.interpolate(self.head2(map2), scale_factor=2, mode="nearest")
        map1 = nn.functional.interpolate(self.head1(map1), scale_factor=1, mode="nearest")



        

        smoothed = self.smooth(torch.cat([map4, map3, map2, map1], dim=1))
        smoothed = nn.functional.interpolate(smoothed, scale_factor=2, mode="nearest")
        smoothed = self.smooth2(smoothed + map0)
        smoothed = nn.functional.interpolate(smoothed, scale_factor=2, mode="nearest") 

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
        
        super(FPN, self).__init__()
        model = timm.create_model('ghostnet_100', pretrained= True, features_only= True)
        
        self.features= model
        
        self.enc0 = nn.Sequential(self.features.conv_stem, self.features.bn1, self.features.act1) 
        self.enc1 = nn.Sequential(self.features.blocks_0 , self.features.blocks_1)
        self.enc2 = nn.Sequential(self.features.blocks_2, self.features.blocks_3)
        self.enc3 = nn.Sequential(self.features.blocks_4, self.features.blocks_5)
        self.enc4 = nn.Sequential(self.features.blocks_6, self.features.blocks_7)

        self.td1 = nn.Sequential( GhostModule(num_filters, num_filters, kernel_size=3, bias= True),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.td2 = nn.Sequential(GhostModule(num_filters, num_filters, kernel_size=3, bias= True),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.td3 = nn.Sequential(GhostModule(num_filters, num_filters, kernel_size=3, bias= True),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.pad = nn.ReflectionPad2d(1)

        self.lateral4 = nn.Conv2d(160, num_filters, kernel_size=1, bias=False) 
        self.lateral3 = nn.Conv2d(80, num_filters, kernel_size=1, bias=False) 
        self.lateral2 = nn.Conv2d(40, num_filters, kernel_size=1, bias=False) 
        self.lateral1 = nn.Conv2d(24, num_filters, kernel_size=1, bias=False) 
        self.lateral0 = nn.Conv2d(16, num_filters // 2, kernel_size=1, bias=False)

        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, x):

        
        
        enc0 = self.enc0(x)

        enc1 = self.enc1(enc0)  

        enc2 = self.enc2(enc1)  

        enc3 = self.enc3(enc2)  

        enc4 = self.enc4(enc3)  


        # Lateral connections

        lateral4 = self.lateral4(enc4)
        lateral3 = self.lateral3(enc3)
        lateral2 = self.lateral2(enc2)
        lateral1 = self.lateral1(enc1)
        lateral0 = self.lateral0(enc0)

        # Top-down pathway
        map4 = lateral4

        map3 = self.td1(lateral3 + nn.functional.interpolate(lateral4, scale_factor= 2, mode= "nearest"))
        map2 = self.td2(lateral2 + nn.functional.interpolate(map3, scale_factor=2, mode="nearest"))
        map1 = self.td3(lateral1 + nn.functional.interpolate(map2, scale_factor=2, mode="nearest"))
        return lateral0, map1, map2, map3, map4





if __name__== "__main__":

    from torchstat import stat
    model= FPNGhostNet(nn.BatchNorm2d)
    stat(model, (3, 736, 1312))






