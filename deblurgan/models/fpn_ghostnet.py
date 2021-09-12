import torch
import torch.nn as nn
#import logging
#from mobilenet_v2 import MobileNetV2
#from pretrainedmodels import inceptionresnetv2
#import fpn_mobilenet as mn
import numpy as np
import cv2
from pthflops import count_ops

class FPNHead(nn.Module):

    """"this is the FPNHead class common to all backbones"""
    def __init__(self, num_in, num_mid, num_out):
        super(FPNHead, self).__init__()

        self.block0 = nn.Conv2d(num_in, num_mid, kernel_size=3, padding=1, bias=False)
        self.block1 = nn.Conv2d(num_mid, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = nn.functional.relu(self.block0(x), inplace=True)
        x = nn.functional.relu(self.block1(x), inplace=True)
        return x

class FPNGhostNet(nn.Module):

    def __init__(self, norm_layer, output_ch=3, num_filters=64, num_filters_fpn=128, pretrained=True):
        super(FPNGhostNet, self).__init__()
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
            nn.Conv2d(4 * num_filters, num_filters, kernel_size=3, padding=1),
            norm_layer(num_filters),
            nn.ReLU(),
        )

        self.smooth2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters // 2, kernel_size=3, padding=1),
            norm_layer(num_filters // 2),
            nn.ReLU(),
        )

        self.final = nn.Conv2d(num_filters // 2, output_ch, kernel_size=3, padding=1)

    def unfreeze(self):
        self.fpn.unfreeze()

    def forward(self, x):
        map0, map1, map2, map3, map4 = self.fpn(x)

        map4 = nn.functional.upsample(self.head4(map4), scale_factor=8, mode="nearest")
        map3 = nn.functional.upsample(self.head3(map3), scale_factor=4, mode="nearest")
        map2 = nn.functional.upsample(self.head2(map2), scale_factor=2, mode="nearest")
        map1 = nn.functional.upsample(self.head1(map1), scale_factor=1, mode="nearest")

        smoothed = self.smooth(torch.cat([map4, map3, map2, map1], dim=1))
        smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode="nearest")
        smoothed = self.smooth2(smoothed + map0)
        smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode="nearest")

        final = self.final(smoothed)
        res = torch.tanh(final) + x

        return torch.clamp(res, min=-1, max=1)

class FPN(nn.Module):

    def __init__(self, norm_layer, num_filters=128, pretrained=True):
        """Creates an `FPN` instance for feature extraction.
        Args:
          num_filters: the number of filters in each output pyramid level
          pretrained: use ImageNet pre-trained backbone feature extractor
        """

        super(FPN, self).__init__()
        model = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=pretrained)
        self.features= model.blocks
        """model.blocks has 10 elements"""
        #print(len(self.blocks))
        self.enc0 = model.conv_stem # nn.Sequential(*self.features[0:2])
        self.enc1 = nn.Sequential(model.bn1, model.act1, *self.features[0])
        self.enc2 = nn.Sequential(*self.features[1:4])
        self.enc3 = nn.Sequential(*self.features[4:7])
        self.enc4 = nn.Sequential(*self.features[7:10])

        self.td1 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.td2 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.td3 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))

        self.lateral4 = nn.Conv2d(960, num_filters, kernel_size=1, bias=False)
        self.lateral3 = nn.Conv2d(112, num_filters, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(40, num_filters, kernel_size=1, bias=False)
        self.lateral1 = nn.Conv2d(16, num_filters, kernel_size=1, bias=False)
        self.lateral0 = nn.Conv2d(16, num_filters // 2, kernel_size=1, bias=False)

        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, x):

        # Bottom-up pathway, from ResNet
        enc0 = self.enc0(x)

        enc1 = self.enc1(enc0)  # 256

        enc2 = self.enc2(enc1)  # 512

        enc3 = self.enc3(enc2)  # 1024

        enc4 = self.enc4(enc3)  # 2048

        # Lateral connections

        lateral4 = self.lateral4(enc4)
        lateral3 = self.lateral3(enc3)
        lateral2 = self.lateral2(enc2)
        lateral1 = self.lateral1(enc1)
        lateral0 = self.lateral0(enc0)

        """for i in range(5):

            print(f'lateral{i}:')
            eval(f'print(lateral{i}.shape)')"""
        # Top-down pathway
        map4 = lateral4
        map3 = self.td1(lateral3 + nn.functional.upsample(map4, scale_factor=2, mode="nearest"))
        map2 = self.td2(lateral2 + nn.functional.upsample(map3, scale_factor=2, mode="nearest"))
        """for i in range(2,5):
            print(f'map{i}')
            eval(f'print(map{i}.shape)')"""
       # print(f'shape of lateral1:{lateral1.shape}')

        map1 =self.td3(lateral1 + nn.functional.upsample(map2, scale_factor=4, mode="nearest"))
        map1= nn.functional.upsample(map1, scale_factor= 0.5, mode= "nearest")
        return lateral0, map1, map2, map3, map4


"""
def load_ghostnet():

    model = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=True)
    model.eval()
    return model

def load_mobilenet():

    net = MobileNetV2(n_class=1000)
    return net.eval()

def load_inception():

    model= inceptionresnetv2(num_classes=1000, pretrained='imagenet')
    return model.eval()

if __name__=="__main__":

    model= load_mobilenet()
    c_i= 0
    logging.basicConfig(filename='model.log', filemode='w', level= logging.DEBUG)
    logging.info(model)
    for m in model.modules():

        c_i+= 1
        #print(m)
    #print(model.blocks)
    print(f'total modules: {c_i}')
"""
"""input_= torch.unsqueeze(torch.rand([3,256,256]),0)
fpn_= FPN(nn.BatchNorm2d)
output_= fpn_(input_)
for i in range(5):
    eval(f'print(output_[{i}].shape)') 
fpn= FPNGhostNet(nn.BatchNorm2d)
output_=fpn(input_)
#print(output_.shape)
output_= torch.squeeze(output_)
output_= output_.detach().numpy()
#print(output_.shape)
output_= np.transpose(output_,[2,1,0])
#print(output_.shape)
cv2.imshow("img",output_)
cv2.imshow("input", np.transpose(input_.squeeze().detach().numpy(),[2,1,0]))
cv2.waitKey(0)"""
def main(input_, time):
    #input_=torch.rand([1,3, 256, 256])
    #fpn_ = FPN(nn.BatchNorm2d)
    #output_ = fpn_(input_)
    """for i in range(5):
        eval(f'print(output_[{i}].shape)') """
    #input_= torch.Tensor(input_)
    fpn = FPNGhostNet(nn.BatchNorm2d)
    #fpn.cuda()
    #input_.cuda()
    TIME_I= time.time()
    output_ = fpn(input_)
    TIME_F= time.time()
    return TIME_F-TIME_I
    #print(output_.shape)
    #output_ = torch.squeeze(output_)
    #output_ = output_.detach().numpy()
    #print(output_.shape)
    #output_ = np.transpose(output_, [0, 3, 2, 1])
    #print(output_.shape)

    #cv2.imshow("img", output_[0])
    #cv2.imshow("input", np.transpose(input_[0].squeeze().detach().numpy(), [2, 1, 0]))
    #cv2.waitKey(5000)
    #cv2.imshow("img", output_)
    #cv2.imshow("input", np.transpose(input_.squeeze().detach().numpy(), [2, 1, 0]))
    #cv2.waitKey(0)

if __name__== "__main__":

    import time
    arg = torch.rand([2, 3, 256, 256])
    #TIME_I= time.time()
    #t_run, flops= main(arg, time)
    #TIME_F= time.time()
    #return TIME_F-TIME_I
    model =  FPNGhostNet(nn.InstanceNorm2d)
    count_ops(model, arg)
    #time, flops= main(arg, time)
    #macs, params = profile(model, inputs=(arg,))
    #macs, params = clever_format([macs, params], "%.3f")
    #print(f'MACS of ghostnet v2: {macs}')
    #print(f'PARAMS ghostnetv2: {params}')


