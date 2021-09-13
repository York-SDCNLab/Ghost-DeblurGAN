import torch.nn as nn

class B(nn.Module):
     
     def __init__(self):
             super(B, self).__init__()
             a=  nn.Sequential(nn.Conv2d(3,3,3),nn.Conv2d(3,3,3),nn.Conv2d(3,3,3))
             b= nn.ReLU()
             c= nn.Conv2d(3,3,3)
             self.a=a
             self.b= b
             self.c= c
     def forward(self,x):
             x= self.a(x)
             x= self.b(x)
             x= self.c(x)
             return x

