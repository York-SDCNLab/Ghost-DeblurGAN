import argparse
import os
import time
import tqdm
import subprocess
import fpn_ghostnet_v3 as gnv3
import fpn_ghostnet as gn
import torch
import cv2
import numpy as np

def get_args():

    parser= argparse.ArgumentParser()
    parser.add_argument("--weight_path", type= str, default= "/home/vision/ghostnet/deblurgan/best_fpn.h5", help="path to weights")

    return parser.parse_args()

def main(args):

    model= gnv3.FPNGhostNetv3(gnv3.HINet)
    input= torch.rand([1,3, 256,256])
    input_n= input.squeeze().detach().numpy().transpose([2,1,0])
    output= model(input)
    output_n= output.squeeze().detach().numpy().transpose([2,1,0])
    state_dict= torch.load(args.weight_path)['model']
    model_state_dict= model.state_dict()

    #print(type(model_state_dict))
    for key_m in model_state_dict:
        #print(f'laoded key:{key}')
        #print(f'model key:{key_m}')
        key_mod= "module."+key_m
        if key_mod in state_dict:
            #print("key changed")
            model_state_dict[key_m]= state_dict[key_mod]

    model.load_state_dict(model_state_dict)
    output_mod= model(input)
    output_mod_n= output_mod.squeeze().detach().numpy().transpose([2,1,0])
    #print(np.sum(output_mod_n==output_n))
    #print(output_n.shape)
    print("success")
    cv2.imshow("original", output_n)
    cv2.imshow("modified", output_mod_n)
    cv2.waitKey(0)




if __name__=="__main__":

    args= get_args()
    main(args)



