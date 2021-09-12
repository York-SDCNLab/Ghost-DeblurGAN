import fpn_ghostnet
import numpy as np
import cv2
from PIL import Image
import argparse
import os
import torchvision.transforms as transforms
import glob
import torch.nn as nn
import torch

def get_args():

    parser= argparse.ArgumentParser()
    parser.add_argument("--image", type= str, default= None, help= "path to image files or file")
    parser.add_argument("--ext", type= str, default= None, help= "extension of image files")
    parser.add_argument("--output_dir", type= str, default= "/home/vision/ghostnet/ghostnet_output", help= "output dir for predicted images")
    parser.add_argument("--show_image",  action= "store_true", default= False, help= "flag to show image")
    parser.add_argument("--wait_time", type= float,default= 0, help= "wait time" )
    parser.add_argument("--use_cuda", action= "store_true", default= False, help= "use cuda flag")
    parser.add_argument("--out_ext", type= str, default= "png", help= "extension to write output images")

    args= parser.parse_args()
    return args

def read_images(args, reader= "PIL"):

    if os.path.isdir(args.image):

        images= []
        files= glob.glob(args.image+f"/**/*.{args.ext}",recursive= True)
        for file in files:
            if reader=="cv2":
                images.append(cv2.imread(file, cv2.IMREAD_COLOR))
            else:
                images.append(Image.open(file))
        return images
    image= cv2.imread(args.image, cv2.IMREAD_COLOR) if reader=="cv2" else Image.open(args.image)
    return [image]

def get_files(args):

    if os.path.isdir(args.image):
        files = glob.glob(args.image + f"/**/*.{args.ext}", recursive=True)
        return files

    return [args.image]


def pre_process(args, n_channels= 3):

    if not os.path.isdir(args.output_dir):

        os.system(f'mkdir {args.output_dir}')
    if os.path.isdir(args.image):

        assert args.ext, "no extension provided"
        args.show_image= False
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    images= read_images(args)
    input = torch.zeros(len(images), n_channels, 224, 224)

    for i,image in enumerate(images):

        input[i,:,:,:]= preprocess(image)

    return input

def split_chunks(tensor, batch_size= 64):

    chunks= tensor.shape[0]/batch_size
    return torch.chunk(tensor, int(chunks), dim=0)

def post_process(output):

   # output_= torch.zeros(output.shape)
    transform= transforms.ToPILImage()
    return transform(output)

if __name__== "__main__":

    args= get_args()
    input= pre_process(args)
    #cv2.imshow("img", np.transpose(input.squeeze().detach().numpy(),[2,1,0]))
    #cv2.waitKey(2000)
    #assert None
    model= fpn_ghostnet.FPNGhostNet(nn.BatchNorm2d)
    if torch.cuda.is_available() and args.use_cuda:

        input= input.cuda()
        model= model.cuda()
    print(f'input shape is:{input.shape}')


    output= model(input).cpu()
    #cv2.imshow("img", np.transpose(output.cpu().squeeze().detach().numpy(),[2,1,0]))
    #cv2.waitKey(2000)
    #assert None
    #output= output.squeeze()
    #eval(f'output_{i}= model(chunks[{i}])' for i in range())
    #output= output.cpu().detach().numpy()
   # output= post_process(output)
    files= get_files(args)
    #print(files)
    print(f'output shape is:{output.shape}')

    for i, img in enumerate(output):

        #print(f'is is{i}')
        #print(img.shape)
        base, filename= os.path.split(files[i])
        file,ext= os.path.splitext(filename)
        #cv2.imwrite(args.output_dir+'/'+file+'.jpg', cv2.cvtColor(np.transpose(img,[2,1,0]), cv2.COLOR_RGB2BGR))
        img= post_process(img)
        img.save(args.output_dir+'/'+file+f'.{args.out_ext}')