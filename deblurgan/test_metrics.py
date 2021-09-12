from __future__ import print_function
import argparse
import numpy as np
import torch
import cv2
import yaml
import os
from torchvision import models, transforms
from torch.autograd import Variable
import shutil
import glob
import tqdm
from util.metrics import PSNR
from albumentations import Compose, CenterCrop, PadIfNeeded
from PIL import Image
from ssim.ssimlib import SSIM 
from models.networks import get_generator
from functools import partial

torch.cuda.set_device(0)

def get_args():
	parser = argparse.ArgumentParser('Test an image')
	parser.add_argument('--img_folder', required=True, help='GoPRO Folder')
	parser.add_argument('--weights_path', required=True, help='Weights path')
	parser.add_argument("--new_gopro",  action= "store_true", help= "whether to use new go pro dir structure or old go pro dir structure, by default assumes old go pro dir structure, specifying this option will revert")

	return parser.parse_args()


def prepare_dirs(path):
	if os.path.exists(path):
		shutil.rmtree(path)
	os.makedirs(path)


def get_gt_image(path, new_gopro= False):
	
	dir, filename = os.path.split(path)
	base, seq = os.path.split(dir)
	base, _ = os.path.split(base) if new_gopro else (base, None)
	img = cv2.cvtColor(cv2.imread(os.path.join(base, 'sharp', seq, filename )if new_gopro else os.path.join(base, 'sharp', filename )  ), cv2.COLOR_BGR2RGB)
	return img


def test_image(model, image_path, new_gopro= False):
	img_transforms = transforms.Compose([
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
	])
	size_transform = Compose([
		PadIfNeeded(736, 1280)
	])
	crop = CenterCrop(720, 1280)
	img = cv2.imread(image_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img_s = size_transform(image=img)['image']
	img_tensor = torch.from_numpy(np.transpose(img_s / 255, (2, 0, 1)).astype('float32'))
	img_tensor = img_transforms(img_tensor)
	with torch.no_grad():
		img_tensor = Variable(img_tensor.unsqueeze(0).cuda())
		result_image = model(img_tensor)
	result_image = result_image[0].cpu().float().numpy()
	result_image = (np.transpose(result_image, (1, 2, 0)) + 1) / 2.0 * 255.0
	result_image = crop(image=result_image)['image']
	result_image = result_image.astype('uint8')
	gt_image = get_gt_image(image_path, new_gopro)
	_, filename = os.path.split(image_path)
	psnr = PSNR(result_image, gt_image)
	pilFake = Image.fromarray(result_image)
	pilReal = Image.fromarray(gt_image)
	ssim = SSIM(pilFake).cw_ssim_value(pilReal)
	return psnr, ssim

def read_imgs(images):
	
	
	images_= []
	for img_ in images:
		img = cv2.imread(img_)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img_s = size_transform(image=img)['image']
		img_tensor = torch.from_numpy(np.transpose(img_s / 255, (2, 0, 1)).astype('float32'))
		img_tensor = img_transforms(img_tensor)
		images_.append(img_tensor)
	images_= np.array(images_)
	images_= torch.from_numpy(images)
	
	with torch.no_grad():
		
		images_= Variable(images.cuda())
	return images_
	
def test_image_batch(model, images, new_gopro= False):
	img_transforms = transforms.Compose([
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
	])
	size_transform = Compose([
		PadIfNeeded(736, 1280)
	])
	images_= read_imgs(images)
	results_= model(images_)
	results_= results_.cpu().float().numpy()
	results_= (np.transpose(results_, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
	crop = CenterCrop(720, 1280)
	results_= crop(image=results_)['image']
	results_= results.astype('uint8')
	get_gt_img= partial(get_gt_img, new_gopro=new_gopro)
	get_gt_img_v= np.vectorize(get_gt_img)
	gt_images= get_gt_img_v(results_)
	
	

def test(model, files, new_gopro= False):
	psnr = 0
	ssim = 0
	for file in tqdm.tqdm(files):
		cur_psnr, cur_ssim = test_image(model, file, new_gopro)
		psnr += cur_psnr
		ssim += cur_ssim
	print("PSNR = {}".format(psnr / len(files)))
	print("SSIM = {}".format(ssim / len(files)))
	return psnr,ssim


if __name__ == '__main__':
	args = get_args()
	print(args)
	with open('config/config.yaml') as cfg:
		config = yaml.load(cfg)
	model = get_generator(config['model'])
	model.load_state_dict(torch.load(args.weights_path, map_location="cuda:0")['model'])
	model = model.cuda()
	filenames = sorted(glob.glob(args.img_folder  + '/**/*.png' if os.path.isdir(args.img_folder) else args.img_folder, recursive=True)) if args.new_gopro else sorted(glob.glob(args.img_folder  + '/**/blur/*.png' if os.path.isdir(args.img_folder) else args.img_folder, recursive=True))

	_,__=test(model, filenames, args.new_gopro)
