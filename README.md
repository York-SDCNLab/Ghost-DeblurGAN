# Ghost-DeblurGAN

This is the repository of our IROS 2022 paper  [Application of Ghost-DeblurGAN to Fiducial Marker Detection](https://ieeexplore.ieee.org/document/9981701) <br>
An introduction video is available at https://www.youtube.com/watch?v=uYHIDIJQ0r8 <br>

Feature extraction or localization based on the fiducial marker could fail due to motion blur in real-world robotic applications. To solve this problem, a lightweight generative  adversarial network, named Ghost-DeblurGAN, for real-time motion deblurring is developed. Furthermore, on account that there is no existing deblurring benchmark for such task, a new large-scale dataset, YorkTag, is proposed that provides pairs of sharp/blurred images containing fiducial markers. With the proposed model trained and tested on YorkTag, it is demonstrated that when applied along with fiducial marker systems to motion-blurred images, Ghost-DeblurGAN improves the marker detection significantly.

The implementation is modified from https://github.com/VITA-Group/DeblurGANv2.<br> 

# Visual Comparison
Visual comparison of marker detection with and without Ghost-DeblurGAN in robotic applications. <br>
A video captured by a downwards camera onboard a maneuvering UAV ([Qdrone](https://www.quanser.com/products/qdrone/), from the Quanser Inc. )<br>
<img src="https://user-images.githubusercontent.com/58899542/154817276-ac136431-a69b-4630-af15-5496cb7124d1.gif" width="800"> <br>
A video captured by a low-cost CSI camera onboard a moving UGV ([Qcar](https://www.quanser.com/products/qcar/), from the Quanser Inc.) <br>
<img src="https://user-images.githubusercontent.com/58899542/154817295-22e733a5-5f33-439d-a29e-08f5950a8784.gif" width="800"> <br>



# Why it is necessary to propose a new dataset?

Current deblurring benchmarks only contain routine scenes including pedestrians, cars, buildings, and human faces, etc. To illustrate the necessity of proposing a new deblurring benchmark containing fiducial markers, we test [HINet](https://github.com/megvii-model/HINet) which has the SOTA performance on GoPro dataset with a blurred image and apply the Apriltag Detector to the deblurred image (See Fig.1(d)). As shown in the figure, due to the fact that HINet is trained on GoPro dataset which contains no fiducial markers, the marker detection rate is far from satisfying. Again, note that HINet is the SOTA method.

<img src="https://user-images.githubusercontent.com/58899542/132930466-46acdd1d-fed4-4c69-9506-4dc84107bbaa.png" width="600">


To end this, we propose a new large-scale dataset, **YorkTag**, that provides paired blurred and sharp images containing AprilTags and ArUcos. For the sake of obtaining ideal sharp images, we employ the iPhone 12 with the DJI OM 4 stabilizer to capture high-resolution videos. Detailed introduction of the blurred and sharp image pairs generation is available in our paper. Our training set consists of 1577 image pairs, and test set consists of 497 image pairs totalling 2074 blurry-sharp image pairs. We will keep augmenting the yorktag dataset later on.   
Link to the YorkTag dataset utilized in our paper: https://drive.google.com/file/d/1S3wVptR_mzrntuCtEarkXHE6d1zN6jd3/view?usp=sharing
 
<img src="https://user-images.githubusercontent.com/58899542/132930869-a66fb452-9579-4922-980a-94bc5e067ae9.jpeg" width="900">


# Training
## Command
```python train.py``` A video tutorial is available at: https://www.youtube.com/watch?v=JSCA2x3NBHs <br>
By default training script will load conifguration from config/config.yaml
files_a parameter represents blurry images and files_b represents sharp images
modify config.yaml file to change the generator model.
Available model scripts are:
- Ghostnet + Half Instance Normalization (HIN) + Ghost module (GM)
- MobilenetV2


# Testing and Inference
For single image inference,
```python predict.py /path/to/image.png --weights_path=/path/to/weights``` <br>
by default output is written under submit directory

Note: 'model' parameters in config.yaml must correspond to the weights <br>
For testing on single image,<br>
```python test_metrics.py --img_folder=/path/to/image.png --weights_path=/path/to/weights --new_gopro``` <br>
For testing on the dataset utilized in this work,<br>
```python test_metrics.py --img_folder=/base/directory/of/GOPRO/test/blur --weights_path=/path/to/weights --new_gopro ```


# Pre-trained models
For fair comparison we used the same mobilenet model as the original [DeblurGANv2](https://github.com/VITA-Group/DeblurGANv2) and 
trained all models from **scratch** on the [GOPRO dataset](https://drive.google.com/file/d/1KStHiZn5TNm2mo3OLZLjnRvd0vVFCI0W/view).
The metrics in the above table are to illustrate the superiority of Ghost-DeblurGAN over the original deblurGAN-v2 (mobilenetV2). Note that to obtain the deblurring performance shown in the visual comparison, the weights trained on the mix of YorkTag and GoPro should be adopted. These weights are coming soon.
<table align="center">
    <tr>
        <th>Dataset</th>
        <th>Model</th>
        <th>FLOPs</th>
        <th>PSNR/ SSIM</th>
        <th>Link</th>
    </tr>
    <tr>
        <td rowspan="2">GoPro Test Dataset</td>     
        <td>DeblurGAN-v2 (MobileNetV2)</td>
        <td>43.75G</td>
        <td>28.40/ 0.917</td>
        <td><a href="./trained_weights/fpn_ghostnet_gm_hin.h5">fpn_mobilnet_v2.h5</a></td>        
    </tr>
    <tr>
        <td>Ghost-DeblurGAN (Ours)</td>
        <td>20.51G</td>
        <td>28.79/ 0.920</td>
        <td><a href="./trained_weights/fpn_ghostnet_gm_hin.h5">fpn_ghostnet_gm_hin.h5</a></td>
    </tr>
   
</table>

# Citation
If you find this work helpful for your research, please cite our paper:
```
@INPROCEEDINGS{9981701,
  author={Liu, Yibo and Haridevan, Amaldev and Schofield, Hunter and Shan, Jinjun},
  booktitle={2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Application of Ghost-DeblurGAN to Fiducial Marker Detection}, 
  year={2022},
  volume={},
  number={},
  pages={6827-6832},
  doi={10.1109/IROS47612.2022.9981701}}
```
