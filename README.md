# Ghost-DeblurGAN
Motion blur can impede marker detection and marker-based pose estimation, which is common in real-world robotic applications involving fiducial markers. To solve this problem, we propose a novel lightweight generative adversarial network (GAN), Ghost-DeblurGAN, for real-time motion deblurring. Furthermore, a new large-scale dataset, YorkTag, provides pairs of sharp/blurred images containing fiducial markers and is proposed to train and qualitatively and quantitatively evaluate our model. Experimental results demonstrate that when applied along with fudicual marker systems to motion-blurred images, Ghost-DeblurGAN improves the marker detection significantly and mitigates the rotational ambiguity problem in marker-based pose estimation.   
This is an upgrade work based on DeblurGAN-v2 (https://github.com/VITA-Group/DeblurGANv2).  
Link to the introduction video: https://www.youtube.com/watch?v=90T88_M_l3Y  
Link to the paper: https://arxiv.org/abs/2109.03379

Visual comparison of marker detection with and without Ghost-DeblurGAN in robotic applications. (a):  A video captured by a downwards camera onboard a maneuvering UAV (Qdrone, from the Quanser Inc. https://www.quanser.com/products/qdrone/). (b): A video captured by a low-cost CSI camera onboard a moving UGV (Qcar, from the Quanser Inc. https://www.quanser.com/products/qcar/).
![2](https://user-images.githubusercontent.com/58899542/132931107-2761194b-2c94-4f87-a907-57773be92a4e.gif)
![4](https://user-images.githubusercontent.com/58899542/132931220-d1d661f4-b148-4467-9ba0-a859b440caed.gif)



# YorkTag Dataset

Current deblurring benchmarks only contain routine scenes including pedestrians, cars, buildings, and human faces, etc. To illustrate the necessity of proposing a new deblurring benchmark containing fiducial markers, we test HINet (https://github.com/megvii-model/HINet) which has the SOTA performance on GoPro dataset with a blurred image and apply the Apriltag Detector  to the deblurred image (See Fig.1(d)). As shown in the figure, due to the fact that HINet is trained on GoPro dataset which contains no fiducial markers, the marker detection rate is far from satisfying.
![github1](https://user-images.githubusercontent.com/58899542/132930466-46acdd1d-fed4-4c69-9506-4dc84107bbaa.png)


To end this, we propose a new large-scale dataset, YorkTag, that provides paired blurred and sharp images containing AprilTags and ArUcos. For the sake of obtaining ideal sharp images, we employ the iPhone 12 with the DJI OM 4 stabilizer to capture high-resolution videos. Detailed introduction of the blurred and sharp image pairs generation is available in our paper. Our training set consists of 1577 image pairs, and test set consists of 497 image pairs totalling 2074 blurry-sharp image pairs. We will keep augmenting the yorktag dataset later on.   
Link to the YorkTag dataset utilized in our paper: https://drive.google.com/file/d/1S3wVptR_mzrntuCtEarkXHE6d1zN6jd3/view?usp=sharing
![yorktag](https://user-images.githubusercontent.com/58899542/132930869-a66fb452-9579-4922-980a-94bc5e067ae9.jpeg)  
Link to the linear GoPro dataset utilized in this work:https://drive.google.com/file/d/1KStHiZn5TNm2mo3OLZLjnRvd0vVFCI0W/view
