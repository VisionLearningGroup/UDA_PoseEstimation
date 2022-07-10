# UDA_PoseEstimation
Code for [A Unified Framework for Domain Adaptive Pose Estimation](https://arxiv.org/pdf/2204.00172.pdf), accepted at ECCV 2022

# Introduction

Our Unsupervised Recycle-GAN seeks to improve the quality of translation in the unpaired video-to-video translation task and address the temporal and semantic inconsistency issues leveraging pseudo-supervision from synthetic optical flow.

![Overall](figs/fig1.png)

Our Unsupervised Recycle-GAN includes a branch of unsupervised recycle loss and a branch of unsupervised spatial loss that provide more accurate and efficient spatiotemporal consistency constraints during the adversarial training. Built on the basis of an unpaired image-to-image translation framework (Cycle-GAN in this example), our method can effectively improve the temproal and semantic consistency in the resulting videos. 

![Losses](figs/fig2.png)

# Usage

**Data Preparation**

Viper dataset is available via [Recycle-GAN](https://github.com/aayushbansal/Recycle-GAN/), and CityScapes sequence dataset (leftImg8bit_sequence_trainvaltest) is available [Here](https://www.cityscapes-dataset.com/downloads/). 

Organize the dataset in such a way that it contains train/val set and source domain A/ target domain B hierarchically. For Viper-to-CityScapes experiments, A/B will be the frames from Viper/CityScapes while for Video-to-Label experiments, A/B will be the frames/label maps in Viper. 
```
path/to/data/
|-- train
|   |-- A
|   `-- B
`-- val
    |-- A
    `-- B
```

**Viper-to-CityScapes Experiment**
```
python train.py --dataroot path/to/data/ --model unsup_single --dataset_mode unaligned_scale --name v2c_experiment --loadSizeW 542 --loadSizeH 286 --resize_mode rectangle --fineSizeW 512 --fineSizeH 256 --crop_mode rectangle --which_model_netG resnet_6blocks --no_dropout --pool_size 0 --lambda_spa_unsup_A 10 --lambda_spa_unsup_B 10 --lambda_unsup_cycle_A 10 --lambda_unsup_cycle_B 10 --lambda_cycle_A 0 --lambda_cycle_B 0 --lambda_content_A 1 --lambda_content_B 1 --batchSize 1 --noise_level 0.001  --niter_decay 0 --niter 2
python test.py --dataroot path/to/data/ --model unsup_single --dataset_mode unaligned_scale --name v2c_experiment --loadSizeW 512 --loadSizeH 256 --resize_mode rectangle --fineSizeW 512 --fineSizeH 256 --crop_mode none --which_model_netG resnet_6blocks --no_dropout --which_epoch 2
```

**Video-to-Label Experiment**

```
python train.py --dataroot path/to/data/ --model unsup_single --dataset_mode unaligned_scale --name v2l_experiment --loadSizeW 286 --loadSizeH 286 --resize_mode rectangle --fineSizeW 256 --fineSizeH 256 --crop_mode rectangle --which_model_netG resnet_6blocks --no_dropout --pool_size 0 --lambda_spa_unsup_A 10 --lambda_spa_unsup_B 0 --lambda_unsup_cycle_A 0 --lambda_unsup_cycle_B 10 --lambda_cycle_A 10 --lambda_cycle_B 10 --lambda_content_A 0 --lambda_content_B 0 --batchSize 1 --noise_level 0.001  --niter_decay 0 --niter 5
python test.py --dataroot path/to/data/ --model unsup_single --dataset_mode unaligned_scale --name v2l_experiment --loadSizeW 256 --loadSizeH 256 --resize_mode rectangle --fineSizeW 256 --fineSizeH 256 --crop_mode none --which_model_netG resnet_6blocks --no_dropout --which_epoch 5
```
   
**Pretrained Models**

Pretrained models in both experiments are available [here](https://drive.google.com/drive/folders/1WVjQ2Hq1CrtUr3AOlq1PxDuY50KpZ8lh?usp=sharing). Please make sure they are downloaded and saved in the "saved_models" folder under this directory.
             
**Acknowledgment**

Code borrowed from [RegDA](https://github.com/thuml/Transfer-Learning-Library/tree/master/examples/domain_adaptation/keypoint_detection), [UDA-Aniaml](https://github.com/chaneyddtt/UDA-Animal-Pose).
