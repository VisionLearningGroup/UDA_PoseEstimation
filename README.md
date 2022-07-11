# UDA_PoseEstimation
Code for [A Unified Framework for Domain Adaptive Pose Estimation](https://arxiv.org/pdf/2204.00172.pdf), accepted at ECCV 2022. 

<p align="center">
  <img width="650" src="figures/sample.png">
</p>

# Introduction

![Image of Source](https://github.com/VisionLearningGroup/UDA_PoseEstimation/blob/master/figures/pipeline.png)

We propose a unified freamwork for domain adaptive pose estimation on various objects including human body, human hand and animal that requires a (synthetic) labeled source domain dataset and a (real-world) target domain dataset **without** annotations. The system consists of a style trasnfer module to mitigate visual domain gap and a mean-teacher framework to encourage feature-level unsupervised learning from unlabeled images.
For further details regarding our method, please refer to our [paper](https://arxiv.org/pdf/2204.00172.pdf).

# Usage

**Data Preparation**

**Human Dataset**

As instructed by [RegDA](https://github.com/thuml/Transfer-Learning-Library/tree/master/examples/domain_adaptation/keypoint_detection), following datasets can be downloaded automatically:

- [Rendered Handpose Dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html)
- [Hand-3d-Studio Dataset](https://www.yangangwang.com/papers/ZHAO-H3S-2020-02.html)
- [FreiHAND Dataset](https://lmb.informatik.uni-freiburg.de/projects/freihand/)
- [Surreal Dataset](https://www.di.ens.fr/willow/research/surreal/data/)
- [LSP Dataset](http://sam.johnson.io/research/lsp.html)

You need to prepare following datasets manually if you want to use them:
- [Human3.6M Dataset](http://vision.imar.ro/human3.6m/description.php)

**Aniaml Dataset**

Following [UDA-Animal-Pose](https://github.com/chaneyddtt/UDA-Animal-Pose) and [CCSSL](https://github.com/JitengMu/Learning-from-Synthetic-Animals):
- Create a `./animal_data` directory.
- Download the synthetic dataset by running `bash get_dataset.sh`.
- Download the [TigDog](http://calvin-vision.net/datasets/tigdog/) dataset and move folder behaviorDiscovery2.0 to `./animal_data/`.
- Download a cleaned annotation file of the synthetic dataset for better time performance from [here](https://drive.google.com/file/d/1jpGD235mFsVixeVRpcqMzGcbXqUtOvAy/view?usp=sharing) and plac it under `./animal_data/`.
- Download the [cropped images](https://drive.google.com/file/d/1qFX_H2o8_unFpADowjTOcGfr_SwKkuYg/view?usp=sharing) for the TigDog dataset and move the folder real_animal_crop_v4 to `./animal_data/`.

**Pretrained Models**

Before training, please make sure style transfer models are downloaded and saved in the "saved_models" folder under this directory. Pretrained models in all experiments are available [here](https://drive.google.com/drive/folders/1WVjQ2Hq1CrtUr3AOlq1PxDuY50KpZ8lh?usp=sharing). 

# Experiments

**UDA Human Pose Estimation**

SURREAL-to-Human36M
```
python train_human.py path/to/SURREAL path/to/Human36M -s SURREAL -t Human36M --target-train Human36M_mt --log logs/s2h_exp/syn2real --debug --seed 0 --lambda_t 0 --lambda_c 1 --pretrain-epoch 40 --rotation_stu 60 --shear_stu -30 30 --translate_stu 0.05 0.05 --scale_stu 0.6 1.3 --color_stu 0.25 --blur_stu 0 --rotation_tea 60 --shear_tea -30 30 --translate_tea 0.05 0.05 --scale_tea 0.6 1.3 --color_tea 0.25 --blur_tea 0 -b 32 --mask-ratio 0.5 --k 1 --decoder-name saved_models/decoder_s2h_0_1.pth.tar --s2t-freq 0.5 --s2t-alpha 0 1 --t2s-freq 0.5 --t2s-alpha 0 1 --occlude-rate 0.5 --occlude-thresh 0.9 
```

SURREAL-to-LSP
```
python train_human.py path/to/SURREAL path/to/LSP -s SURREAL -t LSP --target-train LSP_mt --log logs/s2l_exp/syn2real --debug --seed 0 --lambda_t 0 --lambda_c 1 --pretrain-epoch 40 --rotation_stu 60 --shear_stu -30 30 --translate_stu 0.05 0.05 --scale_stu 0.6 1.3 --color_stu 0.25 --blur_stu 0 --rotation_tea 60 --shear_tea -30 30 --translate_tea 0.05 0.05 --scale_tea 0.6 1.3 --color_tea 0.25 --blur_tea 0 -b 32 --mask-ratio 0.5 --k 1 --decoder-name saved_models/decoder_s2l_0_1.pth.tar --s2t-freq 0.5 --s2t-alpha 0 1 --t2s-freq 0.5 --t2s-alpha 0 1 --occlude-rate 0.5 --occlude-thresh 0.9 
```

**UDA Hand Pose Estimation**

RHD-to-H3D
```
python train_human.py path/to/RHD path/to/H3D -s RenderedHandPose -t Hand3DStudio --target-train Hand3DStudio_mt --log logs/r2h_exp/syn2real --debug --seed 0 --lambda_t 0  --lambda_c 1 --pretrain-epoch 40  --rotation_stu 180 --shear_stu -30 30 --translate_stu 0.05 0.05 --scale_stu 0.6 1.3 --color_stu 0.25 --blur_stu 0 --rotation_tea 180 --shear_tea -30 30 --translate_tea 0.05 0.05 --scale_tea 0.6 1.3 --color_tea 0.25 --blur_tea 0 -b 32 --mask-ratio 0.5 --k 1 --decoder-name saved_models/decoder_r2h_0_1.pth.tar --s2t-freq 0.5 --s2t-alpha 0 1 --t2s-freq 0.5 --t2s-alpha 0 1 --occlude-rate 0.5 --occlude-thresh 0.9
```

FreiHand-to-H3D
```
python train_human.py path/to/FreiHand path/to/RHD -s FreiHand -t RenderedHandPose --target-train RenderedHandPose_mt --log logs/f2r_exp/syn2real --debug --seed 0 --lambda_t 0  --lambda_c 1 --pretrain-epoch 40  --rotation_stu 180 --shear_stu -30 30 --translate_stu 0.05 0.05 --scale_stu 0.6 1.3 --color_stu 0.25 --blur_stu 0 --rotation_tea 180 --shear_tea -30 30 --translate_tea 0.05 0.05 --scale_tea 0.6 1.3 --color_tea 0.25 --blur_tea 0 -b 32 --mask-ratio 0.5 --k 1 --decoder-name saved_models/decoder_f2r_0_1.pth.tar --s2t-freq 0.5 --s2t-alpha 0 1 --t2s-freq 0.5 --t2s-alpha 0 1 --occlude-rate 0.5 --occlude-thresh 0.9
```

**UDA Animal Pose Estimation**

SyntheticAnimal-to-TigDog
```
python train_animal.py path/to/animal_pose_dataset  --source synthetic_animal_sp_all --target real_animal_all --target_ssl real_animal_all_mt --train_on_all_cat --log logs/syn2real_animal/syn2real --debug --seed 0 --lambda_c 1 --pretrain-epoch 40 --rotation_stu 60 --shear_stu -30 30 --translate_stu 0.05 0.05 --scale_stu 0.6 1.3 --color_stu 0.25 --blur_stu 0 --rotation_tea 60 --shear_tea -30 30 --translate_tea 0.05 0.05 --scale_tea 0.6 1.3 --color_tea 0.25 --blur_tea 0 --k 1 -b 32  --mask-ratio 0.5 --decoder-name saved_models/decoder_animal_0_1.pth.tar --s2t-freq 0.5 --s2t-alpha 0 1 --t2s-freq 0.5 --t2s-alpha 0 1 --occlude-rate 0.5 --occlude-thresh 0.9
```

SyntheticAnimal-to-AnimalPose
```
python train_animal_other.py path/to/animal_pose_dataset  --source synthetic_animal_sp_all_other --target animal_pose --target_ssl animal_pose_mt --train_on_all_cat --log logs/syn2animal_pose/syn2real --debug --seed 0 --lambda_c 1 --pretrain-epoch 40 --rotation_stu 60 --shear_stu -30 30 --translate_stu 0.05 0.05 --scale_stu 0.6 1.3 --color_stu 0.25 --blur_stu 0 --rotation_tea 60 --shear_tea -30 30 --translate_tea 0.05 0.05 --scale_tea 0.6 1.3 --color_tea 0.25 --blur_tea 0 --k 1 -b 32  --mask-ratio 0.5 --decoder-name saved_models/decoder_animal_0_1.pth.tar --s2t-freq 0.5 --s2t-alpha 0 1 --t2s-freq 0.5 --t2s-alpha 0 1 --occlude-rate 0.5 --occlude-thresh 0.9
```
             
# Acknowledgment

Code borrowed from [RegDA](https://github.com/thuml/Transfer-Learning-Library/tree/master/examples/domain_adaptation/keypoint_detection), [UDA-Aniaml](https://github.com/chaneyddtt/UDA-Animal-Pose).
