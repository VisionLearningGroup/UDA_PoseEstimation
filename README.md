# UDA_PoseEstimation
Code for [A Unified Framework for Domain Adaptive Pose Estimation](https://arxiv.org/pdf/2204.00172.pdf), accepted at ECCV 2022

# Introduction

# Usage

**Data Preparation**

**Pretrained Models**

Before training, please make sure style transfer models are downloaded and saved in the "saved_models" folder under this directory. Pretrained models in all experiments are available [here](https://drive.google.com/drive/folders/1WVjQ2Hq1CrtUr3AOlq1PxDuY50KpZ8lh?usp=sharing). 

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
