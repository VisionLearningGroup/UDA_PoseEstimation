"""
@author: Kaihong Wang
@contact: kaiwkh@bu.edu
"""
from __future__ import print_function, absolute_import
import os
import numpy as np
import json
import random
import math

import torch
import torch.utils.data as data

from scipy.io import loadmat
import glob
import scipy.misc
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import argparse
from .keypoint_dataset import Animal18KeypointDataset
from .util import isfile, im_to_torch, fliplr, shufflelr_ori, crop_ori, color_normalize, to_torch, transform, draw_labelmap_ori


# generate data with only 18 keypoints to save loading time
def generate_data_file():
    data = {}
    data_dir = './animal_data'
    animal_list = ['horse', 'tiger']
    for animal in sorted(animal_list):
        img_list = glob.glob(os.path.join(data_dir, 'synthetic_animal', animal + '_combineds5r5_texture', '*img.png'))
        img_list = sorted(img_list)
        print(len(img_list))

        bbox_all = []
        kpts_all = []
        if animal == 'horse':
            idxs = np.array(
                [1718, 1684, 1271, 1634, 1650, 1643, 1659, 925, 392, 564, 993, 726, 1585, 1556, 427, 1548, 967,
                 877])
        elif animal == 'tiger':
            idxs = np.array(
                [2753, 2679, 2032, 1451, 1287, 3085, 1632, 229, 1441, 1280, 2201, 1662, 266, 158, 270, 152, 219, 129])
        else:
            raise Exception('animal should be horse/tiger')

        train_idxs = np.load('./cached_data/synthetic_animal/' +animal+'_combineds5r5_texture'+ '/train_idxs.npy')
        valid_idxs = np.load('./cached_data/synthetic_animal/' + animal + '_combineds5r5_texture' + '/valid_idxs.npy')
        train_idxs = train_idxs.tolist()
        valid_idxs = valid_idxs.tolist()
        for img_path in img_list:
            kpts_path = img_path[:-7] + 'kpts.npy'
            pts = np.load(kpts_path)
            pts = pts
            y_min = min(pts[:, 1])
            y_max = max(pts[:, 1])
            x_min = min(pts[:, 0])
            x_max = max(pts[:, 0])
            bbox = [x_min, x_max, y_min, y_max]
            pts_18 = pts[idxs]
            kpts_all.append(pts_18.tolist())
            bbox_all.append(bbox)
        data[animal] = {'keypoints': kpts_all, 'imgpath': img_list, 'bbox': bbox_all, 'train_idxs': train_idxs,
                        'valid_idxs': valid_idxs}
    with open('{}/clean_data/keypoints_18.json'.format(data_dir), 'w') as f:
        json.dump(data, f)
    print('Generate data files done')


class Synthetic_Animal_SP_All(Animal18KeypointDataset):

    eye = (0, 1)
    chin = (2,)
    hoof = (3, 4, 5, 6)
    hip = (7,)
    knee = (8, 9, 10, 11)
    shoulder = (12, 13)
    elbow = (14, 15, 16, 17)
    all = tuple(range(18))

    right_front_leg = (3, 8, 14)
    left_front_leg = (4, 9, 15)
    right_back_leg = (5, 10, 16)
    left_back_leg = (6, 11, 17)
    right_torso = (13, 7)
    right_face = (1, 2)
    left_torso = (12, 7)
    left_face = (0, 2)

    colored_skeleton = {
        "right_front_leg": (right_front_leg, [0, 0, 255]),
        "left_front_leg": (left_front_leg, [0, 127, 255]),
        "right_back_leg": (right_back_leg, [0, 255, 255]),
        "left_back_leg": (left_back_leg, [0, 255, 0]),
        "right_torso": (right_torso, [255, 0, 0]),
        "right_face": (right_face, [255, 0, 0]),
        "left_torso": (left_torso, [255, 0, 255]),
        "left_face": (left_face, [255, 0, 255]),
    }
    keypoints_group = {
        "eye": eye,
        "chin": chin,
        "hoof": hoof,
        "hip": hip,
        "knee": knee, 
        "shoulder": shoulder,
        "elbow": elbow, 
        "all": all,
    }

    num_keypoints = 18

    def __init__(self, is_train=True, is_aug=True, **kwargs):
        print("init all synthetic animal super augmentation")
        animal_total = ['horse', 'tiger']
        self.animal = animal_total if kwargs['animal'] == 'all' else [kwargs['animal']]

        self.nParts = 18
        self.img_folder = kwargs['image_path']  # root image folders
        self.is_train = is_train  # training set or test set
        self.is_aug = is_aug  # training set or test set
        self.inp_res = kwargs['inp_res']
        self.out_res = kwargs['out_res']
        self.sigma = kwargs['sigma']
        self.scale_factor = kwargs['scale_factor']
        self.rot_factor = kwargs['rot_factor']
        self.label_type = kwargs['label_type']
        self.train_with_occlusion = True   # whether use occluded joints

        # create train/val split
        self.data_dict = {}
        self.train_set = []
        self.valid_set = []
        self.load_animal()
        self.mean, self.std = self._compute_mean()

        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        self.seq = iaa.Sequential(
            [
                sometimes(iaa.Affine(
                    scale={"x": (0.5, 1.5), "y": (0.5, 1.5)},
                    # scale images to 50-150% of their size, individually per axis
                    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                    # translate by -5 to +5 percent (per axis)
                    rotate=(-30, 30),  # rotate by -30 to +30 degrees
                    shear=(-20, 20),  # shear by -20 to +20 degrees
                    order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                    mode='constant'  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                sometimes(iaa.AdditiveGaussianNoise(scale=0.5 * 255, per_channel=0.5)),
                sometimes(iaa.GaussianBlur(sigma=(1.0, 5.0))),
                sometimes(iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)),  # improve or worsen the contrast
            ],
            random_order=True
        )

    def load_animal(self):

        # img_list contains all image paths
        data_file_path = '{}/clean_data/keypoints_18.json'.format(self.img_folder)
        with open(data_file_path, 'r') as f:
            data_file = json.load(f)
        for animal in self.animal:
            annot = data_file[animal]
            imgpath = annot['imgpath']
            if self.img_folder != "animal_data":
                imgpath = [i.replace("animal_data", self.img_folder) for i in imgpath]
            train_idxs = annot['train_idxs']
            valid_idxs = annot['valid_idxs']
            print('Animal: {}, training samples: {}, valid samples: {}'.format(animal, len(train_idxs), len(valid_idxs)))
            self.data_dict.update(dict.fromkeys(imgpath))
            for idx in range(len(train_idxs)):
                self.train_set.append(imgpath[train_idxs[idx]])
                pts = np.array(annot['keypoints'][train_idxs[idx]], dtype=np.float32)
                bbox = annot['bbox'][train_idxs[idx]]
                self.data_dict[imgpath[train_idxs[idx]]] = {'pts': pts, 'bbox': bbox}
            for idx in range(len(valid_idxs)):
                valid_idx = valid_idxs[idx]
                self.valid_set.append(imgpath[valid_idx])
                pts = np.array(annot['keypoints'][valid_idx], dtype=np.float32)
                bbox = annot['bbox'][valid_idx]
                self.data_dict[imgpath[valid_idx]] = {'pts': pts, 'bbox': bbox}
        print('--Training set : {} samples, Valid set : {} samples'.format(len(self.train_set), len(self.valid_set)))

    def _compute_mean(self):
        # compute or load data statistics
        animal = self.animal[0] if len(self.animal) == 1 else 'all'
        meanstd_file = './cached_data/synthetic_animal/' + animal + '_combineds5r5_texture' + '/mean.pth.tar'
        if isfile(meanstd_file):
            print('load from mean file:', meanstd_file)
            meanstd = torch.load(meanstd_file)
        else:
            print("generate mean file")
            mean = torch.zeros(3)
            std = torch.zeros(3)
            for img_path in self.train_set:
                img = load_image_ori(img_path)  # CxHxW
                mean += img.view(img.size(0), -1).mean(1)
                std += img.view(img.size(0), -1).std(1)
            mean /= len(self.train_set)
            std /= len(self.train_set)
            meanstd = {
                'mean': mean,
                'std': std,
            }
            torch.save(meanstd, meanstd_file)
        if self.is_train:
            print('    Mean: %.4f, %.4f, %.4f' % (meanstd['mean'][0], meanstd['mean'][1], meanstd['mean'][2]))
            print('    Std:  %.4f, %.4f, %.4f' % (meanstd['std'][0], meanstd['std'][1], meanstd['std'][2]))

        return meanstd['mean'], meanstd['std']

    def __getitem__(self, index):

        dataset = self.train_set if self.is_train else self.valid_set
        img_path = dataset[index]
        x_min, x_max, y_min, y_max = self.data_dict[img_path]['bbox']
        pts = self.data_dict[img_path]['pts']

        # update keypoints visibility for different number of keypoints
        if self.train_with_occlusion:
            pts[:, 2] = 1
        else:
            pts *= pts[:, 2].reshape(-1, 1)

        pts_aug = pts[:, :2].copy()

        # center and scale
        # do not add additional 15 pixels as real data as synthetic data always includes full 18 joints,
        # bbox can directly be generated from keypoints
        x_min = np.clip(x_min, 0, 640)
        y_min = np.clip(y_min, 0, 480)
        x_max = np.clip(x_max, 0, 640)
        y_max = np.clip(y_max, 0, 480)

        c = torch.Tensor(((x_min + x_max) / 2.0, (y_min + y_max) / 2.0))
        s = max(x_max - x_min, y_max - y_min) / 200.0 * 1.25

        # For single-animal pose estimation with a centered/scaled figure
        img = np.array(imageio.imread(img_path))[:, :, :3]
        img_aug = np.expand_dims(img, axis=0)
        pts_aug = np.expand_dims(pts_aug, axis=0)

        # import cv2
        # print(img_aug.shape)
        # cv2.imwrite("t0.png", img_aug[0][:,:,::-1])
        r = 0
        if self.is_train and self.is_aug:
            img_aug, pts_aug = self.seq(images=img_aug, keypoints=pts_aug)

        # cv2.imwrite("t1.png", img_aug[0][:,:,::-1])

        img = img_aug.squeeze(0)
        img = im_to_torch(img)
        pts[:, :2] = pts_aug
        pts = torch.Tensor(pts)

        for j in range(pts.size()[0]):
            if pts[j][0] < 0 or pts[j][1] < 0 or pts[j][0] > 640 or pts[j][1] > 480:
                pts[j] = 0

        if self.is_train:
            if random.random() <= 0.5:
                img = torch.from_numpy(fliplr(img.numpy())).float()
                pts = shufflelr_ori(pts, width=img.size(2), dataset='real_animal')
                c[0] = img.size(2) - c[0]

        # Prepare image and groundtruth map
        # print(img.size())
        # print(c, s, [self.inp_res, self.inp_res])
        # from torchvision.utils import save_image
        # save_image(img, "t2.png")
        inp = crop_ori(img, c, s, [self.inp_res, self.inp_res], rot=r)
        # save_image(inp, "t3.png")
        # raise ValueError
        inp = color_normalize(inp, self.mean, self.std)


        # Generate ground truth
        tpts = pts.clone()
        tpts_inpres = pts.clone()
        nparts = tpts.shape[0]
        target = torch.zeros(nparts, self.out_res, self.out_res)
        target_weight = tpts[:, 2].clone().view(nparts, 1)

        for i in range(nparts):
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2] + 1, c, s, [self.out_res, self.out_res], rot=r))
                tpts_inpres[i, 0:2] = to_torch(transform(tpts_inpres[i, 0:2] + 1, c, s, [self.inp_res, self.inp_res], rot=r))
                target[i], vis = draw_labelmap_ori(target[i], tpts[i] - 1, self.sigma, type=self.label_type)
                target_weight[i, 0] *= vis
        tpts[:, 2] = target_weight.view(-1)

        # Meta info
        meta = {'index': index, 'center': c, 'scale': s,
                'pts': pts, 'tpts': tpts, 'keypoint2d': tpts_inpres}
        return inp, target, target_weight, meta

    def __len__(self):
        if self.is_train:
            return len(self.train_set)
        else:
            return len(self.valid_set)


def synthetic_animal_sp_all(**kwargs):
    return Synthetic_Animal_SP_All(**kwargs)


# synthetic_animal_sp.njoints = 3299  # ugly but works
synthetic_animal_sp_all.njoints = 18

if __name__ == '__main__':
    generate_data_file()


