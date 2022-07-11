import os
import json
import random
from PIL import ImageFile, Image
import torch
import os.path as osp

from ._util import download as download_data, check_exits
from .keypoint_dataset import Hand21KeypointDataset
from .util import *

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Hand3DStudio_mt(Hand21KeypointDataset):
    """`Hand-3d-Studio Dataset <https://www.yangangwang.com/papers/ZHAO-H3S-2020-02.html>`_

    Args:
        root (str): Root directory of dataset
        split (str, optional): The dataset split, supports ``train``, ``test``, or ``all``.
        task (str, optional): The task to create dataset. Choices include ``'noobject'``: only hands without objects, \
            ``'object'``: only hands interacting with hands, and ``'all'``: all hands. Default: 'noobject'.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transforms (callable, optional): A function/transform that takes in a dict (which contains PIL image and
            its labels) and returns a transformed version. E.g, :class:`~common.vision.transforms.keypoint_detection.Resize`.
        image_size (tuple): (width, height) of the image. Default: (256, 256)
        heatmap_size (tuple): (width, height) of the heatmap. Default: (64, 64)
        sigma (int): sigma parameter when generate the heatmap. Default: 2

    .. note::
        We found that the original H3D image is in high resolution while most part in an image is background,
        thus we crop the image and keep only the surrounding area of hands (1.5x bigger than hands) to speed up training.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            H3D_crop/
                annotation.json
                part1/
                part2/
                part3/
                part4/
                part5/
    """
    def __init__(self, root, split='train', task='noobject', download=True,  k=1, 
                    transforms_base=None, transforms_stu=None, transforms_tea=None, **kwargs):
        assert split in ['train', 'test', 'all', 'train-val', 'val']
        self.split = split
        assert task in ['noobject', 'object', 'all']
        self.task = task

        if download:
            download_data(root, "H3D_crop", "H3D_crop.tar", "https://cloud.tsinghua.edu.cn/f/d4e612e44dc04d8eb01f/?dl=1")
        else:
            check_exits(root, "H3D_crop")

        root = osp.join(root, "H3D_crop")
        # load labels
        annotation_file = os.path.join(root, 'annotation.json')
        print("loading from {}".format(annotation_file))
        with open(annotation_file) as f:
            samples = list(json.load(f))
        if task == 'noobject':
            samples = [sample for sample in samples if int(sample['without_object']) == 1]
        elif task == 'object':
            samples = [sample for sample in samples if int(sample['without_object']) == 0]
        self.transforms_base = transforms_base
        self.transforms_stu = transforms_stu
        self.transforms_tea = transforms_tea
        self.k = k

        random.seed(42)
        random.shuffle(samples)
        samples_len = len(samples)
        samples_split = min(int(samples_len * 0.2), 3200)

        if split == 'train':
            samples = samples[samples_split:]
        elif split == 'test':
            samples = samples[:samples_split]
        elif split == 'train-val':
            samples = samples[2*samples_split:]
        elif split == 'val':
            samples = samples[samples_split:2*samples_split]

        super(Hand3DStudio_mt, self).__init__(root, samples, **kwargs)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_name = sample['name']
        image_path = os.path.join(self.root, image_name)
        image = Image.open(image_path)
        keypoint3d_camera = np.array(sample['keypoint3d'])  # NUM_KEYPOINTS x 3
        keypoint2d = np.array(sample['keypoint2d'])  # NUM_KEYPOINTS x 2
        intrinsic_matrix = np.array(sample['intrinsic_matrix'])
        Zc = keypoint3d_camera[:, 2]

        image, data = self.transforms_base(image, keypoint2d=keypoint2d, intrinsic_matrix=intrinsic_matrix)
        keypoint2d = data['keypoint2d']
        intrinsic_matrix = data['intrinsic_matrix']

        image_stu, data_stu = self.transforms_stu(image, keypoint2d=keypoint2d, intrinsic_matrix=intrinsic_matrix)
        keypoint2d_stu = data_stu['keypoint2d']
        intrinsic_matrix_stu = data_stu['intrinsic_matrix']
        aug_param_stu = data_stu['aug_param']
        keypoint3d_camera_stu = keypoint2d_to_3d(keypoint2d_stu, intrinsic_matrix_stu, Zc)

        # noramlize 2D pose:
        visible = np.ones((self.num_keypoints, ), dtype=np.float32)
        visible = visible[:, np.newaxis]
        # 2D heatmap
        target_stu, target_weight_stu = generate_target(keypoint2d_stu, visible, self.heatmap_size, self.sigma, self.image_size)
        target_stu = torch.from_numpy(target_stu)
        target_weight_stu = torch.from_numpy(target_weight_stu)

        target_ori, target_weight_ori = generate_target(keypoint2d, visible, self.heatmap_size, self.sigma, self.image_size)
        target_ori = torch.from_numpy(target_ori)
        target_weight_ori = torch.from_numpy(target_weight_ori)

        # normalize 3D pose:
        # put middle finger metacarpophalangeal (MCP) joint in the center of the coordinate system
        # and make distance between wrist and middle finger MCP joint to be of length 1
        keypoint3d_n_stu = keypoint3d_camera_stu - keypoint3d_camera_stu[9:10, :]
        keypoint3d_n_stu = keypoint3d_n_stu / np.sqrt(np.sum(keypoint3d_n_stu[0, :] ** 2))

        meta_stu = {
            'image': image_name,
            'target_small_stu': generate_target(keypoint2d_stu, visible, (8, 8), self.sigma, self.image_size),
            'keypoint2d_ori': keypoint2d,  
            'target_ori': target_ori,  
            'target_weight_ori': target_weight_ori,  
            'keypoint2d_stu': keypoint2d_stu,  # （NUM_KEYPOINTS x 2）
            'keypoint3d_stu': keypoint3d_n_stu,  # （NUM_KEYPOINTS x 3）
            'aug_param_stu': aug_param_stu,
        }

        images_tea, targets_tea, target_weights_tea, metas_tea = [], [], [], []
        for _ in range(self.k):
            image_tea, data_tea = self.transforms_tea(image, keypoint2d=keypoint2d, intrinsic_matrix=intrinsic_matrix)
            keypoint2d_tea = data_tea['keypoint2d']
            intrinsic_matrix_tea = data_tea['intrinsic_matrix']
            aug_param_tea = data_tea['aug_param']
            keypoint3d_camera_tea = keypoint2d_to_3d(keypoint2d_tea, intrinsic_matrix_tea, Zc)

            # 2D heatmap
            target_tea, target_weight_tea = generate_target(keypoint2d_tea, visible, self.heatmap_size, self.sigma, self.image_size)
            target_tea = torch.from_numpy(target_tea)
            target_weight_tea = torch.from_numpy(target_weight_tea)

            # normalize 3D pose:
            # put middle finger metacarpophalangeal (MCP) joint in the center of the coordinate system
            # and make distance between wrist and middle finger MCP joint to be of length 1
            keypoint3d_n_tea = keypoint3d_camera_tea - keypoint3d_camera_tea[9:10, :]
            keypoint3d_n_tea = keypoint3d_n_tea / np.sqrt(np.sum(keypoint3d_n_tea[0, :] ** 2))

            meta_tea = {
                'image': image_name,
                'target_small_tea': generate_target(keypoint2d_tea, visible, (8, 8), self.sigma, self.image_size),
                'keypoint2d_tea': keypoint2d_tea,  # （NUM_KEYPOINTS x 2）
                'keypoint3d_tea': keypoint3d_n_tea,  # （NUM_KEYPOINTS x 3）
                'aug_param_tea': aug_param_tea,
            }
            images_tea.append(image_tea)
            targets_tea.append(target_tea)
            target_weights_tea.append(target_weight_tea) 
            metas_tea.append(meta_tea)

        return image_stu, target_stu, target_weight_stu, meta_stu, images_tea, targets_tea, target_weights_tea, metas_tea


class Hand3DStudioAll_mt(Hand3DStudio_mt):
    """
    `Hand-3d-Studio Dataset <https://www.yangangwang.com/papers/ZHAO-H3S-2020-02.html>`_

    """
    def __init__(self,  root, task='all', **kwargs):
        super(Hand3DStudioAll_mt, self).__init__(root, task=task, **kwargs)