import torch
import os
import pickle

from ._util import download as download_data, check_exits
from ..transforms.keypoint_detection import *
from .keypoint_dataset import Hand21KeypointDataset
from .util import *


class RenderedHandPose_mt(Hand21KeypointDataset):
    """`Rendered Handpose Dataset <https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html>`_

    Args:
        root (str): Root directory of dataset
        split (str, optional): The dataset split, supports ``train``, ``test``, or ``all``.
        task (str, optional): Placeholder.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transforms (callable, optional): A function/transform that takes in a dict (which contains PIL image and
            its labels) and returns a transformed version. E.g, :class:`~common.vision.transforms.keypoint_detection.Resize`.
        image_size (tuple): (width, height) of the image. Default: (256, 256)
        heatmap_size (tuple): (width, height) of the heatmap. Default: (64, 64)
        sigma (int): sigma parameter when generate the heatmap. Default: 2

    .. note:: In `root`, there will exist following files after downloading.
        ::
            RHD_published_v2/
                training/
                evaluation/
    """
    def __init__(self, root, split='train', task='all', download=True, k=1, 
                    transforms_base=None, transforms_stu=None, transforms_tea=None, **kwargs):
        if download:
            download_data(root, "RHD_published_v2", "RHD_v1-1.zip", "https://lmb.informatik.uni-freiburg.de/data/RenderedHandpose/RHD_v1-1.zip")
        else:
            check_exits(root, "RHD_published_v2")

        root = os.path.join(root, "RHD_published_v2")

        assert split in ['train', 'test', 'all', 'train-val', 'val']
        self.split = split
        self.transforms_base = transforms_base
        self.transforms_stu = transforms_stu
        self.transforms_tea = transforms_tea
        self.k = k
        if split == 'all':
            samples = self.get_samples(root, 'train') + self.get_samples(root, 'test')
        else:
            if split == 'val':
                samples = self.get_samples(root, 'train')
                samples = [e for i, e in enumerate(samples) if i % 5 == 0]
            elif split == 'train-val':
                samples = self.get_samples(root, 'train')
                samples = [e for i, e in enumerate(samples) if i % 5 != 0]
            else:
                samples = self.get_samples(root, split)

        super(RenderedHandPose_mt, self).__init__(
            root, samples, **kwargs)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_name = sample['name']
        image_path = os.path.join(self.root, image_name)
        image = Image.open(image_path)

        keypoint3d_camera = np.array(sample['keypoint3d'])  # NUM_KEYPOINTS x 3
        keypoint2d = np.array(sample['keypoint2d'])  # NUM_KEYPOINTS x 2
        intrinsic_matrix = np.array(sample['intrinsic_matrix'])
        Zc = keypoint3d_camera[:, 2]

        # Crop the images such that the hand is at the center of the image
        # The images will be 1.5 times larger than the hand
        # The crop process will change Xc and Yc, leaving Zc with no changes
        bounding_box = get_bounding_box(keypoint2d)
        w, h = image.size
        left, upper, right, lower = scale_box(bounding_box, w, h, 1.5)
        image, keypoint2d = crop(image, upper, left, lower - upper, right - left, keypoint2d)

        # Change all hands to right hands
        if sample['left'] is False:
            image, keypoint2d = hflip(image, keypoint2d)

        image, data = self.transforms_base(image, keypoint2d=keypoint2d, intrinsic_matrix=intrinsic_matrix)
        keypoint2d = data['keypoint2d']
        intrinsic_matrix = data['intrinsic_matrix']

        image_stu, data_stu = self.transforms_stu(image, keypoint2d=keypoint2d, intrinsic_matrix=intrinsic_matrix)
        keypoint2d_stu = data_stu['keypoint2d']
        intrinsic_matrix_stu = data_stu['intrinsic_matrix']
        aug_param_stu = data_stu['aug_param']
        keypoint3d_camera_stu = keypoint2d_to_3d(keypoint2d_stu, intrinsic_matrix_stu, Zc)

        # noramlize 2D pose:
        visible = np.array(sample['visible'], dtype=np.float32)
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
        z_stu = keypoint3d_n_stu[:, 2]
        meta_stu = {
            'image': image_name,
            'target_small_stu': generate_target(keypoint2d_stu, visible, (8, 8), self.sigma, self.image_size),
            'keypoint2d_ori': keypoint2d,  
            'target_ori': target_ori,  
            'target_weight_ori': target_weight_ori,  
            'keypoint2d_stu': keypoint2d_stu,  # （NUM_KEYPOINTS x 2）
            'keypoint3d_stu': keypoint3d_n_stu,  # （NUM_KEYPOINTS x 3）
            'aug_param_stu': aug_param_stu,
            'z_stu': z_stu,
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
            z_tea = keypoint3d_n_tea[:, 2]

            meta_tea = {
                'image': image_name,
                'target_small_tea': generate_target(keypoint2d_tea, visible, (8, 8), self.sigma, self.image_size),
                'keypoint2d_tea': keypoint2d_tea,  # （NUM_KEYPOINTS x 2）
                'keypoint3d_tea': keypoint3d_n_tea,  # （NUM_KEYPOINTS x 3）
                'aug_param_tea': aug_param_tea,
                'z_tea': z_tea,
            }

            images_tea.append(image_tea)
            targets_tea.append(target_tea)
            target_weights_tea.append(target_weight_tea)
            metas_tea.append(meta_tea)

        return image_stu, target_stu, target_weight_stu, meta_stu, images_tea, targets_tea, target_weights_tea, metas_tea

    def get_samples(self, root, task, min_size=64):
        if task == 'train':
            set = 'training'
        else:
            set = 'evaluation'
        # load annotations of this set
        with open(os.path.join(root, set, 'anno_%s.pickle' % set), 'rb') as fi:
            anno_all = pickle.load(fi)

        samples = []
        left_hand_index = [0, 4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17]
        right_hand_index = [i+21 for i in left_hand_index]
        for sample_id, anno in anno_all.items():
            image_name = os.path.join(set, 'color', '%.5d.png' % sample_id)
            mask_name = os.path.join(set, 'mask', '%.5d.png' % sample_id)
            keypoint2d = anno['uv_vis'][:, :2]
            keypoint3d = anno['xyz']
            intrinsic_matrix = anno['K']
            visible = anno['uv_vis'][:, 2]

            left_hand_keypoint2d = keypoint2d[left_hand_index] # NUM_KEYPOINTS x 2
            left_box = get_bounding_box(left_hand_keypoint2d)
            right_hand_keypoint2d = keypoint2d[right_hand_index]  # NUM_KEYPOINTS x 2
            right_box = get_bounding_box(right_hand_keypoint2d)

            w, h = 320, 320
            scaled_left_box = scale_box(left_box, w, h, 1.5)
            left, upper, right, lower = scaled_left_box
            size = max(right - left, lower - upper)
            if size > min_size and np.sum(visible[left_hand_index]) > 16 and area(*intersection(scaled_left_box, right_box)) / area(*scaled_left_box) < 0.3:
                sample = {
                    'name': image_name,
                    'mask_name': mask_name,
                    'keypoint2d': left_hand_keypoint2d,
                    'visible': visible[left_hand_index],
                    'keypoint3d': keypoint3d[left_hand_index],
                    'intrinsic_matrix': intrinsic_matrix,
                    'left': True
                }
                samples.append(sample)

            scaled_right_box = scale_box(right_box, w, h, 1.5)
            left, upper, right, lower = scaled_right_box
            size = max(right - left, lower - upper)
            if size > min_size and np.sum(visible[right_hand_index]) > 16 and area(*intersection(scaled_right_box, left_box)) / area(*scaled_right_box) < 0.3:
                sample = {
                    'name': image_name,
                    'mask_name': mask_name,
                    'keypoint2d': right_hand_keypoint2d,
                    'visible': visible[right_hand_index],
                    'keypoint3d': keypoint3d[right_hand_index],
                    'intrinsic_matrix': intrinsic_matrix,
                    'left': False
                }
                samples.append(sample)

        return samples