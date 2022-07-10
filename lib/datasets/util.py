"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import numpy as np
import cv2
import os
import torch
import scipy
from PIL import Image

def generate_target(joints, joints_vis, heatmap_size, sigma, image_size):
    """Generate heatamap for joints.

    Args:
        joints: (K, 2)
        joints_vis: (K, 1)
        heatmap_size: W, H
        sigma:
        image_size:

    Returns:

    """
    num_joints = joints.shape[0]
    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    target_weight[:, 0] = joints_vis[:, 0]

    target = np.zeros((num_joints,
                       heatmap_size[1],
                       heatmap_size[0]),
                      dtype=np.float32)
    tmp_size = sigma * 3
    image_size = np.array(image_size)
    heatmap_size = np.array(heatmap_size)

    for joint_id in range(num_joints):
        feat_stride = image_size / heatmap_size
        mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if mu_x >= heatmap_size[0] or mu_y >= heatmap_size[1] \
                or mu_x < 0 or mu_y < 0:
            # If not, just return the image as is
            target_weight[joint_id] = 0
            continue

        # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
        img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

        v = target_weight[joint_id]
        if v > 0.5:
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return target, target_weight


def keypoint2d_to_3d(keypoint2d: np.ndarray, intrinsic_matrix: np.ndarray, Zc: np.ndarray):
    """Convert 2D keypoints to 3D keypoints"""
    uv1 = np.concatenate([np.copy(keypoint2d), np.ones((keypoint2d.shape[0], 1))], axis=1).T * Zc  # 3 x NUM_KEYPOINTS
    xyz = np.matmul(np.linalg.inv(intrinsic_matrix), uv1).T  # NUM_KEYPOINTS x 3
    return xyz


def keypoint3d_to_2d(keypoint3d: np.ndarray, intrinsic_matrix: np.ndarray):
    """Convert 3D keypoints to 2D keypoints"""
    keypoint2d = np.matmul(intrinsic_matrix, keypoint3d.T).T  # NUM_KEYPOINTS x 3
    keypoint2d = keypoint2d[:, :2] / keypoint2d[:, 2:3]  # NUM_KEYPOINTS x 2
    return keypoint2d


def scale_box(box, image_width, image_height, scale, pad=False):
    """
    Change `box` to a square box.
    The side with of the square box will be `scale` * max(w, h)
    where w and h is the width and height of the origin box
    """
    left, upper, right, lower = box
    center_x, center_y = (left + right) / 2, (upper + lower) / 2
    w, h = right - left, lower - upper
    side_with = min(round(scale * max(w, h)), min(image_width, image_height))
    left = round(center_x - side_with / 2)
    right = left + side_with - 1
    upper = round(center_y - side_with / 2)
    lower = upper + side_with - 1
    if not pad:
        if left < 0:
            left = 0
            right = side_with - 1
        if right >= image_width:
            right = image_width - 1
            left = image_width - side_with
        if upper < 0:
            upper = 0
            lower = side_with -1
        if lower >= image_height:
            lower = image_height - 1
            upper = image_height - side_with

    return left, upper, right, lower

def get_bounding_box(keypoint2d: np.array):
    """Get the bounding box for keypoints"""
    left = np.min(keypoint2d[:, 0])
    right = np.max(keypoint2d[:, 0])
    upper = np.min(keypoint2d[:, 1])
    lower = np.max(keypoint2d[:, 1])
    return left, upper, right, lower


def visualize_heatmap(image, heatmaps, filename):
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR).copy()
    H, W = heatmaps.shape[1], heatmaps.shape[2]
    resized_image = cv2.resize(image, (int(W), int(H)))
    heatmaps = heatmaps.mul(255).clamp(0, 255).byte().cpu().numpy()
    for k in range(heatmaps.shape[0]):
        heatmap = heatmaps[k]
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        masked_image = colored_heatmap * 0.3 + resized_image * 0.7
        cv2.imwrite(filename.format(k), masked_image)
        

def area(left, upper, right, lower):
    return max(right - left + 1, 0) * max(lower - upper + 1, 0)


def intersection(box_a, box_b):
    left_a, upper_a, right_a, lower_a = box_a
    left_b, upper_b, right_b, lower_b = box_b
    return max(left_a, left_b), max(upper_a, upper_b), min(right_a, right_b), min(lower_a, lower_b)

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def isfile(fname):
    return os.path.isfile(fname) 

def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1)) # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img

def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0)) # H*W*C
    return img

def fliplr(x):
    if x.ndim == 3:
        x = np.transpose(np.fliplr(np.transpose(x, (0, 2, 1))), (0, 2, 1))
    elif x.ndim == 4:
        for i in range(x.shape[0]):
            x[i] = np.transpose(np.fliplr(np.transpose(x[i], (0, 2, 1))), (0, 2, 1))
    return x.astype(float)

def shufflelr_ori(x, width, dataset):
    """
    flip coords
    """
    if dataset == 'mpii':
        matchedParts = (
            [0,5],   [1,4],   [2,3],
            [10,15], [11,14], [12,13]
        )

    elif dataset == '_300w':
        matchedParts = ([0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9],
                        [17, 26], [18, 25], [19, 26], [20, 23], [21, 22], [36, 45], [37, 44],
                        [38, 43], [39, 42], [41, 46], [40, 47], [31, 35], [32, 34], [50, 52],
                        [49, 53], [48, 54], [61, 63], [62, 64], [67, 65], [59, 55], [58, 56])
    elif dataset == 'scut':
        matchedParts = ([1, 21], [2, 20], [3, 19], [4, 18], [5, 17], [6, 16], [7, 15],
                        [8, 14], [9, 13], [10, 12], [26, 32], [25, 33], [24, 34], [23, 35],
                        [22, 36], [27, 41], [28, 40], [29, 39], [30, 38], [31, 37],
                        [49, 55], [48, 56], [47, 57], [46, 50], [45, 51], [44, 52], [43, 53], [42, 54], [58, 59],
                        [60, 72], [61, 71], [62, 70], [63, 69], [64, 68], [65, 67],
                        [79, 73], [78, 74], [77, 75], [80, 85], [81, 84], [82, 83])
    elif dataset == 'real_animal':
        matchedParts = ([0, 1], [3, 4], [5, 6], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17])
    elif dataset == 'animal_pose':
        matchedParts = ([0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13])
    else:
        print('Not supported dataset: ' + dataset)

    # Flip horizontal
    x[:, 0] = width - x[:, 0]

    # Change left-right parts
    for pair in matchedParts:
        tmp = x[pair[0], :].clone()
        x[pair[0], :] = x[pair[1], :]
        x[pair[1], :] = tmp

    return x

def transform(pt, center, scale, res, invert=0, rot=0):
    # Transform pixel location to different reference
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1

def crop_ori(img, center, scale, res, rot=0):
    img = im_to_numpy(img)

    # Preprocessing for efficient cropping
    ht, wd = img.shape[0], img.shape[1]
    sf = scale * 200.0 / res[0]
    if sf < 2:
        sf = 1
    else:
        new_size = int(np.math.floor(max(ht, wd) / sf))
        new_ht = int(np.math.floor(ht / sf))
        new_wd = int(np.math.floor(wd / sf))
        if new_size < 2:
            return torch.zeros(res[0], res[1], img.shape[2]) \
                        if len(img.shape) > 2 else torch.zeros(res[0], res[1])
        else:
            img = scipy.misc.imresize(img, [new_ht, new_wd])
            # img = np.array(Image.fromarray(img.astype(np.uint8)).resize([new_ht, new_wd]))
            center = center * 1.0 / sf
            scale = scale / sf

    # Upper left point
    ul = np.array(transform([0, 0], center, scale, res, invert=1))
    # Bottom right point
    br = np.array(transform(res, center, scale, res, invert=1))

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(img.shape[1], br[0])
    old_y = max(0, ul[1]), min(img.shape[0], br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    new_img = im_to_torch(scipy.misc.imresize(new_img, res))
    # new_img = im_to_torch(np.array(Image.fromarray(new_img.astype(np.uint8)).resize(res)))
    return new_img

def get_transform(center, scale, res, rot=0):
    """
    General image processing functions
    """
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t

def color_normalize(x, mean, std):
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)

    for t, m, s in zip(x, mean, std):
        t.sub_(m)
    return x

def draw_labelmap_ori(img, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    img = to_numpy(img)

    # Check that any part of the gaussian is in-bounds
    pt = pt.to(torch.int32)
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    # if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
    #         br[0] < 0 or br[1] < 0):
    if (br[0] >= img.shape[1] or br[1] >= img.shape[0] or
            ul[0] < 0 or ul[1] < 0):
        # If not, just return the image as is
        return to_torch(img), 0

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)


    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])
    if img_y[1]-img_y[0] > 7:
        print('here')
    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return to_torch(img), 1

def load_image_ori(img_path):
    # H x W x C => C x H x W
    return im_to_torch(scipy.misc.imread(img_path, mode='RGB'))
    # return im_to_torch(np.array(Image.open(img_path)))
