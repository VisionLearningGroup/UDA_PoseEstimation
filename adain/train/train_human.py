import argparse
import sys
sys.path.append('..')
sys.path.append('path/to/UDAPE/adain')
from pathlib import Path
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from torchvision import transforms
from tqdm import tqdm
from net import decoder, vgg, Net
from torchvision.utils import save_image
import cv2
import skimage.io
import numpy as np
from lib.data import ForeverDataIterator
from lib import datasets
from lib.transforms import keypoint_detection as T
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.manual_seed(0)
np.random.seed(0)

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--source', type=str)
parser.add_argument('--target', type=str)
parser.add_argument('--source_root', type=str)
parser.add_argument('--target_root', type=str)
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--image-size', type=int, default=256,
                    help='input image size')
parser.add_argument('--resize-scale', nargs='+', type=float, default=(0.6, 1.3),
                    help='scale range for the RandomResizeCrop augmentation')
parser.add_argument('--rotation', type=int, default=180,
                    help='rotation range of the RandomRotation augmentation')
parser.add_argument('--heatmap-size', type=int, default=64,
                    help='output heatmap size')
parser.add_argument('--color', type=float, default=0,
                    help='color range of the jitter augmentation')
parser.add_argument('--blur', type=float, default=0,
                    help='blur range of the jitter augmentation')
parser.add_argument('--shear', nargs='+', type=float, default=(0, 0),
                    help='shear range for the RandomResizeCrop augmentation')
parser.add_argument('--translate', nargs='+', type=float, default=(0, 0),
                    help='tranlate range for the RandomResizeCrop augmentation')
parser.add_argument('--scale', nargs='+', type=float, default=(0.6, 1.3),
                    help='scale range for the RandomResizeCrop augmentation')
parser.add_argument('--rotation_stu', type=int, default=180,
                    help='rotation range of the RandomRotation augmentation')
parser.add_argument('--color_stu', type=float, default=0,
                    help='color range of the jitter augmentation')
parser.add_argument('--blur_stu', type=float, default=0,
                    help='blur range of the jitter augmentation')
parser.add_argument('--shear_stu', nargs='+', type=float, default=(0, 0),
                    help='shear range for the RandomResizeCrop augmentation')
parser.add_argument('--translate_stu', nargs='+', type=float, default=(0, 0),
                    help='tranlate range for the RandomResizeCrop augmentation')
parser.add_argument('--scale_stu', nargs='+', type=float, default=(0.6, 1.3),
                    help='scale range for the RandomResizeCrop augmentation')

parser.add_argument('--rotation_tea', type=int, default=180,
                    help='rotation range of the RandomRotation augmentation')
parser.add_argument('--color_tea', type=float, default=0,
                    help='color range of the jitter augmentation')
parser.add_argument('--blur_tea', type=float, default=0,
                    help='blur range of the jitter augmentation')
parser.add_argument('--shear_tea', nargs='+', type=float, default=(0, 0),
                    help='shear range for the RandomResizeCrop augmentation')
parser.add_argument('--translate_tea', nargs='+', type=float, default=(0, 0),
                    help='tranlate range for the RandomResizeCrop augmentation')
parser.add_argument('--scale_tea', nargs='+', type=float, default=(0.6, 1.3),
                    help='scale range for the RandomResizeCrop augmentation')


# training options
parser.add_argument('--save_model_dir', default='./saved_model',
                    help='Directory to save the model')
parser.add_argument('--exp_name', default='./',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=500000)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--style_weight', type=float, default=1.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--log_img_interval', type=int, default=5000)
args = parser.parse_args()

normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
def denormalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    ten = x.clone()
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # return torch.clamp(ten, 0, 1)
    return ten
    
train_transform = T.Compose([
    T.RandomResizedCrop(size=args.image_size, scale=args.resize_scale),
    T.RandomAffineRotation(args.rotation, args.shear, args.translate, args.scale),
    T.ColorJitter(brightness=args.color, contrast=args.color, saturation=args.color),
    T.GaussianBlur(high=args.blur),
    T.ToTensor(),
    normalize
])
val_transform = T.Compose([
    T.Resize(args.image_size),
    T.ToTensor(),
    normalize
])
base_transform = T.Compose([
    T.RandomResizedCrop(size=args.image_size, scale=args.resize_scale),
])
tgt_train_transform_stu = T.Compose([
    T.RandomAffineRotation(args.rotation_stu, args.shear_stu, args.translate_stu, args.scale_stu),
    T.ColorJitter(brightness=args.color_stu, contrast=args.color_stu, saturation=args.color_stu),
    T.GaussianBlur(high=args.blur_stu),
    T.ToTensor(),
    normalize
])
tgt_train_transform_tea = T.Compose([
    T.RandomAffineRotation(args.rotation_tea, args.shear_tea, args.translate_tea, args.scale_tea),
    T.ColorJitter(brightness=args.color_tea, contrast=args.color_tea, saturation=args.color_tea),
    T.GaussianBlur(high=args.blur_tea),
    T.ToTensor(),
    normalize
])

image_size = (args.image_size, args.image_size)
heatmap_size = (args.heatmap_size, args.heatmap_size)

source_dataset = datasets.__dict__[args.source]
train_source_dataset = source_dataset(root=args.source_root, transforms=train_transform,
                                        image_size=image_size, heatmap_size=heatmap_size)
train_source_loader = DataLoader(train_source_dataset, batch_size=4,
                                    shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
target_dataset = datasets.__dict__[args.target]
train_target_dataset = target_dataset(root=args.target_root, transforms_base=base_transform,
                                        transforms_stu=tgt_train_transform_stu, transforms_tea=tgt_train_transform_tea, 
                                        image_size=image_size, heatmap_size=heatmap_size)
train_target_loader = DataLoader(train_target_dataset, batch_size=4,
                                    shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
train_source_iter = ForeverDataIterator(train_source_loader)
train_target_iter = ForeverDataIterator(train_target_loader)

device = torch.device('cuda')
exp_name = args.exp_name
log_root = "logs/" + exp_name

save_model_dir = Path(os.path.join(log_root, args.save_model_dir))
save_model_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(log_root)
log_dir.mkdir(exist_ok=True, parents=True)

fname = os.path.join(log_dir, "log_" + exp_name + ".txt")
out = os.path.join(log_root, "save_imgs/save_img_" + exp_name + "/")
if not os.path.exists(out):
    os.makedirs(out)
f = open(fname, "w")
f.close()

decoder = decoder
vgg = vgg
vgg_pretrained_path = '../saved_models/vgg_normalised.pth'
vgg.load_state_dict(torch.load(vgg_pretrained_path))
vgg = nn.Sequential(*list(vgg.children())[:31])
transfer_network = Net(vgg, decoder)
transfer_network.train()
transfer_network.to(device)
decoder_optimizer = torch.optim.Adam(transfer_network.decoder.parameters(), lr=args.lr)


i = 0

for e in tqdm(range(args.max_iter)):
    source_image, _, _, _ = next(train_source_iter)
    _, _, _, _, target_images, _, _, _ = next(train_target_iter)
    target_image = target_images[0]

    if np.random.rand() > 0.5:
        content_images = source_image
        style_images = target_image
    else:
        content_images = target_image
        style_images = source_image

    content_images = content_images.to(device).float() 
    style_images = style_images.to(device).float()

    # decoder
    loss_c, loss_s, g_t = transfer_network(content_images, style_images)
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    decoder_loss = loss_c + loss_s
    decoder_optimizer.zero_grad()
    decoder_loss.backward()
    decoder_optimizer.step()

    # log
    f = open(fname,"a")
    report = "iter: " + str(i) + ", decoder_loss: " + str(decoder_loss.item()) +  ", content loss: " + str(loss_c.item()) +  ", style loss: " + str(loss_s.item())
    f.write(report + '\n')

    if i % args.log_img_interval == 0:
        im = torch.cat((
            (denormalize(g_t[0].detach().cpu())), 
            (denormalize(content_images[0].detach().cpu())), 
            (denormalize(style_images[0].detach().cpu()))), axis=2)
        save_image(im, out + str(i) + ".png")

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        save_name = os.path.join(save_model_dir, "decoder_" + exp_name + ".pth.tar")
        torch.save(state_dict, save_name)
    i += 1
    if i >= args.max_iter:
        break

