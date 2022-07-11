import random
import time
import warnings
import sys
import argparse
import shutil
import os
import shutil
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToPILImage
import torch.nn.functional as F
import torchvision.transforms.functional as tF
import lib.models as models
from lib.models.loss import JointsMSELoss, ConsLoss
import lib.datasets as datasets
import lib.transforms.keypoint_detection as T
from lib.transforms import Denormalize
from lib.data import ForeverDataIterator
from lib.meter import AverageMeter, ProgressMeter, AverageMeterDict, AverageMeterList
from lib.keypoint_detection import accuracy
from lib.logger import CompleteLogger
from lib.models import Style_net
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
recover_min = torch.tensor([-2.1179, -2.0357, -1.8044]).to(device)
recover_max = torch.tensor([2.2489, 2.4285, 2.64]).to(device)

def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log + '_' + args.arch, args.phase)

    logger.write(' '.join(f'{k}={v}' for k, v in vars(args).items()))
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    src_train_transform = T.Compose([
        T.RandomResizedCrop(size=args.image_size, scale=args.resize_scale),
        T.RandomAffineRotation(args.rotation_stu, args.shear_stu, args.translate_stu, args.scale_stu),
        T.ColorJitter(brightness=args.color_stu, contrast=args.color_stu, saturation=args.color_stu),
        T.GaussianBlur(high=args.blur_stu),
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
    val_transform = T.Compose([
        T.Resize(args.image_size),
        T.ToTensor(),
        normalize
    ])
    image_size = (args.image_size, args.image_size)
    heatmap_size = (args.heatmap_size, args.heatmap_size)
    source_dataset = datasets.__dict__[args.source]
    train_source_dataset = source_dataset(root=args.source_root, transforms=src_train_transform,
                                          image_size=image_size, heatmap_size=heatmap_size)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_source_dataset = source_dataset(root=args.source_root, split='test', transforms=val_transform,
                                        image_size=image_size, heatmap_size=heatmap_size)
    val_source_loader = DataLoader(val_source_dataset, batch_size=args.test_batch, shuffle=False, pin_memory=True)

    target_dataset = datasets.__dict__[args.target_train]
    train_target_dataset = target_dataset(root=args.target_root, transforms_base=base_transform,
                                          transforms_stu=tgt_train_transform_stu, transforms_tea=tgt_train_transform_tea, 
                                          k=args.k, image_size=image_size, heatmap_size=heatmap_size)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    target_dataset = datasets.__dict__[args.target]
    val_target_dataset = target_dataset(root=args.target_root, split='test', transforms=val_transform,
                                        image_size=image_size, heatmap_size=heatmap_size)
    val_target_loader = DataLoader(val_target_dataset, batch_size=args.test_batch, shuffle=False, pin_memory=True)

    logger.write("Source train: {}".format(len(train_source_loader)))
    logger.write("Target train: {}".format(len(train_target_loader)))
    logger.write("Source test: {}".format(len(val_source_loader)))
    logger.write("Target test: {}".format(len(val_target_loader)))

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model

    student = models.__dict__[args.arch](num_keypoints=train_source_dataset.num_keypoints).cuda()
    teacher = models.__dict__[args.arch](num_keypoints=train_source_dataset.num_keypoints).cuda()

    if args.decoder_name is not None:
        decoder = Style_net.decoder
        decoder_pretrained_path = args.decoder_name
        decoder.load_state_dict(torch.load(decoder_pretrained_path))
        vgg = Style_net.vgg
        vgg_pretrained_path = 'saved_models/vgg_normalised.pth'
        vgg.load_state_dict(torch.load(vgg_pretrained_path))
        vgg = torch.nn.Sequential(*list(vgg.children())[:31])
        style_net = Style_net.Net(vgg, decoder)
        style_net.requires_grad=False
    else:
        style_net = None
    
    criterion = JointsMSELoss()
    con_criterion = ConsLoss()

    if args.SGD:
        stu_optimizer = SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001, nesterov=True)
    else:
        stu_optimizer = Adam(student.parameters(), lr=args.lr)

    tea_optimizer = OldWeightEMA(teacher, student, alpha=args.teacher_alpha)

    lr_scheduler = MultiStepLR(stu_optimizer, args.lr_step, args.lr_factor)

    student = torch.nn.DataParallel(student).cuda()
    teacher = torch.nn.DataParallel(teacher).cuda()
    if style_net is not None:
        style_net = torch.nn.DataParallel(style_net).cuda()


    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        student.load_state_dict(checkpoint['student'])
        teacher.load_state_dict(checkpoint['teacher'])
        stu_optimizer.load_state_dict(checkpoint['stu_optimizer'])
        # tea_optimizer.load_state_dict(checkpoint['tea_optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1

    elif args.pretrain:
        pretrained_dict = torch.load(args.pretrain, map_location='cpu')['student']
        model_dict = student.state_dict()
        # remove keys from pretrained dict that doesn't appear in model dict
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        student.load_state_dict(pretrained_dict, strict=False)
        teacher.load_state_dict(pretrained_dict, strict=False)


    # define visualization function
    tensor_to_image = Compose([
        Denormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToPILImage()
    ])

    def visualize(image, keypoint2d, name):
        """
        Args:
            image (tensor): image in shape 3 x H x W
            keypoint2d (tensor): keypoints in shape K x 2
            name: name of the saving image
        """
        train_source_dataset.visualize(tensor_to_image(image),
                                       keypoint2d, logger.get_image_path("{}.jpg".format(name)))

    if args.phase == 'test':
        # evaluate on validation set
        source_val_acc = validate(val_source_loader, teacher, criterion, None, args)
        target_val_acc = validate(val_target_loader, teacher, criterion, visualize, args)

        logger.write("Source: {:4.3f} Target: {:4.3f}".format(source_val_acc['all'], target_val_acc['all']))
        for name, acc in target_val_acc.items():
            logger.write("{}: {:4.3f}".format(name, acc))
        return

    # start training
    best_acc = 0

    for epoch in range(start_epoch, args.epochs):
        logger.set_epoch(epoch)
        lr_scheduler.step()

        # train for one epoch
        if epoch < args.pretrain_epoch:
            pretrain(train_source_iter, train_target_iter, student, style_net, criterion, stu_optimizer, epoch, visualize if args.debug else None, args)
        else:
            if epoch == args.pretrain_epoch:
                pretrained_dict = torch.load(logger.get_checkpoint_path('best_pt'), map_location='cpu')['student']
                model_dict = student.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                student.load_state_dict(pretrained_dict, strict=False)
                teacher.load_state_dict(pretrained_dict, strict=False)

            train(train_source_iter, train_target_iter, student, teacher, style_net, criterion, con_criterion, 
                    stu_optimizer, tea_optimizer, epoch,visualize if args.debug else None, args)

        # evaluate on validation set
        if epoch < args.pretrain_epoch:
            source_val_acc = validate(val_source_loader, student, criterion, None, args)
            target_val_acc = validate(val_target_loader, student, criterion, visualize if args.debug else None, args)
        else:
            source_val_acc = validate(val_source_loader, teacher, criterion, None, args)
            target_val_acc = validate(val_target_loader, teacher, criterion, visualize if args.debug else None, args)

        if target_val_acc['all'] > best_acc:
            torch.save(
                {
                    'student': student.state_dict(),
                    'teacher': teacher.state_dict(),
                    'stu_optimizer': stu_optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args
                }, logger.get_checkpoint_path('best_pt' if epoch < args.pretrain_epoch else 'best')
            )
            best_acc = target_val_acc['all']
        logger.write("Epoch: {} Source: {:4.3f} Target: {:4.3f} Target(best): {:4.3f}".format(epoch, source_val_acc['all'], target_val_acc['all'], best_acc))
        for name, acc in target_val_acc.items():
            logger.write("{}: {:4.3f}".format(name, acc))

    logger.close()

def pretrain(train_source_iter, train_target_iter, student, style_net, criterion, stu_optimizer, epoch: int, visualize, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses_all = AverageMeter('Loss (all)', ":.4e")
    losses_s = AverageMeter('Loss (s)', ":.4e")
    acc_s = AverageMeter("Acc (s)", ":3.2f")

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses_all, losses_s, acc_s],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    student.train()

    end = time.time()
    scaler = torch.cuda.amp.GradScaler()

    for i in range(args.iters_per_epoch):
        stu_optimizer.zero_grad()

        x_s, label_s, weight_s, meta_s = next(train_source_iter)
        x_s = x_s.to(device)
        label_s = label_s.to(device)
        weight_s = weight_s.to(device)

        if style_net is not None and args.s2t_freq > np.random.rand():
            with torch.no_grad():
                _, _, _, _ , x_ts, _, _ , _= next(train_target_iter)
                x_t = x_ts[0].to(device)
                _a = np.random.uniform(*args.s2t_alpha)
                x_s = style_net(x_s, x_t, _a)[2]
                x_s = torch.maximum(torch.minimum(x_s.permute(0,2,3,1), recover_max), recover_min).permute(0,3,1,2)
        # measure data loading time
        data_time.update(time.time() - end)

        with torch.cuda.amp.autocast():
            y_s = student(x_s)
            loss_s = criterion(y_s, label_s, weight_s)

        loss_all = loss_s 
        scaler.scale(loss_all).backward()
        scaler.step(stu_optimizer)
        scaler.update()

        _, avg_acc_s, cnt_s, pred_s = accuracy(y_s.detach().cpu().numpy(),
                                               label_s.detach().cpu().numpy())
        acc_s.update(avg_acc_s, cnt_s)
        losses_all.update(loss_all, x_s.size(0))
        losses_s.update(loss_s, x_s.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            if visualize is not None:
                visualize(x_s[0], pred_s[0] * args.image_size / args.heatmap_size, "source_{}_pred.jpg".format(i))
                visualize(x_s[0], meta_s['keypoint2d'][0], "source_{}_label.jpg".format(i))


def train(train_source_iter, train_target_iter, student, teacher, style_net, criterion, con_criterion,
          stu_optimizer, tea_optimizer, epoch: int, visualize, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses_all = AverageMeter('Loss (all)', ":.4e")
    losses_s = AverageMeter('Loss (s)', ":.4e")
    losses_c = AverageMeter('Loss (c)', ":.4e")
    acc_s = AverageMeter("Acc (s)", ":3.2f")

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses_all, losses_s, losses_c, acc_s],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    student.train()
    teacher.train()

    end = time.time()
    scaler = torch.cuda.amp.GradScaler()

    for i in range(args.iters_per_epoch):
        stu_optimizer.zero_grad()

        x_s, label_s, weight_s, meta_s = next(train_source_iter)
        x_t_stu, _, _, meta_t_stu, x_t_teas, _, _, meta_t_tea = next(train_target_iter)

        x_s = x_s.to(device)
        x_s_ori = x_s.clone()
        x_t_stu = x_t_stu.to(device)
        x_t_teas = [x_t_tea.to(device) for x_t_tea in x_t_teas]
        x_t_teas_ori = [x_t_tea.clone() for x_t_tea in x_t_teas]
        label_s = label_s.to(device)
        weight_s = weight_s.to(device)
        label_t = meta_t_stu['target_ori'].to(device)
        weight_t = meta_t_stu['target_weight_ori'].to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        ratio = args.image_size / args.heatmap_size

        with torch.no_grad():
            if style_net is not None and args.s2t_freq > np.random.rand():
                _a = np.random.uniform(*args.s2t_alpha)
                x_s = style_net(x_s, x_t_teas_ori[0], _a)[2]
                x_s = torch.maximum(torch.minimum(x_s.permute(0,2,3,1), recover_max), recover_min).permute(0,3,1,2)

            if style_net is not None and args.t2s_freq > np.random.rand():
                _a = np.random.uniform(*args.t2s_alpha)
                x_t_teas = [style_net(x_t_tea, x_s_ori, _a)[2] for x_t_tea in x_t_teas]
                x_t_teas = [torch.maximum(torch.minimum(x_t_tea.permute(0,2,3,1), recover_max), recover_min).permute(0,3,1,2) for x_t_tea in x_t_teas]

            y_t_teas = [teacher(x_t_tea) for x_t_tea in x_t_teas] # softmax on w, h
            y_t_tea_recon = torch.zeros_like(y_t_teas[0]).cuda() # b, c, h, w
            tea_mask = torch.zeros(y_t_teas[0].shape[:2]).cuda() # b, c
            for ind in range(x_t_teas[0].size(0)):
                recons = torch.zeros(args.k, *y_t_teas[0].size()[1:]) # k, c, h, w
                for _k in range(args.k):
                    angle, [trans_x, trans_y], [shear_x, shear_y], scale = meta_t_tea[_k]['aug_param_tea']
                    _angle, _trans_x, _trans_y, _shear_x, _shear_y, _scale = angle[ind].item(), trans_x[ind].item(), trans_y[ind].item(), shear_x[ind].item(), shear_y[ind].item(), scale[ind].item() 
                    temp = tF.affine(y_t_teas[_k][ind], 0., translate=[_trans_x/ratio, _trans_y/ratio], shear=[0., 0.], scale=1.)
                    temp = tF.affine(temp, _angle, translate=[0., 0.], shear=[0., 0.], scale=_scale)
                    temp = tF.affine(temp, 0., translate=[0, 0], shear=[_shear_x, _shear_y], scale=1.) # c, h, w
                    recons[_k] = temp # c, h, w

                y_t_tea_recon[ind] = torch.mean(recons, dim=0) # (c, h, w)
                tea_mask[ind] = 1.

            angle, [trans_x, trans_y], [shear_x, shear_y], scale = meta_t_stu['aug_param_stu']
            # adaptively occlude keypoints
            if args.occlude_rate > -1: ###
                b, k, h, w = y_t_tea_recon.size()
                conf = y_t_tea_recon.amax(dim=(2,3))

                pred_position = y_t_tea_recon.view(b, k, -1).argmax(-1)
                pred_position = torch.stack([pred_position % w, pred_position // w], -1).cpu().numpy()

                conf_table = conf >= args.occlude_thresh # b, c

                for _b in range(b):
                    if (conf_table[_b].sum() > 0 and np.random.rand() <= args.occlude_rate):
                        _angle, _trans_x, _trans_y, _shear_x, _shear_y, _scale = angle[_b].item(), trans_x[_b].item(), trans_y[_b].item(), shear_x[_b].item(), shear_y[_b].item(), scale[_b].item() 
                        temp = tF.affine(x_t_stu[_b], 0., translate=[_trans_x/ratio, _trans_y/ratio], shear=[0., 0.], scale=1.)
                        temp = tF.affine(temp, _angle, translate=[0., 0.], shear=[0., 0.], scale=_scale)
                        temp = tF.affine(temp, 0., translate=[0., 0.], shear=[_shear_x, _shear_y], scale=1.)

                        # randomly select a point to occlude
                        candidates = torch.arange(0, k)[conf_table[_b]]
                        _c = np.random.choice(candidates)
                        
                        # calculate the occlusion border
                        position = (pred_position[_b, _c] * ratio).astype(np.int) 

                        left = max(position[1] - args.occlude_size, 0)
                        right = min(position[1] + args.occlude_size, args.image_size)
                        upper = max(position[0] - args.occlude_size, 0)
                        bottom = min(position[0] + args.occlude_size, args.image_size)
                        
                        # paste with random patch
                        left_src = np.random.randint(args.image_size - (right - left) + 1)
                        right_src = left_src + right - left
                        upper_src = np.random.randint(args.image_size - (bottom - upper) + 1)
                        bottom_src = upper_src + bottom - upper
                        temp[:, left:right, upper:bottom] = temp[:, left_src:right_src, upper_src:bottom_src]

                        # warp it back
                        x_t_stu[_b] = tF.affine(temp, -_angle, translate=[-_trans_x/ratio, -_trans_y/ratio], shear=[-_shear_x, -_shear_y], scale=1./_scale)
                    
        with torch.cuda.amp.autocast():
            y_s = student(x_s) ###################
            y_t_stu = student(x_t_stu) # softmax on w, h

            y_t_stu_recon = torch.zeros_like(y_t_stu).cuda() # b, c, h, w
            for ind in range(x_t_stu.size(0)):
                _angle, _trans_x, _trans_y, _shear_x, _shear_y, _scale = angle[ind].item(), trans_x[ind].item(), trans_y[ind].item(), shear_x[ind].item(), shear_y[ind].item(), scale[ind].item() 
                temp = tF.affine(y_t_stu[ind], 0., translate=[_trans_x/ratio, _trans_y/ratio], shear=[0., 0.], scale=1.)
                temp = tF.affine(temp, _angle, translate=[0., 0.], shear=[0., 0.], scale=_scale)
                y_t_stu_recon[ind] = tF.affine(temp, 0., translate=[0., 0.], shear=[_shear_x, _shear_y], scale=1.)

            loss_s = criterion(y_s, label_s, weight_s)

            activates = y_t_tea_recon.amax(dim=(2,3))
            y_t_tea_recon = rectify(y_t_tea_recon, sigma=args.sigma)
            mask_thresh = torch.kthvalue(activates.view(-1), int(args.mask_ratio * activates.numel()))[0].item()
            tea_mask = tea_mask * activates>mask_thresh
            
            loss_c = con_criterion(y_t_stu_recon, y_t_tea_recon, tea_mask=tea_mask)

        loss_all = loss_s + args.lambda_c * loss_c

        scaler.scale(loss_all).backward()
        scaler.step(stu_optimizer)
        tea_optimizer.step()

        scaler.update()

        # measure accuracy and record loss
        _, avg_acc_s, cnt_s, pred_s = accuracy(y_s.detach().cpu().numpy(),
                                               label_s.detach().cpu().numpy())
        acc_s.update(avg_acc_s, cnt_s)
        losses_all.update(loss_all, x_s.size(0))
        losses_s.update(loss_s, x_s.size(0))
        losses_c.update(loss_c, x_s.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            if visualize is not None:
                visualize(x_s[0], pred_s[0] * args.image_size / args.heatmap_size, "source_{}_pred.jpg".format(i))
                visualize(x_s[0], meta_s['keypoint2d'][0], "source_{}_label.jpg".format(i))


def validate(val_loader, model, criterion, visualize, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.2e')
    acc = AverageMeterList(list(range(val_loader.dataset.num_keypoints)), ":3.2f",  ignore_val=-1)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses], 
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (x, label, weight, meta) in enumerate(val_loader):
            x = x.to(device)
            label = label.to(device)
            weight = weight.to(device)

            # compute output
            y = model(x)
            loss = criterion(y, label, weight)

            # measure accuracy and record loss
            losses.update(loss.item(), x.size(0))
            acc_per_points, avg_acc, cnt, pred = accuracy(y.cpu().numpy(),
                                                          label.cpu().numpy())
            acc.update(acc_per_points, x.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.val_print_freq == 0:
                progress.display(i)
                if visualize is not None:
                    visualize(x[0], pred[0] * args.image_size / args.heatmap_size, "val_{}_pred.jpg".format(i))
                    visualize(x[0], meta['keypoint2d'][0], "val_{}_label.jpg".format(i))

    return val_loader.dataset.group_accuracy(acc.average())

     


if __name__ == '__main__':
    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )

    parser = argparse.ArgumentParser(description='Source Only for Keypoint Detection Domain Adaptation')
    # dataset parameters
    parser.add_argument('source_root', help='root path of the source dataset')
    parser.add_argument('target_root', help='root path of the target dataset')
    parser.add_argument('-s', '--source', help='source domain(s)')
    parser.add_argument('-t', '--target', help='target domain(s)')
    parser.add_argument('--target-train', help='target domain(s)')
    parser.add_argument('--resize-scale', nargs='+', type=float, default=(0.6, 1.3),
                        help='scale range for the RandomResizeCrop augmentation')
    parser.add_argument('--image-size', type=int, default=256,
                        help='input image size')
    parser.add_argument('--heatmap-size', type=int, default=64,
                        help='output heatmap size')
    parser.add_argument('--sigma', type=int, default=2,
                        help='')
    parser.add_argument('--k', type=int, default=1,
                        help='')

    # augmentation
    parser.add_argument('--rotation_stu', type=int, default=180,
                        help='rotation range of the RandomRotation augmentation')
    parser.add_argument('--color_stu', type=float, default=0.25,
                        help='color range of the jitter augmentation')
    parser.add_argument('--blur_stu', type=float, default=0,
                        help='blur range of the jitter augmentation')
    parser.add_argument('--shear_stu', nargs='+', type=float, default=(-30, 30),
                        help='shear range for the RandomResizeCrop augmentation')
    parser.add_argument('--translate_stu', nargs='+', type=float, default=(0.05, 0.05),
                        help='tranlate range for the RandomResizeCrop augmentation')
    parser.add_argument('--scale_stu', nargs='+', type=float, default=(0.6, 1.3),
                        help='scale range for the RandomResizeCrop augmentation')
    parser.add_argument('--rotation_tea', type=int, default=180,
                        help='rotation range of the RandomRotation augmentation')
    parser.add_argument('--color_tea', type=float, default=0.25,
                        help='color range of the jitter augmentation')
    parser.add_argument('--blur_tea', type=float, default=0,
                        help='blur range of the jitter augmentation')
    parser.add_argument('--shear_tea', nargs='+', type=float, default=(-30, 30),
                        help='shear range for the RandomResizeCrop augmentation')
    parser.add_argument('--translate_tea', nargs='+', type=float, default=(0.05, 0.05),
                        help='tranlate range for the RandomResizeCrop augmentation')
    parser.add_argument('--scale_tea', nargs='+', type=float, default=(0.6, 1.3),
                        help='scale range for the RandomResizeCrop augmentation')
    parser.add_argument('--s2t-freq', type=float, default=0.5)
    parser.add_argument('--s2t-alpha', nargs='+', type=float, default=(0, 1))
    parser.add_argument('--t2s-freq', type=float, default=0.5)
    parser.add_argument('--t2s-alpha', nargs='+', type=float, default=(0, 1))

    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='pose_resnet101',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: pose_resnet101)')

    parser.add_argument("--resume", type=str, default=None,
                        help="where restore model parameters from.")
    parser.add_argument("--pretrain", type=str, default=None,
                        help="where restore model parameters from.")
    parser.add_argument("--decoder-name", type=str, default=None,
                        help="where restore style_net model parameters from.")

    # training parameters
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--test-batch', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lambda_c', default=1., type=float)
    parser.add_argument('--teacher_alpha', default=0.999, type=float)
    parser.add_argument('--lr-step', default=[45, 60], type=tuple, help='parameter for lr scheduler')
    parser.add_argument('--lr-factor', default=0.1, type=float, help='parameter for lr scheduler')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=70, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--val-print-freq', default=2000, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--log", type=str, default='src_only',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test'],
                        help="When phase is 'test', only test the model.")
    parser.add_argument('--debug', action="store_true",
                        help='In the debug mode, save images and predictions')
    parser.add_argument('--mask-ratio', type=float, default=0.5,
                        help='')
    parser.add_argument('--SGD', action="store_true",
                        help='')
    parser.add_argument('--pretrain-epoch', type=int, default=-1,
                        help='pretrain-epoch')
    parser.add_argument('--occlude-rate', type=float, default=0.5)
    parser.add_argument('--occlude-thresh', type=float, default=0.9,
                        help='')
    parser.add_argument('--occlude-size', type=int, default=10,
                        help='')

    args = parser.parse_args()
    main(args)

