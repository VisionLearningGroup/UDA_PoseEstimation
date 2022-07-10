# ------------------------------------------------------------------------------
# Copyright (c) Southeast University. Licensed under the MIT License.
# Written by Sen Yang (yangsenius@seu.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from collections import OrderedDict

import copy
from typing import Optional, List


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


__all__ = ['get_pose_net_trans', 'TransPoseR', 'TransPoseR_layer4', 'get_pose_net_trans50']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, base_width=64):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers,
                 norm=None, pe_only_at_begin=False, return_atten_map=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.pe_only_at_begin = pe_only_at_begin
        self.return_atten_map = return_atten_map
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                target_list: Optional[list] = None,
                return_output_list: Optional[bool] = False):
        output = src
        atten_maps_list = []

        output_list = [output]
        if target_list is None:
            target_list = [None] * len(self.layers)

        for index, layer in enumerate(self.layers):
            if self.return_atten_map:
                output, att_map = layer(output, src_mask=mask, pos=pos,
                                        src_key_padding_mask=src_key_padding_mask, target=target_list[index])
                atten_maps_list.append(att_map)
            else:
                output = layer(output, src_mask=mask,  pos=pos,
                               src_key_padding_mask=src_key_padding_mask, target=target_list[index])

            if return_output_list:
                output_list.append(output)

            # only add position embedding to the first atttention layer
            pos = None if self.pe_only_at_begin else pos

        if self.norm is not None:
            output = self.norm(output)

        if self.return_atten_map:
            return output, torch.stack(atten_maps_list)
        elif return_output_list:
            return output, torch.stack(output_list)
        else:
            return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerEncoderLayer(nn.Module):
    """ Modified from https://github.com/facebookresearch/detr/blob/master/models/transformer.py"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, return_atten_map=False, target_value=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.return_atten_map = return_atten_map

        self.target_value = target_value

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     target: Optional[Tensor] = None):

        if target is None:
            q = k = self.with_pos_embed(src, pos)
        else:
            q = self.with_pos_embed(src, pos)
            k = self.with_pos_embed(target, pos)

        if self.return_atten_map:
            if self.target_value:
                src2, att_map = self.self_attn(q, k, value=target,
                                               attn_mask=src_mask,
                                               key_padding_mask=src_key_padding_mask)

            else:
                src2, att_map = self.self_attn(q, k, value=src,
                                           attn_mask=src_mask,
                                           key_padding_mask=src_key_padding_mask)
        else:
            if self.target_value:
                src2 = self.self_attn(q, k, value=target, attn_mask=src_mask,
                                      key_padding_mask=src_key_padding_mask)[0]
            else:
                src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if self.return_atten_map:
            return src, att_map
        else:
            return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    target: Optional[Tensor] = None):
        src2 = self.norm1(src)
        if target is None:
            q = k = self.with_pos_embed(src2, pos)
        else:
            target2 = self.norm1(target)
            q = self.with_pos_embed(src2, pos)
            k = self.with_pos_embed(target2, pos)

        if self.return_atten_map:
            src2, att_map = self.self_attn(q, k, value=src,
                                           attn_mask=src_mask,
                                           key_padding_mask=src_key_padding_mask)
        else:
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        if self.return_atten_map:
            return src, att_map
        else:
            return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                target: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos, target)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos, target)


class TransPoseR(nn.Module):

    def __init__(self, block, layers, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS

        super(TransPoseR, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.cfg = cfg
        d_model = cfg.MODEL.DIM_MODEL
        self.d_model = d_model
        dim_feedforward = cfg.MODEL.DIM_FEEDFORWARD
        encoder_layers_num = cfg.MODEL.ENCODER_LAYERS
        n_head = cfg.MODEL.N_HEAD
        pos_embedding_type = cfg.MODEL.POS_EMBEDDING
        w, h = cfg.MODEL.IMAGE_SIZE

        self.reduce = nn.Conv2d(self.inplanes, d_model, 1, bias=False)
        self._make_position_embedding(w, h, d_model, pos_embedding_type)

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            activation='relu',
            return_atten_map=False,
            target_value=kwargs['args'].target_value
        )
        self.global_encoder = TransformerEncoder(
            encoder_layer,
            encoder_layers_num,
            return_atten_map=False
        )

        # used for deconv layers
        self.inplanes = d_model
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,   # 1
            extra.NUM_DECONV_FILTERS,  # [d_model]
            extra.NUM_DECONV_KERNELS,  # [4]
        )

        self.deconv_layers_layer3 = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS-1,  # 1
            extra.NUM_DECONV_FILTERS[:-1],  # [d_model]
            extra.NUM_DECONV_KERNELS[:-1],  # [4]
        )

        self.deconv_layers_after = self._make_deconv_layer(
            1,  # 1
            extra.NUM_DECONV_FILTERS[:1],  # [d_model]
            extra.NUM_DECONV_KERNELS[:1],  # [4]
        )

        self.final_layer = nn.Conv2d(
            in_channels=d_model,
            out_channels=kwargs['num_keypoints'],
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        assert pe_type in ['none', 'learnable', 'sine']
        if pe_type == 'none':
            self.pos_embedding = None
            logger.info("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h // 8
                self.pe_w = w // 8
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(
                    torch.randn(length, 1, d_model))
                logger.info("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                logger.info("==> Add Sine PositionEmbedding~")

    def _make_sine_position_embedding(self, d_model, temperature=10000,
                                      scale=2*math.pi):
        # logger.info(">> NOTE: this is for testing on unseen input resolutions")
        # # NOTE generalization test with interploation
        # self.pe_h, self.pe_w = 256 // 8 , 192 // 8 #self.pe_h, self.pe_w
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(2, 0, 1)
        return pos  # [h*w, 1, d_model]

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)



    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 16, 64, 64, 64
        x = self.layer1(x)
        # 16, 256, 64, 64
        x = self.layer2(x)
        # 16, 512, 32, 32
        x = self.reduce(x)
        # 16, 256, 32, 32
        bs, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        # 1024 (32*32), 16, 256
        x = self.global_encoder(x, pos=self.pos_embedding)
        # 1024 (32*32), 16 (batch), 256 (channel)
        x = x.permute(1, 2, 0).contiguous().view(bs, c, h, w)
        # 16, 256, 32, 32
        x = self.deconv_layers(x)
        # 16, 256, 64, 64
        x = self.final_layer(x)
        # 16, 21, 64, 64

        return x

    def init_weights(self, pretrained=''):
        # if os.path.isfile(pretrained):
        logger.info('=> init final conv weights from normal distribution')
        for name, m in self.final_layer.named_modules():
            if isinstance(m, nn.Conv2d):
                logger.info(
                    '=> init {}.weight as normal(0, 0.001)'.format(name))
                logger.info('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)


            # pretrained_state_dict = torch.load(pretrained)
            # logger.info('=> loading pretrained model {}'.format(pretrained))
            # existing_state_dict = {}
            # for name, m in pretrained_state_dict.items():
            #     if name in self.state_dict():
            #         existing_state_dict[name] = m
            #         print(":: {} is loaded from {}".format(name, pretrained))
            # self.load_state_dict(existing_state_dict, strict=False)
        # else:
        #     logger.info(
        #         '=> NOTE :: ImageNet Pretrained Weights {} are not loaded ! Please Download it'.format(pretrained))
        #     logger.info('=> init weights from normal distribution')
        #     for m in self.modules():
        #         if isinstance(m, nn.Conv2d):
        #             nn.init.normal_(m.weight, std=0.001)
        #         elif isinstance(m, nn.BatchNorm2d):
        #             nn.init.constant_(m.weight, 1)
        #             nn.init.constant_(m.bias, 0)
        #         elif isinstance(m, nn.ConvTranspose2d):
        #             nn.init.normal_(m.weight, std=0.001)
        #             if self.deconv_with_bias:
        #                 nn.init.constant_(m.bias, 0)

    def init_weights_imagenet(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init final conv weights from normal distribution')
            for name, m in self.final_layer.named_modules():
                if isinstance(m, nn.Conv2d):
                    logger.info(
                        '=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)


                pretrained_state_dict = torch.load(pretrained)
                logger.info('=> loading pretrained model {}'.format(pretrained))
                existing_state_dict = {}
                for name, m in pretrained_state_dict.items():
                    if name in self.state_dict():
                        existing_state_dict[name] = m
                        print(":: {} is loaded from {}".format(name, pretrained))
                self.load_state_dict(existing_state_dict, strict=False)
        else:
            logger.info(
                '=> NOTE :: ImageNet Pretrained Weights {} are not loaded ! Please Download it'.format(pretrained))
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)


    def get_parameters(self, lr=1.):

        params_dict = [
            {'params': self.conv1.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.bn1.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.layer1.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.layer2.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.layer3.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.layer4.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.reduce.parameters(), 'lr': lr},
            {'params': self.global_encoder.parameters(), 'lr': lr},
            {'params': self.deconv_layers.parameters(), 'lr': lr},
            {'params': self.final_layer.parameters(), 'lr': lr},
        ]

        if self.pos_embedding is not None:
            params_dict.append({'params': self.pos_embedding, 'lr': lr})

        return params_dict




class TransPoseR_layer4(TransPoseR):

    def __init__(self, block, layers, cfg, resnet, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        super(TransPoseR_layer4, self).__init__(block, layers, cfg, **kwargs)

        # self.layer1 = self._make_layer(block, 64, layers[2])
        # self.layer2 = self._make_layer(block, 128, layers[3], stride=2)
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.reduce = nn.Conv2d(2048, self.d_model, 1, bias=False)
        self.base_width = 64


    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        assert pe_type in ['none', 'learnable', 'sine']
        if pe_type == 'none':
            self.pos_embedding = None
            logger.info("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h // 32
                self.pe_w = w // 32
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(
                    torch.randn(length, 1, d_model))
                logger.info("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                logger.info("==> Add Sine PositionEmbedding~")


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 16, 64, 64, 64
        x = self.layer1(x)
        # 16, 256, 64, 64
        x = self.layer2(x)
        # 16, 512, 32, 32
        x = self.layer3(x)
        # 16, 1024, 16, 16
        x = self.layer4(x)
        # 16, 2048, 8, 8
        x = self.reduce(x)
        # 16, 256, 8, 8

        bs, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        x = self.global_encoder(x, pos=self.pos_embedding)
        x = x.permute(1, 2, 0).contiguous().view(bs, c, h, w)
        # 16, 256, 8, 8
        x = self.deconv_layers(x)
        # 16, 256, 16, 16
        x = self.final_layer(x)

        return x


    # def get_parameters(self, lr=1.):
    #
    #     return [
    #         {'params': self.conv1.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
    #         {'params': self.bn1.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
    #         {'params': self.layer1.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
    #         {'params': self.layer2.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
    #         {'params': self.layer3.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
    #         {'params': self.layer4.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
    #         {'params': self.reduce.parameters(), 'lr': lr},
    #         {'params': self.pos_embedding, 'lr': lr},
    #         {'params': self.global_encoder.parameters(), 'lr': lr},
    #         {'params': self.deconv_layers.parameters(), 'lr': lr},
    #         {'params': self.final_layer.parameters(), 'lr': lr},
    #     ]

    def get_parameters(self, lr=1.):

        params_dict = [
            {'params': self.conv1.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.bn1.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.layer1.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.layer2.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.layer3.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.layer4.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.reduce.parameters(), 'lr': lr},
            {'params': self.global_encoder.parameters(), 'lr': lr},
            {'params': self.deconv_layers.parameters(), 'lr': lr},
            {'params': self.final_layer.parameters(), 'lr': lr},
        ]

        if self.pos_embedding is not None:
            params_dict.append({'params': self.pos_embedding, 'lr': lr})

        return params_dict


class projection_MLP(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        hidden_dim = in_dim
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class TransPoseR_layer4_cross(TransPoseR):

    def __init__(self, block, layers, cfg, resnet, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        super(TransPoseR_layer4_cross, self).__init__(block, layers, cfg, **kwargs)

        # self.layer1 = self._make_layer(block, 64, layers[2])
        # self.layer2 = self._make_layer(block, 128, layers[3], stride=2)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        # self.relu = nn.ReLU(inplace=True)


        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.reduce = nn.Conv2d(2048, self.d_model, 1, bias=False)
        self.base_width = 64



    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        assert pe_type in ['none', 'learnable', 'sine']
        if pe_type == 'none':
            self.pos_embedding = None
            logger.info("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h // 32
                self.pe_w = w // 32
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(
                    torch.randn(length, 1, d_model))
                logger.info("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                logger.info("==> Add Sine PositionEmbedding~")


    def forward(self, x, cross=False, return_feat=False, detach_all=False, detach_q=False, detach_k=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 16, 64, 64, 64
        x = self.layer1(x)
        # 16, 256, 64, 64
        x = self.layer2(x)
        # 16, 512, 32, 32
        x = self.layer3(x)
        # 16, 1024, 16, 16
        x = self.layer4(x)
        # 16, 2048, 8, 8
        x = self.reduce(x)
        # 16, 256, 8, 8

        bs, c, h, w = x.shape
        self.bs, self.c, self.h, self.w = bs, c, h, w

        x_feat = x.flatten(2).permute(2, 0, 1)
        ## (# pixel x batch x dim)

        x = self.global_encoder(x_feat, pos=self.pos_embedding, target_list=[x_feat])

        if cross:
            rand_idx = torch.randperm(bs)
            x_cross_feat = x_feat[:, rand_idx, :]

            if detach_all:
                x_cross_feat = self.global_encoder(x_feat.detach(), pos=self.pos_embedding, target_list=[x_cross_feat.detach()])
            elif detach_q:
                x_cross_feat = self.global_encoder(x_feat.detach(), pos=self.pos_embedding,
                                                   target_list=[x_cross_feat])
            elif detach_k:
                x_cross_feat = self.global_encoder(x_feat, pos=self.pos_embedding,
                                                   target_list=[x_cross_feat.detach()])
            else:
                x_cross_feat = self.global_encoder(x_feat, pos=self.pos_embedding, target_list=[x_cross_feat])

            x_cross_feat = x_cross_feat.permute(1, 2, 0).contiguous().view(bs, c, h, w)
            # 16, 256, 8, 8
            x_cross_feat = self.deconv_layers(x_cross_feat)
            # 16, 256, 16, 16
            x_cross_feat = self.final_layer(x_cross_feat)

        x = x.permute(1, 2, 0).contiguous().view(bs, c, h, w)
        # 16, 256, 8, 8
        x = self.deconv_layers(x)
        # 16, 256, 16, 16
        x = self.final_layer(x)

        if cross:
            if return_feat:
                return x, x_cross_feat, x_feat
            else:
                return x, x_cross_feat

        if return_feat:
            return x, x_feat

        else:
            return x


    def forward_cross(self, x, x2):

        x = torch.cat((x, x2))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 16, 64, 64, 64
        x = self.layer1(x)
        # 16, 256, 64, 64
        x = self.layer2(x)
        # 16, 512, 32, 32
        x = self.layer3(x)
        # 16, 1024, 16, 16
        x = self.layer4(x)
        # 16, 2048, 8, 8
        x = self.reduce(x)

        x = torch.split(x, x.shape[0]//2, dim=0)
        x, x2 = x[0], x[1]
        bs, c, h, w = x.shape
        x_feat = x.flatten(2).permute(2, 0, 1)
        x2_feat = x2.flatten(2).permute(2, 0, 1)
        x_cross_feat = self.global_encoder(x_feat, pos=self.pos_embedding, target_list=[x2_feat])


    def forward_with_cross_feat(self, feat_s, feat_t):

        x_cross_feat = self.global_encoder(feat_s, pos=self.pos_embedding, target_list=[feat_t])

        x_cross_feat = x_cross_feat.permute(1, 2, 0).contiguous().view(self.bs, self.c, self.h, self.w)
        # 16, 256, 8, 8
        x_cross_feat = self.deconv_layers(x_cross_feat)
        # 16, 256, 16, 16
        x_cross_feat = self.final_layer(x_cross_feat)
        return x_cross_feat





class TransPoseR_layer4_cross_contrastive(TransPoseR):

    def __init__(self, block, layers, cfg, resnet, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        super(TransPoseR_layer4_cross_contrastive, self).__init__(block, layers, cfg, **kwargs)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.reduce = nn.Conv2d(2048, self.d_model, 1, bias=False)
        self.base_width = 64
        self.projection = projection_MLP(in_dim=self.d_model, out_dim=self.d_model)


    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        assert pe_type in ['none', 'learnable', 'sine']
        if pe_type == 'none':
            self.pos_embedding = None
            logger.info("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h // 32
                self.pe_w = w // 32
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(
                    torch.randn(length, 1, d_model))
                logger.info("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                logger.info("==> Add Sine PositionEmbedding~")


    def forward(self, x, cross=False, return_feat=False, detach_all=False, detach_q=False, detach_k=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 16, 64, 64, 64
        x = self.layer1(x)
        # 16, 256, 64, 64
        x = self.layer2(x)
        # 16, 512, 32, 32
        x = self.layer3(x)
        # 16, 1024, 16, 16
        x = self.layer4(x)
        # 16, 2048, 8, 8
        x = self.reduce(x)

        projection_feat = x.flatten(-2)
        projection_feat = projection_feat.permute(0, 2, 1)
        projection_feat = self.projection(projection_feat)

        # 16, 256, 8, 8

        bs, c, h, w = x.shape
        self.bs, self.c, self.h, self.w = bs, c, h, w

        x_feat = x.flatten(2).permute(2, 0, 1)
        ## (# pixel x batch x dim)

        x = self.global_encoder(x_feat, pos=self.pos_embedding, target_list=[x_feat])

        if cross:
            rand_idx = torch.randperm(bs)
            x_cross_feat = x_feat[:, rand_idx, :]

            if detach_all:
                x_cross_feat = self.global_encoder(x_feat.detach(), pos=self.pos_embedding, target_list=[x_cross_feat.detach()])
            elif detach_q:
                x_cross_feat = self.global_encoder(x_feat.detach(), pos=self.pos_embedding,
                                                   target_list=[x_cross_feat])
            elif detach_k:
                x_cross_feat = self.global_encoder(x_feat, pos=self.pos_embedding,
                                                   target_list=[x_cross_feat.detach()])
            else:
                x_cross_feat = self.global_encoder(x_feat, pos=self.pos_embedding, target_list=[x_cross_feat])

            x_cross_feat = x_cross_feat.permute(1, 2, 0).contiguous().view(bs, c, h, w)
            # 16, 256, 8, 8
            x_cross_feat = self.deconv_layers(x_cross_feat)
            # 16, 256, 16, 16
            x_cross_feat = self.final_layer(x_cross_feat)

        x = x.permute(1, 2, 0).contiguous().view(bs, c, h, w)
        # 16, 256, 8, 8
        x = self.deconv_layers(x)
        # 16, 256, 16, 16
        x = self.final_layer(x)

        if cross:
            if return_feat:
                return x, x_cross_feat, x_feat, projection_feat
            else:
                return x, x_cross_feat, projection_feat

        if return_feat:
            return x, x_feat, projection_feat

        else:
            return x, projection_feat


    def forward_cross(self, x, x2):

        x = torch.cat((x, x2))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 16, 64, 64, 64
        x = self.layer1(x)
        # 16, 256, 64, 64
        x = self.layer2(x)
        # 16, 512, 32, 32
        x = self.layer3(x)
        # 16, 1024, 16, 16
        x = self.layer4(x)
        # 16, 2048, 8, 8
        x = self.reduce(x)

        x = torch.split(x, x.shape[0]//2, dim=0)
        x, x2 = x[0], x[1]
        bs, c, h, w = x.shape
        x_feat = x.flatten(2).permute(2, 0, 1)
        x2_feat = x2.flatten(2).permute(2, 0, 1)
        x_cross_feat = self.global_encoder(x_feat, pos=self.pos_embedding, target_list=[x2_feat])


    def forward_with_cross_feat(self, feat_s, feat_t):

        x_cross_feat = self.global_encoder(feat_s, pos=self.pos_embedding, target_list=[feat_t])

        x_cross_feat = x_cross_feat.permute(1, 2, 0).contiguous().view(self.bs, self.c, self.h, self.w)
        # 16, 256, 8, 8
        x_cross_feat = self.deconv_layers(x_cross_feat)
        # 16, 256, 16, 16
        x_cross_feat = self.final_layer(x_cross_feat)
        return x_cross_feat


    def consistency_loss(self, feat_s, feat_t):

        x_feat = self.global_encoder(feat_s, pos=self.pos_embedding, target_list=[feat_s])
        x_feat = x_feat.permute(1, 2, 0).contiguous().view(self.bs, self.c, self.h, self.w)
        x_cross_feat = self.global_encoder(feat_s, pos=self.pos_embedding, target_list=[feat_t])
        x_cross_feat = x_cross_feat.permute(1, 2, 0).contiguous().view(self.bs, self.c, self.h, self.w)

        # loss = F.mse_loss(x_feat.detach(), x_cross_feat)
        return x_feat, x_cross_feat

    def get_parameters(self, lr=1.):

        params_dict = [
            {'params': self.conv1.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.bn1.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.layer1.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.layer2.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.layer3.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.layer4.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.reduce.parameters(), 'lr': lr},
            {'params': self.projection.parameters(), 'lr': lr},
            {'params': self.global_encoder.parameters(), 'lr': lr},
            {'params': self.deconv_layers.parameters(), 'lr': lr},
            {'params': self.final_layer.parameters(), 'lr': lr},
        ]

        if self.pos_embedding is not None:
            params_dict.append({'params': self.pos_embedding, 'lr': lr})

        return params_dict



class TransPoseR_layer4_cross_multiple_encoder(TransPoseR):

    def __init__(self, block, layers, cfg, resnet, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        super(TransPoseR_layer4_cross_multiple_encoder, self).__init__(block, layers, cfg, **kwargs)

        # self.layer1 = self._make_layer(block, 64, layers[2])
        # self.layer2 = self._make_layer(block, 128, layers[3], stride=2)
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.reduce = nn.Conv2d(2048, self.d_model, 1, bias=False)
        self.base_width = 64



    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        assert pe_type in ['none', 'learnable', 'sine']
        if pe_type == 'none':
            self.pos_embedding = None
            logger.info("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h // 32
                self.pe_w = w // 32
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(
                    torch.randn(length, 1, d_model))
                logger.info("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                logger.info("==> Add Sine PositionEmbedding~")


    def forward(self, x, cross=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 16, 64, 64, 64
        x = self.layer1(x)
        # 16, 256, 64, 64
        x = self.layer2(x)
        # 16, 512, 32, 32
        x = self.layer3(x)
        # 16, 1024, 16, 16
        x = self.layer4(x)
        # 16, 2048, 8, 8
        x = self.reduce(x)
        # 16, 256, 8, 8

        bs, c, h, w = x.shape
        x_feat = x.flatten(2).permute(2, 0, 1)
        ## (# pixel x batch x dim)

        x, x_output_list = self.global_encoder(x_feat, pos=self.pos_embedding, return_output_list=True)
        # x_output_list = x_output_list[:-1, ...]1

        if cross:
            rand_idx = torch.randperm(bs)
            x_output_cross_feat = x_output_list[:, :, rand_idx, :]
            x_cross_feat = self.global_encoder(x_feat, pos=self.pos_embedding, target_list=x_output_cross_feat)

            x_cross_feat = x_cross_feat.permute(1, 2, 0).contiguous().view(bs, c, h, w)
            # 16, 256, 8, 8
            x_cross_feat = self.deconv_layers(x_cross_feat)
            # 16, 256, 16, 16
            x_cross_feat = self.final_layer(x_cross_feat)

        x = x.permute(1, 2, 0).contiguous().view(bs, c, h, w)
        # 16, 256, 8, 8
        x = self.deconv_layers(x)
        # 16, 256, 16, 16
        x = self.final_layer(x)

        if cross:
            return x, x_cross_feat
        else:
            return x




class TransPoseR_layer3_layer4(TransPoseR):

    def __init__(self, block, layers, cfg, resnet, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        super(TransPoseR_layer3_layer4, self).__init__(block, layers, cfg, **kwargs)

        # self.layer1 = self._make_layer(block, 64, layers[2])
        # self.layer2 = self._make_layer(block, 128, layers[3], stride=2)
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.reduce = nn.Conv2d(2048, self.d_model, 1, bias=False)
        self.reduce_layer3 = nn.Conv2d(1024, self.d_model, 1, bias=False)

        self.base_width = 64

        d_model = cfg.MODEL.DIM_MODEL
        self.d_model = d_model
        dim_feedforward = cfg.MODEL.DIM_FEEDFORWARD
        encoder_layers_num = cfg.MODEL.ENCODER_LAYERS
        n_head = cfg.MODEL.N_HEAD
        pos_embedding_type = cfg.MODEL.POS_EMBEDDING
        w, h = cfg.MODEL.IMAGE_SIZE

        self._make_position_embedding_layer3(w, h, d_model, pos_embedding_type)

        encoder_layer3 = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward*4,
            activation='relu',
            return_atten_map=False
        )
        self.global_encoder_layer3 = TransformerEncoder(
            encoder_layer3,
            encoder_layers_num,
            return_atten_map=False
        )

        self.final_layer = nn.Conv2d(
            in_channels=d_model*2,
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )


    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        assert pe_type in ['none', 'learnable', 'sine']
        if pe_type == 'none':
            self.pos_embedding = None
            logger.info("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h // 32
                self.pe_w = w // 32
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(
                    torch.randn(length, 1, d_model))
                logger.info("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                logger.info("==> Add Sine PositionEmbedding~")

    def _make_position_embedding_layer3(self, w, h, d_model, pe_type='sine'):
        assert pe_type in ['none', 'learnable', 'sine']
        if pe_type == 'none':
            self.pos_embedding_layer3 = None
            logger.info("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h // 16
                self.pe_w = w // 16
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding_layer3 = nn.Parameter(
                    torch.randn(length, 1, d_model))
                logger.info("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding_layer3 = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                logger.info("==> Add Sine PositionEmbedding~")


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 16, 64, 64, 64
        x = self.layer1(x)
        # 16, 256, 64, 64
        x = self.layer2(x)
        # 16, 512, 32, 32
        x = self.layer3(x)

        layer3  = self.reduce_layer3(x)
        # 16, 1024, 16, 16
        x = self.layer4(x)
        # 16, 2048, 8, 8
        x = self.reduce(x)
        # 16, 256, 8, 8

        bs, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        x = self.global_encoder(x, pos=self.pos_embedding)
        x = x.permute(1, 2, 0).contiguous().view(bs, c, h, w)
        # 16, 256, 8, 8
        x = self.deconv_layers(x)

        ##

        bs, c, h, w = layer3.shape
        layer3 = layer3.flatten(2).permute(2, 0, 1)
        layer3 = self.global_encoder_layer3(layer3, pos=self.pos_embedding_layer3)
        layer3 = layer3.permute(1, 2, 0).contiguous().view(bs, c, h, w)
        # 16, 256, 8, 8
        layer3 = self.deconv_layers_layer3(layer3)

        ##
        concat = torch.cat((x, layer3), dim=1)
        # 16, 256, 16, 16
        x = self.final_layer(concat)

        return x


    def get_parameters(self, lr=1.):

        return [
            {'params': self.conv1.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.bn1.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.layer1.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.layer2.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.layer3.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.layer4.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.reduce.parameters(), 'lr': lr},
            {'params': self.reduce_layer3.parameters(), 'lr': lr},
            {'params': self.pos_embedding, 'lr': lr},
            {'params': self.pos_embedding_layer3, 'lr': lr},
            {'params': self.global_encoder.parameters(), 'lr': lr},
            {'params': self.global_encoder_layer3.parameters(), 'lr': lr},
            {'params': self.deconv_layers.parameters(), 'lr': lr},
            {'params': self.deconv_layers_layer3.parameters(), 'lr': lr},
            {'params': self.final_layer.parameters(), 'lr': lr},
        ]



class TransPoseR_layer3_layer4_add(TransPoseR):

    def __init__(self, block, layers, cfg, resnet, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        super(TransPoseR_layer3_layer4_add, self).__init__(block, layers, cfg, **kwargs)

        # self.layer1 = self._make_layer(block, 64, layers[2])
        # self.layer2 = self._make_layer(block, 128, layers[3], stride=2)
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.reduce = nn.Conv2d(2048, self.d_model, 1, bias=False)
        self.reduce_layer3 = nn.Conv2d(1024, self.d_model, 1, bias=False)

        self.base_width = 64

        d_model = cfg.MODEL.DIM_MODEL
        self.d_model = d_model
        dim_feedforward = cfg.MODEL.DIM_FEEDFORWARD
        encoder_layers_num = cfg.MODEL.ENCODER_LAYERS
        n_head = cfg.MODEL.N_HEAD
        pos_embedding_type = cfg.MODEL.POS_EMBEDDING
        w, h = cfg.MODEL.IMAGE_SIZE

        self._make_position_embedding_layer3(w, h, d_model, pos_embedding_type)

        encoder_layer3 = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward*4,
            activation='relu',
            return_atten_map=False
        )
        self.global_encoder_layer3 = TransformerEncoder(
            encoder_layer3,
            encoder_layers_num,
            return_atten_map=False
        )

        self.final_layer = nn.Conv2d(
            in_channels=d_model,
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )


    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        assert pe_type in ['none', 'learnable', 'sine']
        if pe_type == 'none':
            self.pos_embedding = None
            logger.info("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h // 32
                self.pe_w = w // 32
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(
                    torch.randn(length, 1, d_model))
                logger.info("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                logger.info("==> Add Sine PositionEmbedding~")

    def _make_position_embedding_layer3(self, w, h, d_model, pe_type='sine'):
        assert pe_type in ['none', 'learnable', 'sine']
        if pe_type == 'none':
            self.pos_embedding_layer3 = None
            logger.info("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h // 16
                self.pe_w = w // 16
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding_layer3 = nn.Parameter(
                    torch.randn(length, 1, d_model))
                logger.info("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding_layer3 = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                logger.info("==> Add Sine PositionEmbedding~")


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 16, 64, 64, 64
        x = self.layer1(x)
        # 16, 256, 64, 64
        x = self.layer2(x)
        # 16, 512, 32, 32
        x = self.layer3(x)

        layer3  = self.reduce_layer3(x)
        # 16, 1024, 16, 16
        x = self.layer4(x)
        # 16, 2048, 8, 8
        x = self.reduce(x)
        # 16, 256, 8, 8

        bs, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        x = self.global_encoder(x, pos=self.pos_embedding)
        x = x.permute(1, 2, 0).contiguous().view(bs, c, h, w)
        # 16, 256, 8, 8
        x = self.deconv_layers(x)

        ##

        bs, c, h, w = layer3.shape
        layer3 = layer3.flatten(2).permute(2, 0, 1)
        layer3 = self.global_encoder_layer3(layer3, pos=self.pos_embedding_layer3)
        layer3 = layer3.permute(1, 2, 0).contiguous().view(bs, c, h, w)
        # 16, 256, 8, 8
        layer3 = self.deconv_layers_layer3(layer3)

        ##
        concat = x + layer3
        # 16, 256, 16, 16
        x = self.final_layer(concat)

        return x


    def get_parameters(self, lr=1.):

        return [
            {'params': self.conv1.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.bn1.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.layer1.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.layer2.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.layer3.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.layer4.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.reduce.parameters(), 'lr': lr},
            {'params': self.reduce_layer3.parameters(), 'lr': lr},
            {'params': self.pos_embedding, 'lr': lr},
            {'params': self.pos_embedding_layer3, 'lr': lr},
            {'params': self.global_encoder.parameters(), 'lr': lr},
            {'params': self.global_encoder_layer3.parameters(), 'lr': lr},
            {'params': self.deconv_layers.parameters(), 'lr': lr},
            {'params': self.deconv_layers_layer3.parameters(), 'lr': lr},
            {'params': self.final_layer.parameters(), 'lr': lr},
        ]


class TransPoseR_layer4_residual(TransPoseR):

    def __init__(self, block, layers, cfg, resnet, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        super(TransPoseR_layer4_residual, self).__init__(block, layers, cfg, **kwargs)

        # self.layer1 = self._make_layer(block, 64, layers[2])
        # self.layer2 = self._make_layer(block, 128, layers[3], stride=2)
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.reduce = nn.Conv2d(2048, self.d_model, 1, bias=False)
        self.base_width = 64


    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        assert pe_type in ['none', 'learnable', 'sine']
        if pe_type == 'none':
            self.pos_embedding = None
            logger.info("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h // 32
                self.pe_w = w // 32
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(
                    torch.randn(length, 1, d_model))
                logger.info("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                logger.info("==> Add Sine PositionEmbedding~")


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 16, 64, 64, 64
        x = self.layer1(x)
        # 16, 256, 64, 64
        x = self.layer2(x)
        # 16, 512, 32, 32
        x = self.layer3(x)
        # 16, 1024, 16, 16
        x = self.layer4(x)
        # 16, 2048, 8, 8
        reduced = self.reduce(x)
        # 16, 256, 8, 8

        bs, c, h, w = reduced.shape
        x = reduced.flatten(2).permute(2, 0, 1)
        x = self.global_encoder(x, pos=self.pos_embedding)
        x = x.permute(1, 2, 0).contiguous().view(bs, c, h, w)
        # 16, 256, 8, 8
        x = x + reduced
        x = F.relu(x)
        x = self.deconv_layers(x)
        # 16, 256, 16, 16
        x = self.final_layer(x)

        return x


    def get_parameters(self, lr=1.):

        return [
            {'params': self.conv1.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.bn1.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.layer1.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.layer2.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.layer3.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.layer4.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.reduce.parameters(), 'lr': lr},
            {'params': self.pos_embedding, 'lr': lr},
            {'params': self.global_encoder.parameters(), 'lr': lr},
            {'params': self.deconv_layers.parameters(), 'lr': lr},
            {'params': self.final_layer.parameters(), 'lr': lr},
        ]



class TransPoseR_layer4_deconv_3(TransPoseR):

    def __init__(self, block, layers, cfg, resnet, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        super(TransPoseR_layer4_deconv_3, self).__init__(block, layers, cfg, **kwargs)

        # self.layer1 = self._make_layer(block, 64, layers[2])
        # self.layer2 = self._make_layer(block, 128, layers[3], stride=2)
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.reduce = nn.Conv2d(2048, self.d_model, 1, bias=False)
        self.base_width = 64


    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        assert pe_type in ['none', 'learnable', 'sine']
        if pe_type == 'none':
            self.pos_embedding = None
            logger.info("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h // 8
                self.pe_w = w // 8
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(
                    torch.randn(length, 1, d_model))
                logger.info("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                logger.info("==> Add Sine PositionEmbedding~")


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 16, 64, 64, 64
        x = self.layer1(x)
        # 16, 256, 64, 64
        x = self.layer2(x)
        # 16, 512, 32, 32
        x = self.layer3(x)
        # 16, 1024, 16, 16
        x = self.layer4(x)
        # 16, 2048, 8, 8
        x = self.reduce(x)
        # 16, 256, 8, 8

        x = self.deconv_layers(x)

        bs, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        x = self.global_encoder(x, pos=self.pos_embedding)
        x = x.permute(1, 2, 0).contiguous().view(bs, c, h, w)
        # 16, 256, 8, 8
        x = self.deconv_layers_after(x)
        # 16, 256, 16, 16
        x = self.final_layer(x)

        return x


    def get_parameters(self, lr=1.):

        return [
            {'params': self.conv1.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.bn1.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.layer1.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.layer2.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.layer3.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.layer4.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.reduce.parameters(), 'lr': lr},
            {'params': self.pos_embedding, 'lr': lr},
            {'params': self.global_encoder.parameters(), 'lr': lr},
            {'params': self.deconv_layers.parameters(), 'lr': lr},
            {'params': self.deconv_layers_after.parameters(), 'lr': lr},
            {'params': self.final_layer.parameters(), 'lr': lr},
        ]




class TransPoseR_layer4_deconv_4(TransPoseR):

    def __init__(self, block, layers, cfg, resnet, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS
        super(TransPoseR_layer4_deconv_4, self).__init__(block, layers, cfg, **kwargs)

        # self.layer1 = self._make_layer(block, 64, layers[2])
        # self.layer2 = self._make_layer(block, 128, layers[3], stride=2)
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.reduce = nn.Conv2d(2048, self.d_model, 1, bias=False)
        self.base_width = 64


    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        assert pe_type in ['none', 'learnable', 'sine']
        if pe_type == 'none':
            self.pos_embedding = None
            logger.info("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h // 4
                self.pe_w = w // 4
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(
                    torch.randn(length, 1, d_model))
                logger.info("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                logger.info("==> Add Sine PositionEmbedding~")


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 16, 64, 64, 64
        x = self.layer1(x)
        # 16, 256, 64, 64
        x = self.layer2(x)
        # 16, 512, 32, 32
        x = self.layer3(x)
        # 16, 1024, 16, 16
        x = self.layer4(x)
        # 16, 2048, 8, 8
        x = self.reduce(x)
        # 16, 256, 8, 8

        x = self.deconv_layers(x)

        bs, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)

        x = self.global_encoder(x, pos=self.pos_embedding)
        x = x.permute(1, 2, 0).contiguous().view(bs, c, h, w)
        # 16, 256, 8, 8
        # x = self.deconv_layers_after(x)
        # 16, 256, 16, 16
        x = self.final_layer(x)

        return x


    def get_parameters(self, lr=1.):

        return [
            {'params': self.conv1.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.bn1.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.layer1.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.layer2.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.layer3.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.layer4.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.reduce.parameters(), 'lr': lr},
            {'params': self.pos_embedding, 'lr': lr},
            {'params': self.global_encoder.parameters(), 'lr': lr},
            {'params': self.deconv_layers.parameters(), 'lr': lr},
            {'params': self.deconv_layers_after.parameters(), 'lr': lr},
            {'params': self.final_layer.parameters(), 'lr': lr},
        ]







resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


models_dict = {"TransPoseR" : TransPoseR,
               "TransPoseR_layer4" : TransPoseR_layer4,
               "TransPoseR_layer4_cross" : TransPoseR_layer4_cross,
               "TransPoseR_layer4_deconv_4" : TransPoseR_layer4_deconv_4,
               "TransPoseR_layer4_cross_contrastive" : TransPoseR_layer4_cross_contrastive,
               }



def get_pose_net_trans(cfg, arch, backbone, num_keypoints, **kwargs):
    from ..resnet import resnet50, resnet101

    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
    block_class, layers = resnet_spec[num_layers]

    if backbone == 'resnet50':
        backbone_net = resnet50(pretrained=True)
    elif backbone == 'resnet101':
        backbone_net = resnet101(pretrained=True)

    # backbone_net = resnet50()

    model = models_dict[arch](block_class, layers, cfg, backbone_net, num_keypoints=num_keypoints, args=kwargs['args'])

    if cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model


def get_pose_net_trans50(cfg, arch, **kwargs):
    from ..resnet import resnet50, resnet101

    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
    block_class, layers = resnet_spec[num_layers]

    # if backbone == 'resnet50':
    #     backbone_net = resnet50(pretrained=True)
    # elif backbone == 'resnet101':
    #     backbone_net = resnet101(pretrained=True)

    backbone_net = resnet50()
    model = models_dict[arch](block_class, layers, cfg, backbone_net, **kwargs)

    if cfg.MODEL.INIT_WEIGHTS:
        model.init_weights_imagenet(cfg.MODEL.PRETRAINED)

    return model



def get_pose_net(cfg, **kwargs):

    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
    block_class, layers = resnet_spec[num_layers]
    model = TransPoseR(block_class, layers, cfg, **kwargs)

    if cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model, cfg



def get_pose_net_layer4(cfg, **kwargs):

    from ..resnet import resnet50

    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
    block_class, layers = resnet_spec[num_layers]
    model = TransPoseR_layer4(block_class, layers, cfg, resnet50(), **kwargs)

    if cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model


def get_pose_net_layer4_cross(cfg, **kwargs):

    from ..resnet import resnet50

    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
    block_class, layers = resnet_spec[num_layers]
    model = TransPoseR_layer4_cross(block_class, layers, cfg, resnet50(), **kwargs)

    if cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model


def get_pose_net_layer4_cross_multiple_encoders(cfg, **kwargs):

    from ..resnet import resnet50

    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
    block_class, layers = resnet_spec[num_layers]
    model = TransPoseR_layer4_cross_multiple_encoder(block_class, layers, cfg, resnet50(), **kwargs)

    if cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model



def get_pose_net_layer3_layer4(cfg, **kwargs):

    from ..resnet import resnet50

    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
    block_class, layers = resnet_spec[num_layers]
    model = TransPoseR_layer3_layer4(block_class, layers, cfg, resnet50(), **kwargs)

    if cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model


def get_pose_net_layer3_layer4_add(cfg, **kwargs):

    from ..resnet import resnet50

    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
    block_class, layers = resnet_spec[num_layers]
    model = TransPoseR_layer3_layer4_add(block_class, layers, cfg, resnet50(), **kwargs)

    if cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model



def get_pose_net_layer4_residual(cfg, **kwargs):

    from ..resnet import resnet50

    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
    block_class, layers = resnet_spec[num_layers]
    model = TransPoseR_layer4_residual(block_class, layers, cfg, resnet50(), **kwargs)

    if cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model


def get_pose_net_layer4_deconv_first(cfg, **kwargs):

    from ..resnet import resnet50

    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
    block_class, layers = resnet_spec[num_layers]
    model = TransPoseR_layer4_deconv_first(block_class, layers, cfg, resnet50(), **kwargs)

    if cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model