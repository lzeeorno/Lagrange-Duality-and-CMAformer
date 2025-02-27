# -*- coding: utf-8 -*-
import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
import datetime

import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
import joblib
from skimage.io import imread

import torch
import torch.nn as nn
# from torchsummary import summary

import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from dataset.dataset import Dataset_ssl_lits2017_png_unlabeled, Dataset_ssl_lits2017_png

from utilities.metrics import dice_coef_lits, batch_iou, mean_iou, iou_score, hd95_2d
import utilities.losses as losses
from utilities.utils import str2bool, count_params, monitor_gradients
import pandas as pd
from net import (Unet, CMAformer)



from scipy.ndimage import zoom
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    # data preprocessing
    parser.add_argument('--upper', default=200)
    parser.add_argument('--lower', default=-200)
    parser.add_argument('--img_size', default=512)

    # mode name on log record
    parser.add_argument('--model_name', default='CMAformer',
                        choices=['Unet', 'AttUnet', 'res_unet_plus', 'R2Unet', 'R2AttU_Net', 'sepnet', 'KiU_Net',
                                 'Unet3D', 'CMAformer'])
    # pre trained
    parser.add_argument('--pretrained', default=False, type=str2bool)
    # dataset name on log record
    parser.add_argument('--dataset', default="LiTS2017",
                        help='dataset name')
    parser.add_argument('--input-channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--image-ext', default='png',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='png',
                        help='mask file extension')
    parser.add_argument('--aug', default='medAug')
    parser.add_argument('--loss', default='LDC consis+infoNCE contrast')

    # training
    parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=500, type=int,
                        metavar='N', help='early stopping (default: 30)')
    parser.add_argument('--gamma', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=28, type=int,
                        metavar='N', help='mini-batch size (default: set below 10 in 1 NVIDIA4090 if using Large model)')
    parser.add_argument('--optimizer', default='AdamW',
                        choices=['Adam', 'SGD','AdamW'])
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate, Resunet=1e-4, R2Unet=1e-5, Unet=1e-2')
    parser.add_argument('--momentum', default=0.98, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=3e-5, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=True, type=str2bool,
                        help='nesterov')
    parser.add_argument('--deepsupervision', default=False)
    parser.add_argument('--semiSupervised', default=True)
    args = parser.parse_args()

    return args

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = torch.tensor(0.0).cuda()
        self.sum = torch.tensor(0.0).cuda()
        self.count = torch.tensor(0.0).cuda()

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))

def train(args, train_loader, train_loader_unlabeled, model, criterion, ContrastiveLoss, optimizer, lr_decay, epoch, index, lambda_value, alpha):
    losses = AverageMeter()
    dices_1s = AverageMeter()
    dices_2s = AverageMeter()
    contrastives = AverageMeter()
    tot_losses = AverageMeter()

    model.train()
    l2_reg = 0.5
    scaler = torch.cuda.amp.GradScaler()  # Mixed precision training
    accumulation_steps = 3

    for i, ((ct_labeled, mask_labeled), ct_unlabeled) in tqdm(enumerate(zip(train_loader, train_loader_unlabeled)),
                                                              total=len(train_loader)):
        ct_labeled = ct_labeled.cuda()
        mask_labeled = mask_labeled.cuda()
        ct_unlabeled = ct_unlabeled.cuda()

        # Check for NaNs in inputs
        if torch.isnan(ct_labeled).any() or torch.isnan(mask_labeled).any() or torch.isnan(ct_unlabeled).any():
            print("Input contains NaN")

        # with torch.cuda.amp.autocast():  # Mixed precision training
        #     if args.semiSupervised == True:
        # Forward pass for labeled data
        pred_labeled = model(ct_labeled)
        #criterion = ContrastiveLoss new for ablation exp
        loss_labeled = criterion(pred_labeled, mask_labeled)
        # print(loss_labeled)
        if args.semiSupervised == True:
            # Forward pass for unlabeled data
            with torch.no_grad():
                pred_unlabeled = model(ct_unlabeled)

            # Adjust batch size if necessary
            if pred_labeled.size(0) != pred_unlabeled.size(0):
                min_batch_size = min(pred_labeled.size(0), pred_unlabeled.size(0))
                pred_labeled = pred_labeled[:min_batch_size]
                pred_unlabeled = pred_unlabeled[:min_batch_size]

            # Calculate consistency loss
            loss_contrastive = ContrastiveLoss(pred_labeled, pred_unlabeled)
            # print(loss_contrastive)
            # exit()
            tot_loss = (loss_labeled + loss_contrastive)/2

        else:
            loss_contrastive = torch.tensor(0.0).cuda()
            tot_loss = loss_labeled

        dice_1 = dice_coef_lits(pred_labeled, mask_labeled)[0]
        dice_2 = dice_coef_lits(pred_labeled, mask_labeled)[1]

        # # Check for NaNs in losses
        # if torch.isnan(loss_labeled):
        #     print("Loss labeled is NaN")
        # if torch.isnan(loss_contrastive):
        #     print("Loss contrastive is NaN")
        # if torch.isnan(tot_loss):
        #     print("Total consistency loss is NaN")

        losses.update(loss_labeled.item(), ct_labeled.size(0))
        contrastives.update(loss_contrastive.item(), ct_unlabeled.size(0))
        tot_losses.update(tot_loss.item(), ct_labeled.size(0) + ct_unlabeled.size(0))
        dices_1s.update(torch.tensor(dice_1), ct_labeled.size(0))
        dices_2s.update(torch.tensor(dice_2), ct_labeled.size(0))

        # # Compute gradient and do optimizing step
        # optimizer.zero_grad(set_to_none=True)
        # scaler.scale(tot_loss).backward()
        # if (i + 1) % accumulation_steps == 0:
        #     scaler.step(optimizer)
        #     scaler.update()
        #     optimizer.zero_grad(set_to_none=True)
        # Compute gradient and do optimizing step
        optimizer.zero_grad()
        tot_loss.backward()

        # 梯度裁剪
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        # 更新拉格朗日乘子
        # lambda_value += alpha * Lagrange_constraint.item()

    #monitor gradients
    monitor_gradients(model)
    # Update learning rate
    lr_decay.step()
    # if optimizer.param_groups[0]['lr'] < 0.0002:
    #     optimizer.param_groups[0]['lr'] = 0.001
    # else:
    #     lr_decay.step()

    # print(losses.avg)
    # print(contrastives.avg)
    # print(tot_losses.avg)

    log = OrderedDict([
        ('lr', optimizer.param_groups[0]['lr']),
        ('loss', losses.avg),
        ('ContrastiveLoss', contrastives.avg),
        ('tot_loss', tot_losses.avg),
        ('dice_1', dices_1s.avg),
        ('dice_2', dices_2s.avg),
    ])

    return log


def validate(args, val_loader, model, val_criterion, ContrastiveLoss, lambda_value):
    losses = AverageMeter()
    ious = AverageMeter()
    dices_1s = AverageMeter()
    dices_2s = AverageMeter()
    # hd95_s = AverageMeter()
    contrastives = AverageMeter()
    tot_losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (ct_labeled, mask_labeled) in tqdm(enumerate(val_loader), total=len(val_loader)):
            ct_labeled = ct_labeled.cuda()
            mask_labeled = mask_labeled.cuda()

            # with torch.cuda.amp.autocast():
            pred_labeled = model(ct_labeled)
            loss_labeled = val_criterion(pred_labeled, mask_labeled)
            iou = iou_score(pred_labeled, mask_labeled)
            # hd95 = hd95_2d(pred_labeled, mask_labeled)

            if args.semiSupervised == True:
                # Here, we assume unlabeled data is also passed in the validation loader
                if len(ct_labeled) > 1:  # Ensure there's more than one batch to compare
                    ct_unlabeled = ct_labeled[1].unsqueeze(0)  # Use second batch as unlabeled data for simplicity
                    pred_unlabeled = model(ct_unlabeled)
                    loss_contrastive = ContrastiveLoss(pred_labeled, pred_unlabeled)
                else:
                    loss_contrastive = torch.tensor(0.0).cuda()

                tot_loss = (loss_labeled + loss_contrastive)/2
                contrastives.update(loss_contrastive.item(), ct_unlabeled.size(0) if len(ct_labeled) > 1 else 1)
            else:
                tot_loss = loss_labeled

            dice_1 = dice_coef_lits(pred_labeled, mask_labeled)[0]
            dice_2 = dice_coef_lits(pred_labeled, mask_labeled)[1]

            losses.update(loss_labeled.item(), ct_labeled.size(0))
            tot_losses.update(tot_loss.item(), ct_labeled.size(0))
            ious.update(iou, ct_labeled.size(0))
            # hd95_s.update(torch.tensor(hd95), ct_labeled.size(0))
            dices_1s.update(torch.tensor(dice_1), ct_labeled.size(0))
            dices_2s.update(torch.tensor(dice_2), ct_labeled.size(0))

            log = OrderedDict([
                ('loss', losses.avg),
                ('ContrastiveLoss', contrastives.avg),
                ('tot_loss', tot_losses.avg),
                ('iou', ious.avg),
                # ('HD95_avg', hd95_s.avg),
                ('dice_1', dices_1s.avg),
                ('dice_2', dices_2s.avg),
            ])

            return log


def get_gamma(epoch, total_epochs):
    return ((1 - (epoch / total_epochs)) ** 0.9)


def main():
    args = parse_args()

    if args.name is None:
        if args.deepsupervision:
            args.name = '%s_%s' % (args.dataset, args.model_name)
        else:
            args.name = '%s_%s' % (args.dataset, args.model_name)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.exists('../trained_models/SSL/{}_{}/{}'.format(args.dataset, args.model_name, timestamp)):
        os.makedirs('../trained_models/SSL/{}_{}/{}'.format(args.dataset, args.model_name, timestamp))
    print('Config -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    print('------------')


    with open('../trained_models/SSL/{}_{}/{}/args.txt'.format(args.dataset,args.model_name, timestamp), 'w') as f:
        for arg in vars(args):
            print('%s: %s' % (arg, getattr(args, arg)), file=f)

    joblib.dump(args, '../trained_models/SSL/{}_{}/{}/args.pkl'.format(args.dataset, args.model_name, timestamp))

    # define loss function (criterion)
    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        #Lagrange Duality Consistency (LDC) Loss + sup MSE
        #change to Contrastive loss for ablation exp
        criterion = losses.Contrastive_InfoNCE_Loss().cuda()
        #ContrastiveLoss base on InfoNCE loss
        ContrastiveLoss = losses.Contrastive_InfoNCE_Loss().cuda()
        val_criterion = losses.Contrastive_InfoNCE_Loss().cuda()

    cudnn.benchmark = True
    if args.model_name == 'Unet3D':
        img_paths = glob('../data/train_image3D/*')
        mask_paths = glob('../data/train_mask3D/*')
    else:
        # Data loading code
        img_paths = glob('../data/testImage_lits2017_png/*')
        mask_paths = glob('../data/testMask_lits2017_png/*')

    '''
    1. divide dataset into train and validation sets
    2. divide train into labeled and unlabeled sets
    '''
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
        train_test_split(img_paths,
                         mask_paths,
                         test_size=0.2,
                         random_state=seed_value)

    labeled_ratio = 0.5
    train_img_paths_labeled, train_img_paths_unlabeled, train_mask_paths_labeled, _ = \
        train_test_split(train_img_paths,
                         train_mask_paths,
                         test_size=1-labeled_ratio,
                         random_state=seed_value)

    print("train_labeled_num:%s" % str(len(train_img_paths_labeled)))
    print("train_unlabeled_num:%s" % str(len(train_img_paths_unlabeled)))
    print("val_num:%s" % str(len(val_img_paths)))

    # exit()
    # create model
    print("=> creating model %s" % args.model_name)
    choices = ['Unet', 'res_unet_plus', 'R2Unet', 'sepnet', 'KiU_Net', 'AttUnet']
    if args.model_name == 'Unet':
        model = Unet.U_Net(args)
    if args.model_name == 'CMAformer':
        model = CMAformer.CMAformer(args)
        pretrain_path = '../trained_models/SSL/LiTS2017_CMAformer/2024-10-06-22-47-11/epoch96-val_loss:1.4968-val_dice2:0.4514_model.pth'
        print('CMAformer selected!')
    # if args.model_name == 'ResT':
        # rest_path = '../pre_trained/upernet_restv2_base_512_160k_ade20k.pth'
        # model = resT.ResTV2(embed_dims=[96, 192, 384, 768], depths=[1, 3, 16, 3])
    #multi gpu
    model = torch.nn.DataParallel(model).cuda()

    '''
    pretrain
    '''
    if args.pretrained == True:
        print('Pretrained model loading...')
        # checkpoint = torch.load(rest_path, map_location='cpu')
        # model.load_state_dict(checkpoint['model'])
        model.load_state_dict(torch.load(pretrain_path))
        print('Pretrained model loaded!')
    else:
        print('No Pretrained')

    print('{} parameters:{}'.format(args.model_name, count_params(model)))
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                     betas=(0.9, 0.999),
                                     weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        print('SGD optimizer loaded!')
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                      betas=(0.9, 0.999),
                                      weight_decay=args.weight_decay)
        print('AdamW optimizer loaded!')

    # update learning rate（Adam can auto change lr by betas
    lr_decay_list = []
    decay_range = int(args.epochs / 10)

    # for i in range(decay_time):
    #     decay_rate = args.lr/decay_time
    #     lr = lr*(1 - decay_rate)
    start = 10
    # decay_range = decay_range
    for i in range(decay_range):
        lr_decay_list.append(start)
        start += 10

    # print(lr_decay_list)
    # scheduler_mult = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_list, gamma=0.98)
    # data augmentation(not useful for medical image)
    # transform_npy = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     transforms.RandomRotation(degrees=5),
    #     transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
    #     transforms.ToTensor()
    # ])
    def make_odd(num):
        num = math.ceil(num)
        if num % 2 == 0:
            num += 1
        return num

    level = 5
    transform_ct = A.Compose([
        A.ColorJitter(brightness=0.04 * level, contrast=0, saturation=0, hue=0, p=0.2 * level),
        A.ColorJitter(brightness=0, contrast=0.04 * level, saturation=0, hue=0, p=0.2 * level),
        A.Posterize(num_bits=math.floor(8 - 0.8 * level), p=0.2 * level),
        A.Sharpen(alpha=(0.04 * level, 0.1 * level), lightness=(1, 1), p=0.2 * level),
        A.GaussianBlur(blur_limit=(3, make_odd(3 + 0.8 * level)), p=min(0.2 * level, 1)),
        A.GaussNoise(var_limit=(2 * level, 10 * level), mean=0, per_channel=True, p=0.2 * level),
        A.Rotate(limit=4 * level, interpolation=1, border_mode=0, value=0, mask_value=None, p=0.2 * level),
        A.HorizontalFlip(p=0.2 * level),
        A.VerticalFlip(p=0.2 * level),
        A.Affine(scale=(1 - 0.04 * level, 1 + 0.04 * level), translate_percent=None, translate_px=None, rotate=None,
                 shear=None, interpolation=1, cval=0, cval_mask=0, mode=0, fit_output=False, p=0.2 * level),
        A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None,
                 shear={'x': (0, 2 * level), 'y': (0, 0)}
                 , interpolation=1, cval=0, cval_mask=0, mode=0, fit_output=False,
                 p=0.2 * level),  # x
        A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None,
                 shear={'x': (0, 0), 'y': (0, 2 * level)}
                 , interpolation=1, cval=0, cval_mask=0, mode=0, fit_output=False,
                 p=0.2 * level),
        A.Affine(scale=None, translate_percent={'x': (0, 0.02 * level), 'y': (0, 0)}, translate_px=None, rotate=None,
                 shear=None, interpolation=1, cval=0, cval_mask=0, mode=0, fit_output=False,
                 p=0.2 * level),
        A.Affine(scale=None, translate_percent={'x': (0, 0), 'y': (0, 0.02 * level)}, translate_px=None, rotate=None,
                 shear=None, interpolation=1, cval=0, cval_mask=0, mode=0, fit_output=False,
                 p=0.2 * level),
        A.OneOf([
            A.ElasticTransform(alpha=0.1 * level, sigma=0.25 * level, alpha_affine=0.25 * level, p=0.1),
            A.GridDistortion(distort_limit=0.05 * level, p=0.1),
            A.OpticalDistortion(distort_limit=0.05 * level, p=0.1)
        ], p=0.2),
        ToTensorV2()
    ], p=1)
    # transform_ct = transforms.ToTensor()


    train_dataset = Dataset_ssl_lits2017_png(args, train_img_paths_labeled, train_mask_paths_labeled, transform=transform_ct)
    train_dataset_unlabeled = Dataset_ssl_lits2017_png_unlabeled(args, train_img_paths_unlabeled, transform=transform_ct)
    val_dataset = Dataset_ssl_lits2017_png(args, val_img_paths, val_mask_paths, transform=None)

    # # 检查数据集是否为空
    # print(len(train_dataset))
    # print(len(train_dataset_unlabeled))
    # print(len(val_dataset))
    # exit()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=False,
        drop_last=False
    )
    train_loader_unlabeled = torch.utils.data.DataLoader(
        train_dataset_unlabeled,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=False,
        drop_last=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )
    # # # 检查数据集是否为空
    # print(len(train_loader))
    # print(len(train_loader_unlabeled))
    # print(len(val_loader))
    # exit()

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'ContrastiveLoss', 'tot_loss', 'dice_1', 'dice_2',
        'val_loss', 'val_ContrastiveLoss', 'val_tot_loss', 'val_iou', 'val_dice_1', 'val_dice_2'
    ])
    best_loss = 100
    best_train_loss = 100
    best_T_dice = 0
    val_trigger = False
    best_iou = 0
    trigger = 0

    # 初始化拉格朗日乘子和步长
    lambda_value = 0.0  # 初始拉格朗日乘子值
    alpha = 0.01  # 拉格朗日乘子更新步长

    first_time = time.time()
    #lr decay
    # scheduler_mult = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.98)
    # 使用 CosineAnnealingLR 调度器实现余弦退火学习率衰减策略
    scheduler_mult = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

    for i, epoch in enumerate(range(args.epochs)):
        print('Epoch [%d/%d]' % (epoch, args.epochs))
        train_log = train(args, train_loader, train_loader_unlabeled, model, criterion, ContrastiveLoss, optimizer, scheduler_mult, epoch, i, lambda_value, alpha)
        print('lr %.8f - train_loss %.4f - train_ContrastiveLoss %.4f - train_tot_loss %.4f - dice_1 %.4f - dice_2 %.4f'
              % (train_log['lr'], train_log['loss'], train_log['ContrastiveLoss'], train_log['tot_loss'], train_log['dice_1'], train_log['dice_2']))

        train_loss = train_log['loss']
        tumor_dice = train_log['dice_2']
        if train_loss < 0:
            print('Gradient descent not exist!')
            break
        if (train_loss < best_train_loss) and (tumor_dice > best_T_dice):
            val_trigger = True
            best_train_loss = train_loss
            best_T_dice = tumor_dice

        #evaluate on validation set
        if val_trigger == True:
            print("=> Start Validation...")
            val_trigger = False
            val_log = validate(args, val_loader, model, val_criterion, ContrastiveLoss, lambda_value)
            print('lr %.8f - val_loss %.4f - val_ContrastiveLoss %.4f - val_tot_loss %.4f - val_iou %.4f - val_dice_1 %.4f - val_dice_2 %.4f'
                  % (train_log['lr'], val_log['loss'], val_log['ContrastiveLoss'], val_log['tot_loss'], val_log['iou'], val_log['dice_1'], val_log['dice_2']))

            tmp = pd.Series([
                epoch,
                train_log['lr'],
                train_log['loss'].cpu().item(),
                train_log['ContrastiveLoss'].cpu().item(),
                train_log['tot_loss'].cpu().item(),
                train_log['dice_1'].cpu().item(),
                train_log['dice_2'].cpu().item(),
                # train_log['HD95'],
                val_log['loss'].cpu().item(),
                val_log['ContrastiveLoss'].cpu().item(),
                val_log['tot_loss'].cpu().item(),
                val_log['iou'].cpu().item(),
                val_log['dice_1'].cpu().item(),
                val_log['dice_2'].cpu().item(),
            ], index=['epoch', 'lr', 'loss', 'ContrastiveLoss', 'tot_loss', 'dice_1', 'dice_2',
                      'val_loss', 'val_ContrastiveLoss','val_tot_loss', 'val_iou', 'val_dice_1', 'val_dice_2'])

            log = log._append(tmp, ignore_index=True)
            log.to_csv('../trained_models/SSL/{}_{}/{}/{}_{}_{}_batchsize_{}.csv'.format(args.dataset, args.model_name, timestamp, args.model_name,
                                                                                 args.aug, args.loss, args.batch_size),index=False)
            print('save result to csv ->')
            torch.save(model.state_dict(),
                       '../trained_models/SSL/{}_{}/{}/epoch{}-val_loss:{:.4f}-val_dice2:{:.4f}_model.pth'.format(
                           args.dataset, args.model_name, timestamp, epoch, val_log['loss'], val_log['dice_2'])
                       )
            print("=> saved best model .pth")
            # trigger += 1
            # val_loss = val_log['loss']
            # if val_loss < best_loss:
            #     torch.save(model.state_dict(),
            #                '../trained_models/{}_{}/{}/epoch{}-val_loss:{:.4f}-val_dice2:{:.4f}_model.pth'.format(
            #                 args.dataset, args.model_name, timestamp, epoch, val_log['loss'], val_log['dice_2'])
            #                )
            #     best_loss = val_loss
            #     print("=> saved best model")
            #     trigger = 0

            # early stopping
            if not args.early_stop is None:
                if trigger >= args.early_stop:
                    print("=> early stopping")
                    break
        else:
            tmp = pd.Series([
                epoch,
                train_log['lr'],
                train_log['loss'].cpu().item(),
                train_log['ContrastiveLoss'].cpu().item(),
                train_log['tot_loss'].cpu().item(),
                train_log['dice_1'].cpu().item(),
                train_log['dice_2'].cpu().item(),
            ], index=['epoch', 'lr', 'loss', 'ContrastiveLoss', 'tot_loss', 'dice_1', 'dice_2'])

            log = log._append(tmp, ignore_index=True)
            log.to_csv('../trained_models/SSL/{}_{}/{}/{}_{}_{}_batchsize_{}.csv'.format(args.dataset, args.model_name, timestamp,
                                                                                 args.model_name, args.aug, args.loss,
                                                                                 args.batch_size), index=False)
            print('save result to csv ->')
        # print('lr %.8f - train_loss %.4f - iou %.4f - dice_1 %.4f - dice_2 %.4f'
        #       % (train_log['lr'], train_log['loss'], train_log['iou'], train_log['dice_1'], train_log['dice_2']))
        # print('lr %.8f - val_loss %.4f - val_iou %.4f - val_dice_1 %.4f - val_dice_2 %.4f'
        #     %(train_log['lr'], val_log['loss'], val_log['iou'], val_log['dice_1'], val_log['dice_2']))

        # print('loss %.4f - iou %.4f - dice %.4f ' %(train_log['loss'], train_log['iou'], train_log['dice']))
        end_time = time.time()
        print("time:", (end_time - first_time) / 60)


        torch.cuda.empty_cache()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    seed_value = 3407
    np.random.seed(seed_value)
    random.seed(seed_value)
    # os.environ['PYTHONHASHSEED'] = str(seed_value)  # ban hash random, let experiment reproduceable
    # set cpu seed
    torch.manual_seed(seed_value)
    # set gpu seed(1 gpu)
    torch.cuda.manual_seed(seed_value)
    # multi gpu
    # torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    main()
