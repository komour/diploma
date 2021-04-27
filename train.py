import argparse
import os
import shutil
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from MODELS.model_resnet import *
from PIL import ImageFile
import numpy as np
import torch.nn.functional as F

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score

import matplotlib.pyplot as plt

from custom_dataset import DatasetISIC2018

import wandb

from collections import OrderedDict

from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp
from MODELS.model_resnet import *
from torchvision.utils import make_grid, save_image
from math import inf, isnan
import sys

from typing import List

ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='PyTorch ResNet+CBAM ISIC2018 Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet')
parser.add_argument('--depth', default=50, type=int, metavar='D', help='model depth')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', type=int, default=1234, metavar='BS')
parser.add_argument('--prefix', type=str, default='', metavar='PFX',
                    help='prefix for logging & checkpoint saving')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluation only')
parser.add_argument('--cuda-device', type=int, default=0)
parser.add_argument('--run-name', type=str, default='noname run', help='run name on the W&B service')
parser.add_argument('--is-server', type=int, choices=[0, 1], default=1)
parser.add_argument("--tags", nargs='+', default=['default-tag'])
parser.add_argument('--number', type=int, default=0, help='number of run in the run pool')
parser.add_argument('--lmbd', type=int, default=1, help='coefficient for additional loss')
parser.add_argument('--image-size', type=int, default=256, help='coefficient for additional loss')

args = parser.parse_args()
is_server = args.is_server == 1
SAM_AMOUNT = 3
CLASS_AMOUNT = 5
TRAIN_AMOUNT = 1600
VAL_AMOUNT = 400
TEST_AMOUNT = 594

if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')
label_file = 'images-onehot.txt'

epochs_predicted = [[] for _ in range(CLASS_AMOUNT)]
epochs_expected = [[] for _ in range(CLASS_AMOUNT)]

avg_f1_best = 0
avg_f1_val_best = 0
avg_mAP_best = 0
avg_mAP_val_best = 0
avg_prec_best = 0
avg_prec_val_best = 0
avg_recall_best = 0
avg_recall_val_best = 0

# +1 for average value
f1_trn_best = [0. for _ in range(CLASS_AMOUNT + 1)]
f1_val_best = [0. for _ in range(CLASS_AMOUNT + 1)]

mAP_trn_best = [0. for _ in range(CLASS_AMOUNT + 1)]
mAP_val_best = [0. for _ in range(CLASS_AMOUNT + 1)]

prec_trn_best = [0. for _ in range(CLASS_AMOUNT + 1)]
prec_val_best = [0. for _ in range(CLASS_AMOUNT + 1)]

recall_trn_best = [0. for _ in range(CLASS_AMOUNT + 1)]
recall_val_best = [0. for _ in range(CLASS_AMOUNT + 1)]

# SAM miss attention metrics
sam_att_miss = [[] for _ in range(SAM_AMOUNT)]
sam_att_direct = [[] for _ in range(SAM_AMOUNT)]

sam_att_miss_best = [inf for _ in range(SAM_AMOUNT)]
sam_att_direct_best = [-inf for _ in range(SAM_AMOUNT)]

sam_att_miss_val = [[] for _ in range(SAM_AMOUNT)]
sam_att_direct_val = [[] for _ in range(SAM_AMOUNT)]

sam_att_miss_val_best = [inf for _ in range(SAM_AMOUNT)]
sam_att_direct_val_best = [-inf for _ in range(SAM_AMOUNT)]

iou = [[] for _ in range(SAM_AMOUNT)]
iou_best = [-inf for _ in range(SAM_AMOUNT)]

iou_val = [[] for _ in range(SAM_AMOUNT)]
iou_val_best = [-inf for _ in range(SAM_AMOUNT)]

# GradCam miss attention metric
gradcam_miss_att = []
gradcam_miss_att_best = inf

gradcam_direct_att = []
gradcam_direct_att_best = -inf

gradcam_miss_att_val = []
gradcam_miss_att_val_best = inf

gradcam_direct_att_val = []
gradcam_direct_att_val_best = -inf

run = None

train_vis_file = open('vis_train.txt')
train_vis_image_names = train_vis_file.readlines()

val_vis_file = open('vis_val.txt')
val_vis_image_names = val_vis_file.readlines()


def main():
    if is_server:
        wandb.login()
    global args, run
    args = parser.parse_args()
    print("args", args)
    if args.resume is None:
        print("Run w/o checkpoint!")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    # create model
    if args.arch == "resnet34":
        model = models.resnet34(pretrained=True)
        model.fc = nn.Linear(512, CLASS_AMOUNT)
    elif args.arch == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, CLASS_AMOUNT)
    elif args.arch == "resnet101":
        model = models.resnet101(pretrained=True)
        model.fc = nn.Linear(2048, CLASS_AMOUNT)
    elif args.arch == "vgg13":
        model = models.vgg13(pretrained=True)
        model.classifier[6] = nn.Linear(4096, CLASS_AMOUNT)
    elif args.arch == "vgg16":
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, CLASS_AMOUNT)
    elif args.arch == "BAM":
        model = ResidualNet('ImageNet', args.depth, CLASS_AMOUNT, 'BAM', args.image_size)
    else:
        model = ResidualNet('ImageNet', args.depth, CLASS_AMOUNT, 'CBAM', args.image_size)

    # model = torch.nn.DataParallel(model, device_ids=list(range(4)), output_device=args.cuda_device)

    # define loss function (criterion) and optimizer
    pos_weight_train = torch.Tensor(
        [[3.27807486631016, 2.7735849056603774, 12.91304347826087, 0.6859852476290832, 25.229508196721312]])
    if is_server:
        # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_train).cuda()
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_train).cuda(args.cuda_device)
        # sam_criterion_outer = nn.BCELoss(reduction='none').cuda()
        sam_criterion_outer = nn.BCELoss(reduction='none').cuda(args.cuda_device)
        # sam_criterion = nn.BCELoss().cuda()
        sam_criterion = nn.BCELoss().cuda(args.cuda_device)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_train)
        sam_criterion_outer = nn.BCELoss(reduction='none')
        sam_criterion = nn.BCELoss()
    # optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    start_epoch = 0

    # code to load my own checkpoints:
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print(f"=> loading checkpoint '{args.resume}'")
    #         checkpoint = torch.load(args.resume)
    #         state_dict = checkpoint['state_dict']
    #
    #         model.load_state_dict(state_dict)
    #         print(f"=> loaded checkpoint '{args.resume}'")
    #         print(f"epoch = {checkpoint['epoch']}")
    #         start_epoch = checkpoint['epoch']
    #     else:
    #         print(f"=> no checkpoint found at '{args.resume}'")
    #         return -1

    config = dict(
        architecture=f"{args.arch}{args.depth}" if args.arch == "resnet" else args.arch,
        learning_rate=args.lr,
        epochs=args.epochs,
        start_epoch=start_epoch,
        seed=args.seed,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        momentum=args.momentum,
        workers=args.workers,
        prefix=args.prefix,
        resume=args.resume,
        evaluate=args.evaluate,
        lmbd=args.lmbd
    )
    if is_server:
        run = wandb.init(config=config, project="vol.7", name=args.run_name, tags=args.tags)

    if is_server:
        # model = model.cuda()
        model = model.cuda(args.cuda_device)

    # code to load imagenet checkpoint:
    # create dummy layer to init weights in the state_dict
    dummy_fc = torch.nn.Linear(512 * 4, CLASS_AMOUNT)
    torch.nn.init.xavier_uniform_(dummy_fc.weight)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=f'cuda:{args.cuda_device}')
            state_dict = checkpoint['state_dict']

            state_dict['module.fc.weight'] = dummy_fc.weight
            state_dict['module.fc.bias'] = dummy_fc.bias

            # remove `module.` prefix because we don't use torch.nn.DataParallel

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v

            #  load weights to the new added cbam module from the nearest cbam module in checkpoint
            # new_state_dict["cbam_after_layer4.ChannelGate.mlp.1.weight"] = new_state_dict['layer4.2.cbam.ChannelGate.mlp.1.weight']
            # new_state_dict["cbam_after_layer4.ChannelGate.mlp.1.bias"] = new_state_dict['layer4.2.cbam.ChannelGate.mlp.1.bias']
            # new_state_dict["cbam_after_layer4.ChannelGate.mlp.3.weight"] = new_state_dict['layer4.2.cbam.ChannelGate.mlp.3.weight']
            # new_state_dict["cbam_after_layer4.ChannelGate.mlp.3.bias"] = new_state_dict['layer4.2.cbam.ChannelGate.mlp.3.bias']
            # new_state_dict["cbam_after_layer4.SpatialGate.spatial.conv.weight"] = new_state_dict['layer4.2.cbam.SpatialGate.spatial.conv.weight']
            # new_state_dict["cbam_after_layer4.SpatialGate.spatial.bn.weight"] = new_state_dict['layer4.2.cbam.SpatialGate.spatial.bn.weight']
            # new_state_dict["cbam_after_layer4.SpatialGate.spatial.bn.bias"] = new_state_dict['layer4.2.cbam.SpatialGate.spatial.bn.bias']
            # new_state_dict["cbam_after_layer4.SpatialGate.spatial.bn.running_mean"] = new_state_dict['layer4.2.cbam.SpatialGate.spatial.bn.running_mean']
            # new_state_dict["cbam_after_layer4.SpatialGate.spatial.bn.running_var"] = new_state_dict['layer4.2.cbam.SpatialGate.spatial.bn.running_var']

            model.load_state_dict(new_state_dict)

            # model.load_state_dict(state_dict)
            print(f"=> loaded checkpoint '{args.resume}'")
            # print(f"epoch = {checkpoint['epoch']}")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")
            return -1

    if is_server:
        wandb.watch(model, criterion, log="all", log_freq=args.print_freq)
    # print("model")
    # print(model)
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    cudnn.benchmark = True
    # Data loading code
    if args.image_size == 256:
        root_dir = 'data/'
        segm_dir = "images/256ISIC2018_Task1_Training_GroundTruth/"
        size0 = 224
    else:
        root_dir = 'data512/'
        segm_dir = "images/512ISIC2018_Task1_Training_GroundTruth/"
        size0 = 448

    traindir = os.path.join(root_dir, 'train')
    train_labels = os.path.join(root_dir, 'train', 'images_onehot_train.txt')
    valdir = os.path.join(root_dir, 'val')
    val_labels = os.path.join(root_dir, 'val', 'images_onehot_val.txt')
    # testdir = os.path.join(args.data, 'test')
    # test_labels = os.path.join(args.data, 'test', 'images_onehot_test.txt')

    # import pdb
    # pdb.set_trace()
    val_dataset = DatasetISIC2018(
        val_labels,
        valdir,
        segm_dir,
        size0,
        False,
        False,
        transforms.CenterCrop(size0)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    train_dataset = DatasetISIC2018(
        train_labels,
        traindir,
        segm_dir,
        size0,
        True,  # perform flips
        True  # perform random resized crop with size = 224
    )

    # test_dataset = DatasetISIC2018(
    #     test_labels,
    #     testdir,
    #     False,
    #     False,
    #     transforms.CenterCrop(size0)
    # )
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True
    # )
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler
    )

    # need to do visualization with loading from checkpoint (e.g. every 10 epochs)
    epoch_number = 0

    # create_needed_folders_for_hists()
    for epoch in range(start_epoch, start_epoch + args.epochs):
        # adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        clear_expected_predicted()
        train(train_loader, model, criterion, sam_criterion, sam_criterion_outer, optimizer, epoch, epoch_number)

        # evaluate on validation set
        clear_expected_predicted()
        validate(val_loader, model, criterion, epoch, optimizer, epoch_number, sam_criterion, sam_criterion_outer)
        epoch_number += 1
        break

    save_summary()
    if run is not None:
        run.finish()


def measure_gradcam_metrics(no_norm_gc_mask_numpy: torch.Tensor, segm: torch.Tensor, gradcam_direct: List[float],
                            gradcam_miss: List[float]):
    """
    Measure gradcam metric

    @param segm - true mask, shape B x 3 x H x W
    @param no_norm_gc_mask_numpy - Grad-CAM output w/o normalization, shape B x 1 x H x W
    @param gradcam_direct - list of gradcam_direct metric value for current epoch
    @param gradcam_miss - list of gradcam_miss metric value for current epoch
    """

    # initial segm size = [1, 3, 224, 224]
    maxpool = nn.MaxPool3d(kernel_size=(3, 1, 1))
    true_mask = maxpool(segm)
    true_mask_invert = 1 - true_mask

    true_mask_invert = true_mask_invert.detach().clone().cpu()
    true_mask = true_mask.detach().clone().cpu()
    gradcam_mask = no_norm_gc_mask_numpy.detach().clone().cpu()

    # iterate over batch to calculate metrics on each image of the batch
    assert gradcam_mask.size() == true_mask.size() == true_mask_invert.size()
    for i in range(gradcam_mask.size(0)):
        cur_gc = gradcam_mask[i]
        cur_mask = true_mask[i]
        cur_mask_inv = true_mask_invert[i]

        gradcam_miss.append(safe_division(torch.sum(cur_gc * cur_mask_inv), torch.sum(cur_gc)))
        gradcam_direct.append(safe_division(torch.sum(cur_gc * cur_mask), torch.sum(cur_gc)))


def measure_sam_metrics(sam_output: List[torch.Tensor], segm: torch.Tensor, sam_direct: List[List[float]],
                        sam_miss: List[List[float]], jaccard: List[List[float]]):
    """
    Measure SAM attention metrics and IoU

    @param sam_output[i] - SAM-output #i
    @param segm - true mask, shape B x 3 x H x W
    @param sam_miss - list of sam_att_miss metric values for current epoch for each SAM (e.g. sam_miss[0] - for SAM-0)
    @param sam_direct - list of sam_att_direct metric values for current epoch for each SAM (e.g. sam_direct[0] - for SAM-0)
    @param jaccard - list of iou metric values for current epoch for each SAM (e.g. jaccard[0] - for SAM-0)
    """

    # initial segm size = [1, 3, 224, 224]
    maxpool_segm1 = nn.MaxPool3d(kernel_size=(3, 4, 4))
    maxpool_segm2 = nn.MaxPool3d(kernel_size=(3, 8, 8))
    maxpool_segm3 = nn.MaxPool3d(kernel_size=(3, 16, 16))

    true_mask1 = maxpool_segm1(segm)
    true_mask2 = maxpool_segm2(segm)
    true_mask3 = maxpool_segm3(segm)

    true_mask_inv1 = 1 - true_mask1
    true_mask_inv2 = 1 - true_mask2
    true_mask_inv3 = 1 - true_mask3

    true_masks = [true_mask1, true_mask2, true_mask3]
    invert_masks = [true_mask_inv1, true_mask_inv2, true_mask_inv3]

    # measure SAM attention metrics
    for i in range(SAM_AMOUNT):
        cur_sam_batch = sam_output[i].detach().clone().cpu()
        cur_mask_batch = true_masks[i].detach().clone().cpu()
        cur_mask_inv_batch = invert_masks[i].detach().clone().cpu()

        # iterate over batch to calculate metrics on each image of the batch
        assert cur_sam_batch.size() == cur_mask_batch.size()
        for j in range(cur_sam_batch.size(0)):
            cur_sam = cur_sam_batch[j]
            cur_mask = cur_mask_batch[j]
            cur_mask_inv = cur_mask_inv_batch[j]

            sam_miss[i].append(safe_division(torch.sum(cur_sam * cur_mask_inv),
                                             torch.sum(cur_sam)))
            sam_direct[i].append(safe_division(torch.sum(cur_sam * cur_mask),
                                               torch.sum(cur_sam)))
            jaccard[i].append(calculate_iou((cur_sam > 0.5).int(), cur_mask.int()))


def calculate_additional_loss(segm: torch.Tensor, sam_output: torch.Tensor, sam_criterion, sam_criterion_outer):
    """
    @param segm: true mask, shape B x 3 x H x W
    @param sam_output[i] - SAM-output #i
    @param sam_criterion - nn.BCELoss() (witch or w/o CUDA)
    @param sam_criterion_outer - nn.BCELoss(reduction='none') (witch or w/o CUDA)
    """
    maxpool_segm1 = nn.MaxPool3d(kernel_size=(3, 4, 4))
    maxpool_segm2 = nn.MaxPool3d(kernel_size=(3, 8, 8))
    maxpool_segm3 = nn.MaxPool3d(kernel_size=(3, 16, 16))

    true_mask1 = maxpool_segm1(segm)
    true_mask2 = maxpool_segm2(segm)
    true_mask3 = maxpool_segm3(segm)

    true_mask_inv1 = 1 - true_mask1
    true_mask_inv2 = 1 - true_mask2
    true_mask_inv3 = 1 - true_mask3

    true_masks = [true_mask1, true_mask2, true_mask3]
    invert_masks = [true_mask_inv1, true_mask_inv2, true_mask_inv3]

    loss_outer_sum = [0 for _ in range(SAM_AMOUNT)]
    loss_inv_sum = [0 for _ in range(SAM_AMOUNT)]

    assert len(true_masks) == SAM_AMOUNT
    # iterate over SAM number
    for i in range(SAM_AMOUNT):
        # iterate over batch
        for j in range(len(true_masks[i])):
            cur_mask = true_masks[i][j]
            cur_mask_inv = invert_masks[i][j]
            cur_sam_output = sam_output[i][j]

            loss_outer_sum[i] += safe_division(
                torch.sum(sam_criterion_outer(cur_sam_output, cur_mask) * cur_mask_inv),
                torch.sum(cur_mask_inv))

            loss_inv_sum[i] += safe_division(
                torch.sum(sam_criterion_outer(cur_sam_output, cur_mask_inv) * cur_mask),
                torch.sum(cur_mask))

    loss = [None for _ in range(SAM_AMOUNT)]
    loss_inv = [None for _ in range(SAM_AMOUNT)]

    loss_outer = [None for _ in range(SAM_AMOUNT)]
    loss_outer_inv = [None for _ in range(SAM_AMOUNT)]

    for i in range(SAM_AMOUNT):
        loss[i] = args.lmbd * sam_criterion(sam_output[i], true_masks[i])
        loss_inv[i] = args.lmbd * sam_criterion(sam_output[i], invert_masks[i])

        loss_outer[i] = args.lmbd * loss_outer_sum[i] / args.batch_size
        loss_outer_inv[i] = args.lmbd * loss_inv_sum[i] / args.batch_size
    return loss, loss_inv, loss_outer, loss_outer_inv


def choose_add_loss(loss: list, loss_inv: list, loss_outer: list, loss_outer_inv: list):
    """
    The choice of additional loss depends on args.number.
    Arguments - lists of already calculated losses, len of the list equals SAM_AMOUNT

    @param loss - default SAM-loss
    @param loss_inv: SAM-loss calculated with invert mask
    @param loss_outer: outer SAM-loss
    @param loss_outer_inv: outer SAM-loss calculated with invert mask
    """
    if args.number == 0:
        return 0
    assert len(loss) == len(loss_inv) == len(loss_outer) == len(loss_outer_inv) == SAM_AMOUNT

    loss_add = None
    if args.number == 1:
        loss_add = loss[0]
    elif args.number == -1:
        loss_add = loss_inv[0]
    elif args.number == 2:
        loss_add = loss[1]
    elif args.number == -2:
        loss_add = loss_inv[1]
    elif args.number == 3:
        loss_add = loss[2]
    elif args.number == -3:
        loss_add = loss_inv[2]
    elif args.number == 5:
        loss_add = sum(loss)
    elif args.number == -5:
        loss_add = sum(loss_inv)
    elif args.number == 10:
        loss_add = loss_outer[0]
    elif args.number == -10:
        loss_add = loss_outer_inv[0]
    elif args.number == 20:
        loss_add = loss_outer[1]
    elif args.number == -20:
        loss_add = loss_outer_inv[1]
    elif args.number == 30:
        loss_add = loss_outer[2]
    elif args.number == -30:
        loss_add = loss_outer_inv[2]
    elif args.number == 50:
        loss_add = sum(loss_outer)
    elif args.number == -50:
        loss_add = sum(loss_outer_inv)
    return loss_add


def calculate_and_choose_additional_loss(segm: torch.Tensor, sam_output: torch.Tensor, sam_criterion,
                                         sam_criterion_outer):
    """
    Calculate all add loss and select required one for the current run. The choice depends on the args.number.

    @param segm: true mask, shape B x 3 x H x W
    @param sam_output[i] - SAM-output #i
    @param sam_criterion - nn.BCELoss() (witch or w/o CUDA)
    @param sam_criterion_outer - nn.BCELoss(reduction='none') (witch or w/o CUDA)
    """
    return choose_add_loss(*calculate_additional_loss(segm, sam_output, sam_criterion, sam_criterion_outer))


def train(train_loader, model, criterion, sam_criterion, sam_criterion_outer, optimizer, epoch, epoch_number):
    loss_sum_stat = AverageMeter()
    loss_main_stat = AverageMeter()
    loss_add_stat = AverageMeter()
    # switch to train mode
    model.train()

    global iou, sam_att_miss, sam_att_direct, gradcam_miss_att, gradcam_direct_att
    for i, dictionary in enumerate(train_loader):
        input_img = dictionary['image']
        target = dictionary['label']
        segm = dictionary['segm']
        if is_server:
            input_img = input_img.cuda(args.cuda_device)
            target = target.cuda(args.cuda_device)
            segm = segm.cuda(args.cuda_device)

        # get gradcam mask + compute output
        target_layer = model.layer4
        gradcam = GradCAM(model, target_layer=target_layer)
        gc_mask, no_norm_gc_mask, output, sam_output = gradcam(input_img, retain_graph=True)

        # calculate loss
        loss_main = criterion(output, target)
        loss_add = calculate_and_choose_additional_loss(segm, sam_output, sam_criterion, sam_criterion_outer)
        loss_sum = loss_main + loss_add

        # measure metrics
        measure_accuracy(output.data, target)
        measure_gradcam_metrics(no_norm_gc_mask, segm, gradcam_direct_att, gradcam_miss_att)
        measure_sam_metrics(sam_output, segm, sam_att_direct, sam_att_miss, iou)
        loss_add_stat.update(loss_add.item() if loss_add != 0 else loss_add, input_img.size(0))
        loss_main_stat.update(loss_main.item(), input_img.size(0))
        loss_sum_stat.update(loss_sum.item(), input_img.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()
        if i == 10:
            break

        if i % args.print_freq == 0:
            print(f'Train: [{epoch}][{i}/{len(train_loader)}]')
    wandb_log_train(epoch, loss_sum_stat.avg, loss_add_stat.avg, loss_main_stat.avg)


def validate(val_loader, model, criterion, epoch, optimizer, epoch_number, sam_criterion, sam_criterion_outer):
    loss_sum_stat = AverageMeter()
    loss_add_stat = AverageMeter()
    loss_main_stat = AverageMeter()

    # switch to evaluate mode
    model.eval()
    global val_vis_image_names

    global iou_val, sam_att_miss_val, sam_att_direct_val, gradcam_miss_att_val, gradcam_direct_att_val
    for i, dictionary in enumerate(val_loader):
        input_img = dictionary['image']
        target = dictionary['label']
        segm = dictionary['segm']
        if is_server:
            input_img = input_img.cuda(args.cuda_device)
            target = target.cuda(args.cuda_device)
            segm = segm.cuda(args.cuda_device)

        # get gradcam mask + compute output
        target_layer = model.layer4
        gradcam = GradCAM(model, target_layer=target_layer)
        gc_mask, no_norm_gc_mask, output, sam_output = gradcam(input_img)

        # calculate loss
        loss_main = criterion(output, target)
        loss_add = calculate_and_choose_additional_loss(segm, sam_output, sam_criterion, sam_criterion_outer)
        loss_sum = loss_main + loss_add

        # measure metrics
        measure_accuracy(output.data, target)
        measure_gradcam_metrics(no_norm_gc_mask, segm, gradcam_direct_att_val, gradcam_miss_att_val)
        measure_sam_metrics(sam_output, segm, sam_att_direct_val, sam_att_miss_val, iou_val)
        loss_add_stat.update(loss_add.item() if loss_add != 0 else loss_add, input_img.size(0))
        loss_main_stat.update(loss_main.item(), input_img.size(0))
        loss_sum_stat.update(loss_sum.item(), input_img.size(0))

        if i == 10:
            break

        if i % args.print_freq == 0:
            print(f'Validate: [{epoch}][{i}/{len(val_loader)}]')

    wandb_log_val(epoch, loss_sum_stat.avg, loss_add_stat.avg, loss_main_stat.avg)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def write_expected_predicted(target, output):
    # iterate over batch
    assert target.size() == output.size()
    for i in range(target.size(0)):
        cur_target = target[i]
        cur_output = output[i]

        # iterate over classes
        assert cur_target.size(0) == CLASS_AMOUNT
        for j in range(CLASS_AMOUNT):
            epochs_expected[j].append(cur_target[j])
            epochs_predicted[j].append(cur_output[j])


def clear_expected_predicted():
    for exp in epochs_expected:
        exp.clear()
    for pred in epochs_predicted:
        pred.clear()


def count_mAP():
    mAP_per_class = [-1. for _ in range(CLASS_AMOUNT)]
    for i in range(CLASS_AMOUNT):
        mAP_per_class[i] = average_precision_score(epochs_expected[i], epochs_predicted[i])
    avg_mAP = sum(mAP_per_class) / CLASS_AMOUNT
    return mAP_per_class + [avg_mAP]


def count_precision():
    prec_per_class = [-1. for _ in range(CLASS_AMOUNT)]
    for i in range(CLASS_AMOUNT):
        prec_per_class[i] = precision_score(epochs_expected[i], epochs_predicted[i], average="binary")
    avg_prec = sum(prec_per_class) / CLASS_AMOUNT
    return prec_per_class + [avg_prec]


def count_recall():
    recall_per_class = [-1. for _ in range(CLASS_AMOUNT)]
    for i in range(CLASS_AMOUNT):
        recall_per_class[i] = recall_score(epochs_expected[i], epochs_predicted[i], average="binary")
    avg_recall = sum(recall_per_class) / CLASS_AMOUNT
    return recall_per_class + [avg_recall]


def count_f1():
    f1_per_class = [-1. for _ in range(CLASS_AMOUNT)]
    for i in range(CLASS_AMOUNT):
        f1_per_class[i] = f1_score(epochs_expected[i], epochs_predicted[i], average="binary")
    avg_f1 = sum(f1_per_class) / CLASS_AMOUNT
    return f1_per_class + [avg_f1]


def wandb_log_train(epoch, loss_sum, loss_add, loss_main):
    if not is_server:
        return

    global f1_trn_best, mAP_trn_best, prec_trn_best, recall_trn_best
    global sam_att_miss, sam_att_miss_best, iou, iou_best, gradcam_miss_att, gradcam_miss_att_best
    global sam_att_direct, sam_att_direct_best
    global gradcam_direct_att, gradcam_direct_att_best

    mAP_list = count_mAP()
    prec_list = count_precision()
    recall_list = count_recall()
    f1_list = count_f1()

    # lists contains metric value of each class + average value as the last element
    assert len(f1_list) == len(recall_list) == len(prec_list) == len(mAP_list) == CLASS_AMOUNT + 1

    for i in range(CLASS_AMOUNT + 1):
        f1_trn_best[i] = max(f1_list[i], f1_trn_best[i])
        mAP_trn_best[i] = max(mAP_list[i], mAP_trn_best[i])
        prec_trn_best[i] = max(prec_list[i], prec_trn_best[i])
        recall_trn_best[i] = max(recall_list[i], recall_trn_best[i])

    # attention metrics
    iou_avg = [-1. for _ in range(SAM_AMOUNT)]
    sam_att_miss_avg = [-1. for _ in range(SAM_AMOUNT)]
    sam_att_direct_avg = [-1. for _ in range(SAM_AMOUNT)]

    for j in range(SAM_AMOUNT):
        assert len(iou[j]) == TRAIN_AMOUNT
        assert len(sam_att_miss[j]) == TRAIN_AMOUNT
        assert len(sam_att_direct[j]) == TRAIN_AMOUNT

        iou_avg[j] = sum(iou[j]) / len(iou[j])
        sam_att_miss_avg[j] = sum(sam_att_miss[j]) / len(sam_att_miss[j])
        sam_att_direct_avg[j] = sum(sam_att_direct[j]) / len(sam_att_direct[j])

    for j in range(SAM_AMOUNT):
        iou_best[j] = max(iou_best[j], iou_avg[j])
        sam_att_miss_best[j] = min(sam_att_miss_best[j], sam_att_miss_avg[j])
        sam_att_direct_best[j] = max(sam_att_direct_best[j], sam_att_direct_avg[j])

    assert len(gradcam_direct_att) == len(gradcam_miss_att) == TRAIN_AMOUNT
    avg_gradcam_miss_att = sum(gradcam_miss_att) / len(gradcam_miss_att)
    avg_gradcam_direct_att = sum(gradcam_direct_att) / len(gradcam_direct_att)

    gradcam_miss_att_best = min(gradcam_miss_att_best, avg_gradcam_miss_att)
    gradcam_direct_att_best = max(gradcam_direct_att_best, avg_gradcam_direct_att)

    for j in range(SAM_AMOUNT):
        iou[j].clear()
        sam_att_miss[j].clear()
        sam_att_direct[j].clear()

    gradcam_miss_att.clear()
    gradcam_direct_att.clear()
    dict_for_log = make_dict_for_log(suffix="trn", loss_sum=loss_sum, loss_add=loss_add, loss_main=loss_main,
                                     mAP_list=mAP_list, prec_list=prec_list, recall_list=recall_list, f1_list=f1_list,
                                     jaccard=iou_avg, sam_miss=sam_att_direct_avg, sam_direct=sam_att_direct_avg,
                                     gc_miss=avg_gradcam_miss_att, gc_direct=avg_gradcam_direct_att)
    wandb.log(dict_for_log, step=epoch)


def make_dict_for_log(suffix: str, loss_sum: float, loss_add: float, loss_main: float, mAP_list: List[float],
                      prec_list: List[float], recall_list: List[float], f1_list: List[float], jaccard: List[float],
                      sam_miss: List[float], sam_direct: List[float], gc_miss: float, gc_direct: float):
    """Takes all metrics and makes dictionary for wand_log"""
    log_dict = {f'loss/sum_{suffix}': loss_sum, f'loss/add_{suffix}': loss_add, f'loss/main_{suffix}': loss_main,
                f'gradcam_miss_{suffix}': gc_miss, f'gradcam_direct_{suffix}': gc_direct}

    assert len(f1_list) == len(recall_list) == len(prec_list) == len(mAP_list) == CLASS_AMOUNT + 1
    for i in range(CLASS_AMOUNT):
        log_dict[f'f1/c{i + 1}_{suffix}'] = f1_list[i]
        log_dict[f'mAP/c{i + 1}_{suffix}'] = mAP_list[i]
        log_dict[f'prec/c{i + 1}_{suffix}'] = prec_list[i]
        log_dict[f'recall/c{i + 1}_{suffix}'] = recall_list[i]
    log_dict[f'f1/avg_{suffix}'] = f1_list[-1]
    log_dict[f'mAP/avg_{suffix}'] = mAP_list[-1]
    log_dict[f'recall/avg_{suffix}'] = recall_list[-1]
    log_dict[f'prec/avg_{suffix}'] = prec_list[-1]

    assert len(jaccard) == len(sam_miss) == len(sam_direct) == SAM_AMOUNT
    for i in range(SAM_AMOUNT):
        log_dict[f'IoU/{i + 1}_{suffix}'] = jaccard[i]
        log_dict[f'sam_miss/{i + 1}_{suffix}'] = sam_miss[i]  # previous metric name was "sam_att_miss"
        log_dict[f'sam_direct/{i + 1}_{suffix}'] = sam_direct[i]  # previous metric name was "sam_att_direct"
    return log_dict


def wandb_log_val(epoch, loss_sum, loss_add, loss_main):
    if not is_server:
        return
    global f1_val_best, mAP_val_best, prec_val_best, recall_val_best
    global sam_att_miss_val, sam_att_miss_val_best, iou_val, iou_val_best, gradcam_miss_att_val, gradcam_miss_att_val_best
    global sam_att_direct_val, sam_att_direct_val_best
    global gradcam_direct_att_val, gradcam_direct_att_val_best

    mAP_list = count_mAP()
    prec_list = count_precision()
    recall_list = count_recall()
    f1_list = count_f1()

    # lists contains metric value of each class + average value on the last slot
    assert len(f1_list) == len(recall_list) == len(prec_list) == len(mAP_list) == CLASS_AMOUNT + 1

    for i in range(CLASS_AMOUNT + 1):
        f1_val_best[i] = max(f1_list[i], f1_val_best[i])
        mAP_val_best[i] = max(mAP_list[i], mAP_val_best[i])
        prec_val_best[i] = max(prec_list[i], prec_val_best[i])
        recall_val_best[i] = max(recall_list[i], recall_val_best[i])

    # attention metrics
    iou_avg = [-1. for _ in range(SAM_AMOUNT)]
    sam_att_miss_avg = [-1. for _ in range(SAM_AMOUNT)]
    sam_att_direct_avg = [-1. for _ in range(SAM_AMOUNT)]

    for j in range(SAM_AMOUNT):
        assert len(iou_val[j]) == VAL_AMOUNT
        assert len(sam_att_miss_val[j]) == VAL_AMOUNT
        assert len(sam_att_direct_val[j]) == VAL_AMOUNT

        iou_avg[j] = sum(iou_val[j]) / len(iou_val[j])
        sam_att_miss_avg[j] = sum(sam_att_miss_val[j]) / len(sam_att_miss_val[j])
        sam_att_direct_avg[j] = sum(sam_att_direct_val[j]) / len(sam_att_direct_val[j])

    for j in range(SAM_AMOUNT):
        iou_val_best[j] = max(iou_val_best[j], iou_avg[j])
        sam_att_miss_val_best[j] = min(sam_att_miss_val_best[j], sam_att_miss_avg[j])
        sam_att_direct_val_best[j] = max(sam_att_direct_val_best[j], sam_att_direct_avg[j])

    assert len(gradcam_direct_att_val) == len(gradcam_miss_att_val) == VAL_AMOUNT
    avg_gradcam_miss_att = sum(gradcam_miss_att_val) / len(gradcam_miss_att_val)
    avg_gradcam_direct_att = sum(gradcam_direct_att_val) / len(gradcam_direct_att_val)

    gradcam_miss_att_val_best = min(gradcam_miss_att_val_best, avg_gradcam_miss_att)
    gradcam_direct_att_val_best = max(gradcam_direct_att_val_best, avg_gradcam_direct_att)

    for j in range(SAM_AMOUNT):
        iou_val[j].clear()
        sam_att_miss_val[j].clear()
        sam_att_direct_val[j].clear()

    gradcam_miss_att_val.clear()
    gradcam_direct_att_val.clear()

    dict_for_log = make_dict_for_log("val", loss_sum=loss_sum, loss_add=loss_add, loss_main=loss_main,
                                     mAP_list=mAP_list, prec_list=prec_list, recall_list=recall_list, f1_list=f1_list,
                                     jaccard=iou_avg, sam_miss=sam_att_miss_avg, sam_direct=sam_att_direct_avg,
                                     gc_miss=avg_gradcam_miss_att, gc_direct=avg_gradcam_direct_att)
    wandb.log(dict_for_log, step=epoch)


def save_summary():
    if not is_server:
        return
    print("saving summary..")
    # train
    for i in range(SAM_AMOUNT):
        run.summary[f"f1/с{i + 1}_trn'"] = f1_trn_best[i]
        run.summary[f"mAP/с{i + 1}_trn'"] = mAP_trn_best[i]
        run.summary[f"prec/с{i + 1}_trn'"] = prec_trn_best[i]
        run.summary[f"recall/с{i + 1}_trn'"] = recall_trn_best[i]

    run.summary["f1/avg_trn'"] = f1_trn_best[-1]
    run.summary["mAP/avg_trn'"] = mAP_trn_best[-1]
    run.summary["prec/avg_trn'"] = prec_trn_best[-1]
    run.summary["recall/avg_trn'"] = recall_trn_best[-1]

    # val
    for i in range(SAM_AMOUNT):
        run.summary[f"f1/с{i + 1}_val'"] = f1_val_best[i]
        run.summary[f"mAP/с{i + 1}_val'"] = mAP_val_best[i]
        run.summary[f"prec/с{i + 1}_val'"] = prec_val_best[i]
        run.summary[f"recall/с{i + 1}_val'"] = recall_val_best[i]

    run.summary["f1/avg_val'"] = f1_val_best[-1]
    run.summary["mAP/avg_val'"] = mAP_val_best[-1]
    run.summary["prec/avg_val'"] = prec_val_best[-1]
    run.summary["recall/avg_val'"] = recall_val_best[-1]

    for j in range(SAM_AMOUNT):
        run.summary[f"sam_att_miss/{j + 1}_trn'"] = sam_att_miss_best[j]
        run.summary[f"sam_att_direct/{j + 1}_trn'"] = sam_att_direct_best[j]

        run.summary[f"sam_att_miss/{j + 1}_val'"] = sam_att_miss_val_best[j]
        run.summary[f"sam_att_direct/{j + 1}_val'"] = sam_att_direct_val_best[j]

        run.summary[f"IoU/{j + 1}_trn'"] = iou_best[j]
        run.summary[f"IoU/{j + 1}_val'"] = iou_val_best[j]

    run.summary["gradcam_miss_trn'"] = gradcam_miss_att_best
    run.summary["gradcam_miss_val'"] = gradcam_miss_att_val_best

    run.summary["gradcam_direct_trn'"] = gradcam_direct_att_best
    run.summary["gradcam_direct_val'"] = gradcam_direct_att_val_best


def measure_accuracy(output, target):
    th = 0.5
    sigmoid = nn.Sigmoid()
    activated_output = sigmoid(output)
    activated_output = (activated_output > th).float()
    write_expected_predicted(target, activated_output)


def calculate_iou(true_mask, sam_output):
    """
    @type sam_output: torch.Tensor SAM which contains only 0 and 1
    @type true_mask: torch.Tensor - true mask
    shapes: 1 x H x W
    """
    SMOOTH = 1e-8
    assert sam_output.shape == true_mask.shape
    sam_output = sam_output.squeeze()
    true_mask = true_mask.squeeze()

    intersection = (sam_output & true_mask).sum()
    union = (sam_output | true_mask).sum()

    iou_res = (intersection + SMOOTH) / (union + SMOOTH)
    rounded = np.round(iou_res, 5)

    return rounded


def save_checkpoint_to_folder(state, folder_name, checkpoint_number):
    if not os.path.exists(f'./checkpoints/{folder_name}/'):
        os.mkdir(f'./checkpoints/{folder_name}/')
    filename = f'./checkpoints/{folder_name}/{int(checkpoint_number)}.pth'
    torch.save(state, filename)
    print("successfully saved checkpoint")


def create_needed_folders_for_hists():
    if not os.path.exists('hists'):
        os.mkdir('hists')
    for j in range(SAM_AMOUNT):
        if not os.path.exists(f'hists/SAM_{j + 1}'):
            os.mkdir(f'hists/SAM_{j + 1}')


def safe_division(a, b):
    return 0 if b == 0 else a / b


if __name__ == '__main__':
    main()
