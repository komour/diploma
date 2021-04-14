import argparse
import os
import shutil
import time
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
from math import inf
import sys

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

if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')
label_file = 'images-onehot.txt'

c1_expected = []
c1_predicted = []

c2_expected = []
c2_predicted = []

c3_expected = []
c3_predicted = []

c4_expected = []
c4_predicted = []

c5_expected = []
c5_predicted = []

args = parser.parse_args()
is_server = args.is_server == 1
SAM_AMOUNT = 3

avg_f1_best = 0
avg_f1_val_best = 0
avg_mAP_best = 0
avg_mAP_val_best = 0
avg_prec_best = 0
avg_prec_val_best = 0
avg_recall_best = 0
avg_recall_val_best = 0

c1_f1_best = 0
c1_f1_val_best = 0
c2_f1_best = 0
c2_f1_val_best = 0
c3_f1_best = 0
c3_f1_val_best = 0
c4_f1_best = 0
c4_f1_val_best = 0
c5_f1_best = 0
c5_f1_val_best = 0

c1_mAP_best = 0
c1_mAP_val_best = 0
c2_mAP_best = 0
c2_mAP_val_best = 0
c3_mAP_best = 0
c3_mAP_val_best = 0
c4_mAP_best = 0
c4_mAP_val_best = 0
c5_mAP_best = 0
c5_mAP_val_best = 0

c1_prec_best = 0
c1_prec_val_best = 0
c2_prec_best = 0
c2_prec_val_best = 0
c3_prec_best = 0
c3_prec_val_best = 0
c4_prec_best = 0
c4_prec_val_best = 0
c5_prec_best = 0
c5_prec_val_best = 0

c1_recall_best = 0
c1_recall_val_best = 0
c2_recall_best = 0
c2_recall_val_best = 0
c3_recall_best = 0
c3_recall_val_best = 0
c4_recall_best = 0
c4_recall_val_best = 0
c5_recall_best = 0
c5_recall_val_best = 0

# SAM miss attention metric
sam_att = [[] for _ in range(SAM_AMOUNT)]
sam_att_best = [inf for _ in range(SAM_AMOUNT)]

sam_att_val = [[] for _ in range(SAM_AMOUNT)]
sam_att_val_best = [inf for _ in range(SAM_AMOUNT)]

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
    CLASS_AMOUNT = 5
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
        model = ResidualNet('ImageNet', args.depth, CLASS_AMOUNT, 'BAM')
    else:
        model = ResidualNet('ImageNet', args.depth, CLASS_AMOUNT, 'CBAM')

    # model = torch.nn.DataParallel(model, device_ids=list(range(4)), output_device=args.cuda_device)

    # define loss function (criterion) and optimizer
    pos_weight_train = torch.Tensor(
        [[3.27807486631016, 2.7735849056603774, 12.91304347826087, 0.6859852476290832, 25.229508196721312]])
    if is_server:
        # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_train).cuda()
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_train).cuda(args.cuda_device)
        # sam_criterion_inv = nn.BCELoss(reduction='none').cuda()
        sam_criterion_inv = nn.BCELoss(reduction='none').cuda(args.cuda_device)
        # sam_criterion = nn.BCELoss().cuda()
        sam_criterion = nn.BCELoss().cuda(args.cuda_device)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_train)
        sam_criterion_inv = nn.BCELoss(reduction='none')
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
        lmbd = args.lmbd
    )
    if is_server:
        run = wandb.init(config=config, project="vol.6", name=args.run_name, tags=args.tags)

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
            checkpoint = torch.load(args.resume)
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
    root_dir = 'data/'
    traindir = os.path.join(root_dir, 'train')
    train_labels = os.path.join(root_dir, 'train', 'images_onehot_train.txt')
    valdir = os.path.join(root_dir, 'val')
    val_labels = os.path.join(root_dir, 'val', 'images_onehot_val.txt')
    # testdir = os.path.join(args.data, 'test')
    # test_labels = os.path.join(args.data, 'test', 'images_onehot_test.txt')

    # import pdb
    # pdb.set_trace()
    size0 = 224
    val_dataset = DatasetISIC2018(
        val_labels,
        valdir,
        False,
        False,
        transforms.CenterCrop(size0)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    if args.evaluate:
        validate(val_loader, model, criterion, 0, optimizer, args.epochs)
        return

    train_dataset = DatasetISIC2018(
        train_labels,
        traindir,
        True,  # perform flips
        True  # perform random resized crop with size = 224
    )

    # test_dataset = DatasetISIC2018(
    #     test_labels,
    #     testdir,
    #     False,
    #     False,
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
        train(train_loader, model, criterion, sam_criterion, sam_criterion_inv, optimizer, epoch, epoch_number)

        # evaluate on validation set
        clear_expected_predicted()
        validate(val_loader, model, criterion, epoch, optimizer, epoch_number)
        epoch_number += 1

    save_summary()
    run.finish()


def train(train_loader, model, criterion, sam_criterion, sam_criterion_inv, optimizer, epoch, epoch_number):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    #  decide whether to save checkpoint
    # if epoch_number % 10 == 0:
    #     checkpoint_dict = {
    #         'epoch': epoch,
    #         'state_dict': model.state_dict(),
    #         'optimizer': optimizer.state_dict()
    #     }
    #     save_checkpoint_to_folder(checkpoint_dict, args.run_name, epoch_number / 10 + 1)

    global iou, sam_att, gradcam_miss_att, gradcam_direct_att
    for i, dictionary in enumerate(train_loader):
        input_img = dictionary['image']
        target = dictionary['label']
        segm = dictionary['segm']
        if is_server:
            input_img = input_img.cuda(args.cuda_device)
            # input_img = input_img.cuda()
            target = target.cuda(args.cuda_device)
            # target = target.cuda()
            segm = segm.cuda(args.cuda_device)
            # segm = segm.cuda()

        # measure data loading time
        data_time.update(time.time() - end)

        # sam_output shapes:
        # [1, 1, 56, 56]x3 , [1, 1, 28, 28]x4 [1, 1, 14, 14]x6 , [1, 1, 7, 7]x3

        # visualize masks/input_image/sam_output
        # np_sam = torch.squeeze(sam_output[0]).detach().numpy()
        # np_segm = np.moveaxis(torch.squeeze(segm).detach().numpy(), 0, -1)
        # np_input = np.moveaxis(torch.squeeze(input_img).detach().numpy(), 0, -1)
        # plt.imshow(np_segm.astype(np.uint8))
        # plt.imshow(np_sam)
        # plt.show()

        # initial segm size = [1, 3, 224, 224]
        maxpool_segm1 = nn.MaxPool3d(kernel_size=(3, 4, 4))
        maxpool_segm2 = nn.MaxPool3d(kernel_size=(3, 8, 8))
        maxpool_segm3 = nn.MaxPool3d(kernel_size=(3, 16, 16))
        # maxpool_segm4 = nn.MaxPool3d(kernel_size=(3, 32, 32))
        maxpool_segm0 = nn.MaxPool3d(kernel_size=(3, 1, 1))
        #
        processed_segm1 = maxpool_segm1(segm)
        processed_segm2 = maxpool_segm2(segm)
        processed_segm3 = maxpool_segm3(segm)
        # processed_segm4 = maxpool_segm4(segm)
        processed_segm0 = maxpool_segm0(segm)
        #
        processed_segm1_invert = (processed_segm1 + 1) % 2
        processed_segm2_invert = (processed_segm2 + 1) % 2
        processed_segm3_invert = (processed_segm3 + 1) % 2
        # processed_segm4_invert = (processed_segm4 + 1) % 2

        true_mask_invert = (processed_segm0 + 1) % 2

        invert_masks = [processed_segm1_invert, processed_segm2_invert, processed_segm3_invert]

        masks = [processed_segm1, processed_segm2, processed_segm3]

        # compute output
        # output, sam_output = model(input_img)
        # output = model(input_img)

        # calculate gradcam metrics + compute output
        target_layer = model.layer4
        gradcam = GradCAM(model, target_layer=target_layer)

        if args.number == -1:
            gc_mask, no_norm_gc_mask, output, sam_output = gradcam(input_img, retain_graph=True, masks=invert_masks)
        else:
            gc_mask, no_norm_gc_mask, output, sam_output = gradcam(input_img, retain_graph=True)

        # measure gradcam attention metrics
        true_mask_invert = true_mask_invert.detach().cpu().numpy()
        true_mask = processed_segm0.detach().cpu().numpy()
        no_norm_gc_mask_numpy = no_norm_gc_mask.detach().cpu().numpy()

        assert true_mask.shape == true_mask_invert.shape
        assert no_norm_gc_mask_numpy.shape == true_mask_invert.shape
        for j in range(len(no_norm_gc_mask_numpy)):
            cur_gc = no_norm_gc_mask_numpy[j]
            cur_mask = true_mask[j]
            cur_mask_inv = true_mask_invert[j]

            gradcam_miss_att.append(safe_division(np.sum(cur_gc * cur_mask_inv), np.sum(cur_mask_inv)))
            gradcam_direct_att.append(safe_division(np.sum(cur_gc * cur_mask), np.sum(cur_mask)))

        loss0 = criterion(output, target)

        assert len(processed_segm1) == len(processed_segm2) == len(processed_segm3)
        loss_inv_sum1 = 0
        loss_inv_sum2 = 0
        loss_inv_sum3 = 0
        for j in range(len(processed_segm1)):
            loss_inv_sum1 += safe_division(
                torch.sum(sam_criterion_inv(sam_output[0][j], processed_segm1[j]) * processed_segm1_invert[j]),
                torch.sum(processed_segm1_invert[j]))

            loss_inv_sum2 += safe_division(
                torch.sum(sam_criterion_inv(sam_output[1][j], processed_segm2[j]) * processed_segm2_invert[j]),
                torch.sum(processed_segm2_invert[j]))

            loss_inv_sum3 += safe_division(
                torch.sum(sam_criterion_inv(sam_output[2][j], processed_segm3[j]) * processed_segm3_invert[j]),
                torch.sum(processed_segm3_invert[j]))

        loss1_inv = args.lmbd * loss_inv_sum1 / args.batch_size
        loss2_inv = args.lmbd * loss_inv_sum2 / args.batch_size
        loss3_inv = args.lmbd * loss_inv_sum3 / args.batch_size

        loss1 = args.lmbd * sam_criterion(sam_output[0], processed_segm1)
        loss2 = args.lmbd * sam_criterion(sam_output[1], processed_segm2)
        loss3 = args.lmbd * sam_criterion(sam_output[2], processed_segm3)

        loss_comb = loss0
        if args.number == 1:
            loss_comb += loss1
        elif args.number == 2:
            loss_comb += loss2
        elif args.number == 3:
            loss_comb += loss3
        elif args.number == 5:
            loss_comb += loss1 + loss2 + loss3
        elif args.number == 10:
            loss_comb += loss1_inv
        elif args.number == 20:
            loss_comb += loss2_inv
        elif args.number == 30:
            loss_comb += loss3_inv
        elif args.number == 50:
            loss_comb += loss1_inv + loss2_inv + loss3_inv

        # measure sam attention metrics
        for j in range(SAM_AMOUNT):
            predicted_sam = sam_output[j].detach().cpu().numpy()

            predicted = (sam_output[j] > 0.5).int()
            predicted_np = predicted.detach().cpu().numpy()

            expected = masks[j].int()
            expected_np = expected.detach().cpu().numpy()

            for k in range(len(predicted_sam)):
                cur_sam = predicted_sam[k]
                cur_pred = predicted_np[k]
                cur_expected = expected_np[k]

                sam_att[j].append(safe_division(np.sum(cur_sam * invert_masks[j][k].cpu().numpy()),
                                                np.sum(invert_masks[j][k].cpu().numpy())))
                iou[j].append(iou_numpy(cur_expected, cur_pred))

        # measure accuracy and record loss
        measure_accuracy(output.data, target)

        losses.update(loss_comb.item(), input_img.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_comb.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f'\nEpoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})')
            if i > 0:
                print_metrics()
    wandb_log_train(epoch, losses.avg)


def validate(val_loader, model, criterion, epoch, optimizer, epoch_number):
    batch_time = AverageMeter()
    losses = AverageMeter()
    # switch to evaluate mode
    model.eval()
    global val_vis_image_names
    end = time.time()

    # sam_concat = [None for _ in range(SAM_AMOUNT)]
    global iou_val, sam_att_val, gradcam_miss_att_val, gradcam_direct_att_val
    for i, dictionary in enumerate(val_loader):
        input_img = dictionary['image']
        target = dictionary['label']
        segm = dictionary['segm']
        if is_server:
            # input_img = input_img.cuda()
            input_img = input_img.cuda(args.cuda_device)
            # target = target.cuda()
            target = target.cuda(args.cuda_device)
            # segm = segm.cuda()
            segm = segm.cuda(args.cuda_device)

        maxpool_segm1 = nn.MaxPool3d(kernel_size=(3, 4, 4))
        maxpool_segm2 = nn.MaxPool3d(kernel_size=(3, 8, 8))
        maxpool_segm3 = nn.MaxPool3d(kernel_size=(3, 16, 16))
        # maxpool_segm4 = nn.MaxPool3d(kernel_size=(3, 32, 32))
        maxpool_segm0 = nn.MaxPool3d(kernel_size=(3, 1, 1))

        processed_segm1 = maxpool_segm1(segm)
        processed_segm2 = maxpool_segm2(segm)
        processed_segm3 = maxpool_segm3(segm)
        # processed_segm4 = maxpool_segm4(segm)
        processed_segm0 = maxpool_segm0(segm)

        processed_segm1_invert = (processed_segm1 + 1) % 2
        processed_segm2_invert = (processed_segm2 + 1) % 2
        processed_segm3_invert = (processed_segm3 + 1) % 2
        # processed_segm4_invert = (processed_segm4 + 1) % 2
        true_mask_invert = (processed_segm0 + 1) % 2

        invert_mask = [processed_segm1_invert, processed_segm2_invert, processed_segm3_invert]

        mask = [processed_segm1, processed_segm2, processed_segm3]

        # compute output
        # with torch.no_grad():
        #     output, sam_output = model(input_img)
        # output = model(input_img)
        # loss = criterion(output, target)
        # loss = CB_loss(target, output)

        target_layer = model.layer4
        gradcam = GradCAM(model, target_layer=target_layer)

        if args.number == -1:
            gc_mask, no_norm_gc_mask, output, sam_output = gradcam(input_img, masks=invert_mask)
        else:
            gc_mask, no_norm_gc_mask, output, sam_output = gradcam(input_img)

        # measure attention metrics
        true_mask_invert = true_mask_invert.detach().cpu().numpy()
        true_mask = processed_segm0.detach().cpu().numpy()
        no_norm_gc_mask_numpy = no_norm_gc_mask.detach().cpu().numpy()

        assert true_mask.shape == true_mask_invert.shape
        assert no_norm_gc_mask_numpy.shape == true_mask_invert.shape

        for j in range(len(no_norm_gc_mask_numpy)):
            cur_gc = no_norm_gc_mask_numpy[j]
            cur_mask = true_mask[j]
            cur_mask_inv = true_mask_invert[j]
            gradcam_miss_att_val.append(safe_division(np.sum(cur_gc * cur_mask_inv), np.sum(cur_mask_inv)))
            gradcam_direct_att_val.append(safe_division(np.sum(cur_gc * cur_mask), np.sum(cur_mask)))

        loss = criterion(output, target)

        for j in range(SAM_AMOUNT):
            # sam_concat[j] = predicted_sam if sam_concat[j] is None \
            #     else np.concatenate((sam_concat[j], predicted_sam), axis=0)

            predicted_sam = sam_output[j].detach().cpu().numpy()

            predicted = (sam_output[j] > 0.5).int()
            predicted_np = predicted.detach().cpu().numpy()

            expected = mask[j].int()
            expected_np = expected.detach().cpu().numpy()
            for k in range(len(predicted_sam)):
                cur_sam = predicted_sam[k]
                cur_pred = predicted_np[k]
                cur_expected = expected_np[k]

                sam_att_val[j].append(safe_division(np.sum(cur_sam * invert_mask[j][k].cpu().numpy()),
                                                    np.sum(invert_mask[j][k].cpu().numpy())))
                iou_val[j].append(iou_numpy(cur_expected, cur_pred))

        # measure accuracy and record loss
        measure_accuracy(output.data, target)
        losses.update(loss.item(), input_img.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f'Validate: [{i}/{len(val_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})')
            if i != 0:
                print_metrics()
    # c1_f1, c2_f1, c3_f1, c4_f1, c5_f1, avg_f1 = count_f1()
    # if avg_f1 > avg_f1_val_best:
    #     save_checkpoint({
    #         'epoch': epoch,
    #         'state_dict': model.state_dict(),
    #         'optimizer': optimizer.state_dict()
    #     }, args.run_name)

    # for histograms:
    # for j in range(SAM_AMOUNT):
    #     values = sam_concat[j].ravel()
    #     plt.close('all')
    #     plt.hist(values)
    #     plt.savefig(f'hists/SAM_{j + 1}/e{epoch + 1}.pdf')
    wandb_log_val(epoch, losses.avg)


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
    for t in target:
        c1_expected.append(t[0])
        c2_expected.append(t[1])
        c3_expected.append(t[2])
        c4_expected.append(t[3])
        c5_expected.append(t[4])

    for o in output:
        c1_predicted.append(o[0])
        c2_predicted.append(o[1])
        c3_predicted.append(o[2])
        c4_predicted.append(o[3])
        c5_predicted.append(o[4])


def clear_expected_predicted():
    c1_expected.clear()
    c2_expected.clear()
    c3_expected.clear()
    c4_expected.clear()
    c5_expected.clear()

    c1_predicted.clear()
    c2_predicted.clear()
    c3_predicted.clear()
    c4_predicted.clear()
    c5_predicted.clear()


def count_mAP():
    c1_pred = np.asarray(c1_predicted).astype(float)
    c2_pred = np.asarray(c2_predicted).astype(float)
    c3_pred = np.asarray(c3_predicted).astype(float)
    c4_pred = np.asarray(c4_predicted).astype(float)
    c5_pred = np.asarray(c5_predicted).astype(float)

    c1_exp = np.asarray(c1_expected).astype(float)
    c2_exp = np.asarray(c2_expected).astype(float)
    c3_exp = np.asarray(c3_expected).astype(float)
    c4_exp = np.asarray(c4_expected).astype(float)
    c5_exp = np.asarray(c5_expected).astype(float)

    c1_mAP = average_precision_score(c1_exp, c1_pred)
    c2_mAP = average_precision_score(c2_exp, c2_pred)
    c3_mAP = average_precision_score(c3_exp, c3_pred)
    c4_mAP = average_precision_score(c4_exp, c4_pred)
    c5_mAP = average_precision_score(c5_exp, c5_pred)
    avg_mAP = (c1_mAP + c2_mAP + c3_mAP + c4_mAP + c5_mAP) / 5
    return c1_mAP, c2_mAP, c3_mAP, c4_mAP, c5_mAP, avg_mAP


def count_precision():
    c1_pred = np.asarray(c1_predicted).astype(float)
    c2_pred = np.asarray(c2_predicted).astype(float)
    c3_pred = np.asarray(c3_predicted).astype(float)
    c4_pred = np.asarray(c4_predicted).astype(float)
    c5_pred = np.asarray(c5_predicted).astype(float)

    c1_exp = np.asarray(c1_expected).astype(float)
    c2_exp = np.asarray(c2_expected).astype(float)
    c3_exp = np.asarray(c3_expected).astype(float)
    c4_exp = np.asarray(c4_expected).astype(float)
    c5_exp = np.asarray(c5_expected).astype(float)

    c1_prec = precision_score(c1_exp, c1_pred, average="binary")
    c2_prec = precision_score(c2_exp, c2_pred, average="binary")
    c3_prec = precision_score(c3_exp, c3_pred, average="binary")
    c4_prec = precision_score(c4_exp, c4_pred, average="binary")
    c5_prec = precision_score(c5_exp, c5_pred, average="binary")
    avg_prec = (c1_prec + c2_prec + c3_prec + c4_prec + c5_prec) / 5
    return c1_prec, c2_prec, c3_prec, c4_prec, c5_prec, avg_prec


def count_recall():
    c1_pred = np.asarray(c1_predicted).astype(float)
    c2_pred = np.asarray(c2_predicted).astype(float)
    c3_pred = np.asarray(c3_predicted).astype(float)
    c4_pred = np.asarray(c4_predicted).astype(float)
    c5_pred = np.asarray(c5_predicted).astype(float)

    c1_exp = np.asarray(c1_expected).astype(float)
    c2_exp = np.asarray(c2_expected).astype(float)
    c3_exp = np.asarray(c3_expected).astype(float)
    c4_exp = np.asarray(c4_expected).astype(float)
    c5_exp = np.asarray(c5_expected).astype(float)

    c1_recall = recall_score(c1_exp, c1_pred, average="binary")
    c2_recall = recall_score(c2_exp, c2_pred, average="binary")
    c3_recall = recall_score(c3_exp, c3_pred, average="binary")
    c4_recall = recall_score(c4_exp, c4_pred, average="binary")
    c5_recall = recall_score(c5_exp, c5_pred, average="binary")
    avg_recall = (c1_recall + c2_recall + c3_recall + c4_recall + c5_recall) / 5
    return c1_recall, c2_recall, c3_recall, c4_recall, c5_recall, avg_recall


def count_f1():
    c1_pred = np.asarray(c1_predicted).astype(float)
    c2_pred = np.asarray(c2_predicted).astype(float)
    c3_pred = np.asarray(c3_predicted).astype(float)
    c4_pred = np.asarray(c4_predicted).astype(float)
    c5_pred = np.asarray(c5_predicted).astype(float)

    c1_exp = np.asarray(c1_expected).astype(float)
    c2_exp = np.asarray(c2_expected).astype(float)
    c3_exp = np.asarray(c3_expected).astype(float)
    c4_exp = np.asarray(c4_expected).astype(float)
    c5_exp = np.asarray(c5_expected).astype(float)

    c1_f1 = f1_score(c1_exp, c1_pred, average="binary")
    c2_f1 = f1_score(c2_exp, c2_pred, average="binary")
    c3_f1 = f1_score(c3_exp, c3_pred, average="binary")
    c4_f1 = f1_score(c4_exp, c4_pred, average="binary")
    c5_f1 = f1_score(c5_exp, c5_pred, average="binary")
    avg_f1 = (c1_f1 + c2_f1 + c3_f1 + c4_f1 + c5_f1) / 5
    return c1_f1, c2_f1, c3_f1, c4_f1, c5_f1, avg_f1


def wandb_log_train(epoch, loss_avg):
    if not is_server:
        return

    global c1_f1_best, c2_f1_best, c3_f1_best, c4_f1_best, c5_f1_best
    global c1_mAP_best, c2_mAP_best, c3_mAP_best, c4_mAP_best, c5_mAP_best
    global c1_recall_best, c2_recall_best, c3_recall_best, c4_recall_best, c5_recall_best
    global c1_prec_best, c2_prec_best, c3_prec_best, c4_prec_best, c5_prec_best
    global avg_f1_best, avg_mAP_best, avg_recall_best, avg_prec_best
    global sam_att, sam_att_best, iou, iou_best, gradcam_miss_att, gradcam_miss_att_best
    global gradcam_direct_att, gradcam_direct_att_best

    c1_mAP, c2_mAP, c3_mAP, c4_mAP, c5_mAP, avg_mAP = count_mAP()
    c1_prec, c2_prec, c3_prec, c4_prec, c5_prec, avg_prec = count_precision()
    c1_recall, c2_recall, c3_recall, c4_recall, c5_recall, avg_recall = count_recall()
    c1_f1, c2_f1, c3_f1, c4_f1, c5_f1, avg_f1 = count_f1()

    if avg_f1 > avg_f1_best:
        avg_f1_best = avg_f1
    if avg_mAP > avg_mAP_best:
        avg_mAP_best = avg_mAP
    if avg_recall > avg_recall_best:
        avg_recall_best = avg_recall
    if avg_prec > avg_prec_best:
        avg_prec_best = avg_prec

    if c1_mAP > c1_mAP_best:
        c1_mAP_best = c1_mAP
    if c2_mAP > c2_mAP_best:
        c2_mAP_best = c2_mAP
    if c3_mAP > c3_mAP_best:
        c3_mAP_best = c3_mAP
    if c4_mAP > c4_mAP_best:
        c4_mAP_best = c4_mAP
    if c5_mAP > c5_mAP_best:
        c5_mAP_best = c5_mAP

    if c1_prec > c1_prec_best:
        c1_prec_best = c1_prec
    if c2_prec > c2_prec_best:
        c2_prec_best = c2_prec
    if c3_prec > c3_prec_best:
        c3_prec_best = c3_prec
    if c4_prec > c4_prec_best:
        c4_prec_best = c4_prec
    if c5_prec > c5_prec_best:
        c5_prec_best = c5_prec

    if c1_recall > c1_recall_best:
        c1_recall_best = c1_recall
    if c2_recall > c2_recall_best:
        c2_recall_best = c2_recall
    if c3_recall > c3_recall_best:
        c3_recall_best = c3_recall
    if c4_recall > c4_recall_best:
        c4_recall_best = c4_recall
    if c5_recall > c5_recall_best:
        c5_recall_best = c5_recall

    if c1_f1 > c1_f1_best:
        c1_f1_best = c1_f1
    if c2_f1 > c2_f1_best:
        c2_f1_best = c2_f1
    if c3_f1 > c3_f1_best:
        c3_f1_best = c3_f1
    if c4_f1 > c4_f1_best:
        c4_f1_best = c4_f1
    if c5_f1 > c5_f1_best:
        c5_f1_best = c5_f1

    # attention metrics
    iou_avg = [-1. for _ in range(SAM_AMOUNT)]
    sam_att_avg = [-1. for _ in range(SAM_AMOUNT)]

    for j in range(SAM_AMOUNT):
        assert len(iou[j]) == 1600
        assert len(sam_att[j]) == 1600

        iou_avg[j] = sum(iou[j]) / len(iou[j])
        sam_att_avg[j] = sum(sam_att[j]) / len(sam_att[j])

    for j in range(SAM_AMOUNT):
        iou[j] = []
        sam_att[j] = []

    for j in range(SAM_AMOUNT):
        iou_best[j] = max(iou_best[j], iou_avg[j])
        sam_att_best[j] = min(sam_att_best[j], sam_att_avg[j])

    assert len(gradcam_direct_att_val) == len(gradcam_miss_att_val) == 1600
    avg_gradcam_miss_att = sum(gradcam_miss_att) / len(gradcam_miss_att)
    avg_gradcam_direct_att = sum(gradcam_direct_att) / len(gradcam_direct_att)

    gradcam_miss_att_best = min(gradcam_miss_att_best, avg_gradcam_miss_att)
    gradcam_direct_att_best = max(gradcam_direct_att_best, avg_gradcam_direct_att)

    gradcam_miss_att = []
    gradcam_direct_att = []

    wandb.log({"loss_avg_trn": loss_avg,
               "mAP/с1_trn": c1_mAP, "mAP/с2_trn": c2_mAP, "mAP/с3_trn": c3_mAP, "mAP/с4_trn": c4_mAP,
               "mAP/с5_trn": c5_mAP,
               "mAP/avg_trn": avg_mAP,
               "prec/c1_trn": c1_prec, "prec/c2_trn": c2_prec, "prec/c3_trn": c3_prec,
               "prec/c4_trn": c4_prec, "prec/c5_trn": c5_prec, "prec/avg_trn": avg_prec,
               "recall/c1_trn": c1_recall, "recall/c2_trn": c2_recall, "recall/c3_trn": c3_recall,
               "recall/c4_trn": c4_recall,
               "recall/c5_trn": c5_recall, "recall/avg_trn": avg_recall,
               "f1/c1_trn": c1_f1, "f1/c2_trn": c2_f1, "f1/c3_trn": c3_f1, "f1/c4_trn": c4_f1, "f1/c5_trn": c5_f1,
               "f1/avg_trn": avg_f1,
               "IoU/1_trn": iou_avg[0], "IoU/2_trn": iou_avg[1], "IoU/3_trn": iou_avg[2],  # , "IoU/4_trn": iou_avg[3],
               # "IoU/5_trn": iou_avg[4],
               # "IoU/6_trn": iou_avg[5], "IoU/7_trn": iou_avg[6], "IoU/8_trn": iou_avg[7], "IoU/9_trn": iou_avg[8],
               # "IoU/10_trn": iou_avg[9],
               # "IoU/11_trn": iou_avg[10], "IoU/12_trn": iou_avg[11], "IoU/13_trn": iou_avg[12],
               # "IoU/14_trn": iou_avg[13],
               # "IoU/15_trn": iou_avg[14], "IoU/16_trn": iou_avg[15],
               "sam_att/1_trn": sam_att_avg[0], "sam_att/2_trn": sam_att_avg[1], "sam_att/3_trn": sam_att_avg[2],
               # "sam_att/4_trn": sam_att_avg[3], "sam_att/5_trn": sam_att_avg[4], "sam_att/6_trn": sam_att_avg[5],
               # "sam_att/7_trn": sam_att_avg[6], "sam_att/8_trn": sam_att_avg[7], "sam_att/9_trn": sam_att_avg[8],
               # "sam_att/10_trn": sam_att_avg[9], "sam_att/11_trn": sam_att_avg[10], "sam_att/12_trn": sam_att_avg[11],
               # "sam_att/13_trn": sam_att_avg[12], "sam_att/14_trn": sam_att_avg[13],
               # "sam_att/15_trn": sam_att_avg[14], "sam_att/16_trn": sam_att_avg[15],
               "gradcam_miss_trn": avg_gradcam_miss_att,
               "gradcam_direct_trn": avg_gradcam_direct_att
               },
              step=epoch)


def wandb_log_val(epoch, loss_avg):
    if not is_server:
        return

    global c1_f1_val_best, c2_f1_val_best, c3_f1_val_best, c4_f1_val_best, c5_f1_val_best
    global c1_mAP_val_best, c2_mAP_val_best, c3_mAP_val_best, c4_mAP_val_best, c5_mAP_val_best
    global c1_recall_val_best, c2_recall_val_best, c3_recall_val_best, c4_recall_val_best, c5_recall_val_best
    global c1_prec_val_best, c2_prec_val_best, c3_prec_val_best, c4_prec_val_best, c5_prec_val_best
    global avg_f1_val_best, avg_mAP_val_best, avg_recall_val_best, avg_prec_val_best
    global sam_att_val, sam_att_val_best, iou_val, iou_val_best, gradcam_miss_att_val, gradcam_miss_att_val_best
    global gradcam_direct_att_val, gradcam_direct_att_val_best

    c1_mAP, c2_mAP, c3_mAP, c4_mAP, c5_mAP, avg_mAP = count_mAP()
    c1_prec, c2_prec, c3_prec, c4_prec, c5_prec, avg_prec = count_precision()
    c1_recall, c2_recall, c3_recall, c4_recall, c5_recall, avg_recall = count_recall()
    c1_f1, c2_f1, c3_f1, c4_f1, c5_f1, avg_f1 = count_f1()

    if avg_f1 > avg_f1_val_best:
        avg_f1_val_best = avg_f1
    if avg_mAP > avg_mAP_val_best:
        avg_mAP_val_best = avg_mAP
    if avg_recall > avg_recall_val_best:
        avg_recall_val_best = avg_recall
    if avg_prec > avg_prec_val_best:
        avg_prec_val_best = avg_prec

    if c1_mAP > c1_mAP_val_best:
        c1_mAP_val_best = c1_mAP
    if c2_mAP > c2_mAP_val_best:
        c2_mAP_val_best = c2_mAP
    if c3_mAP > c3_mAP_val_best:
        c3_mAP_val_best = c3_mAP
    if c4_mAP > c4_mAP_val_best:
        c4_mAP_val_best = c4_mAP
    if c5_mAP > c5_mAP_val_best:
        c5_mAP_val_best = c5_mAP

    if c1_prec > c1_prec_val_best:
        c1_prec_val_best = c1_prec
    if c2_prec > c2_prec_val_best:
        c2_prec_val_best = c2_prec
    if c3_prec > c3_prec_val_best:
        c3_prec_val_best = c3_prec
    if c4_prec > c4_prec_val_best:
        c4_prec_val_best = c4_prec
    if c5_prec > c5_prec_val_best:
        c5_prec_val_best = c5_prec

    if c1_recall > c1_recall_val_best:
        c1_recall_val_best = c1_recall
    if c2_recall > c2_recall_val_best:
        c2_recall_val_best = c2_recall
    if c3_recall > c3_recall_val_best:
        c3_recall_val_best = c3_recall
    if c4_recall > c4_recall_val_best:
        c4_recall_val_best = c4_recall
    if c5_recall > c5_recall_val_best:
        c5_recall_val_best = c5_recall

    if c1_f1 > c1_f1_val_best:
        c1_f1_val_best = c1_f1
    if c2_f1 > c2_f1_val_best:
        c2_f1_val_best = c2_f1
    if c3_f1 > c3_f1_val_best:
        c3_f1_val_best = c3_f1
    if c4_f1 > c4_f1_val_best:
        c4_f1_val_best = c4_f1
    if c5_f1 > c5_f1_val_best:
        c5_f1_val_best = c5_f1

    # attention metrics
    iou_avg = [-1. for _ in range(SAM_AMOUNT)]
    sam_att_avg = [-1. for _ in range(SAM_AMOUNT)]

    for j in range(SAM_AMOUNT):
        assert len(iou_val[j]) == 400
        assert len(sam_att_val[j]) == 400

        iou_avg[j] = sum(iou_val[j]) / len(iou_val[j])
        sam_att_avg[j] = sum(sam_att_val[j]) / len(sam_att_val[j])

    for j in range(SAM_AMOUNT):
        iou_val[j] = []
        sam_att_val[j] = []

    for j in range(SAM_AMOUNT):
        iou_val_best[j] = max(iou_val_best[j], iou_avg[j])
        sam_att_val_best[j] = min(sam_att_val_best[j], sam_att_avg[j])

    assert len(gradcam_direct_att_val) == len(gradcam_miss_att_val) == 400
    avg_gradcam_miss_att = sum(gradcam_miss_att_val) / len(gradcam_miss_att_val)
    avg_gradcam_direct_att = sum(gradcam_direct_att_val) / len(gradcam_direct_att_val)

    gradcam_miss_att_val_best = min(gradcam_miss_att_val_best, avg_gradcam_miss_att)
    gradcam_direct_att_val_best = max(gradcam_direct_att_val_best, avg_gradcam_direct_att)

    gradcam_miss_att_val = []
    gradcam_direct_att_val = []

    wandb.log({"loss_avg_val": loss_avg,
               "mAP/c1_val": c1_mAP, "mAP/c2_val": c2_mAP, "mAP/c3_val": c3_mAP, "mAP/c4_val": c4_mAP,
               "mAP/c5_val": c5_mAP,
               "mAP/avg_val": avg_mAP,
               "prec/c1_val": c1_prec, "prec/c2_val": c2_prec, "prec/c3_val": c3_prec,
               "prec/c4_val": c4_prec, "prec/c5_val": c5_prec,
               "prec/avg_val": avg_prec,
               "recall/c1_val": c1_recall, "recall/c2_val": c2_recall, "recall/c3_val": c3_recall,
               "recall/c4_val": c4_recall,
               "recall/c5_val": c5_recall, "recall/avg_val": avg_recall,
               "f1/c1_val": c1_f1, "f1/c2_val": c2_f1, "f1/c3_val": c3_f1, "f1/c4_val": c4_f1, "f1/c5_val": c5_f1,
               "f1/avg_val": avg_f1,
               "IoU/1_val": iou_avg[0], "IoU/2_val": iou_avg[1], "IoU/3_val": iou_avg[2],  # "IoU/4_val": iou_avg[3],
               # "IoU/5_val": iou_avg[4],
               # "IoU/6_val": iou_avg[5], "IoU/7_val": iou_avg[6], "IoU/8_val": iou_avg[7], "IoU/9_val": iou_avg[8],
               # "IoU/10_val": iou_avg[9],
               # "IoU/11_val": iou_avg[10], "IoU/12_val": iou_avg[11], "IoU/13_val": iou_avg[12],
               # "IoU/14_val": iou_avg[13],
               # "IoU/15_val": iou_avg[14], "IoU/16_val": iou_avg[15],
               "sam_att/1_val": sam_att_avg[0], "sam_att/2_val": sam_att_avg[1], "sam_att/3_val": sam_att_avg[2],
               # "sam_att/4_val": sam_att_avg[3], "sam_att/5_val": sam_att_avg[4], "sam_att/6_val": sam_att_avg[5],
               # "sam_att/7_val": sam_att_avg[6], "sam_att/8_val": sam_att_avg[7], "sam_att/9_val": sam_att_avg[8],
               # "sam_att/10_val": sam_att_avg[9], "sam_att/11_val": sam_att_avg[10], "sam_att/12_val": sam_att_avg[11],
               # "sam_att/13_val": sam_att_avg[12], "sam_att/14_val": sam_att_avg[13], "sam_att/15_val": sam_att_avg[14],
               # "sam_att/16_val": sam_att_avg[15],
               "gradcam_miss_val": avg_gradcam_miss_att,
               "gradcam_direct_val": avg_gradcam_direct_att
               },
              step=epoch)


def save_summary():
    if not is_server:
        return
    print("saving summary..")
    # train
    run.summary["f1/avg_trn'"] = avg_f1_best
    run.summary["mAP/avg_trn'"] = avg_mAP_best
    run.summary["recall/avg_trn'"] = avg_recall_best
    run.summary["prec/avg_trn'"] = avg_prec_best

    run.summary["mAP/с1_trn'"] = c1_mAP_best
    run.summary["mAP/с2_trn'"] = c2_mAP_best
    run.summary["mAP/с3_trn'"] = c3_mAP_best
    run.summary["mAP/с4_trn'"] = c4_mAP_best
    run.summary["mAP/с5_trn'"] = c5_mAP_best
    # print(f'mAP/с5_trn = {c5_mAP_best}')

    run.summary["prec/c1_trn'"] = c1_prec_best
    run.summary["prec/c2_trn'"] = c2_prec_best
    run.summary["prec/c3_trn'"] = c3_prec_best
    run.summary["prec/c4_trn'"] = c4_prec_best
    run.summary["prec/c5_trn'"] = c5_prec_best
    # print(f'prec/c5_trn = {c5_prec_best}')

    run.summary["recall/c1_trn'"] = c1_recall_best
    run.summary["recall/c2_trn'"] = c2_recall_best
    run.summary["recall/c3_trn'"] = c3_recall_best
    run.summary["recall/c4_trn'"] = c4_recall_best
    run.summary["recall/c5_trn'"] = c5_recall_best
    # print(f'recall/c5_trn = {c5_recall_best}')

    run.summary["f1/c1_trn'"] = c1_f1_best
    run.summary["f1/c2_trn'"] = c2_f1_best
    run.summary["f1/c3_trn'"] = c3_f1_best
    run.summary["f1/c4_trn'"] = c4_f1_best
    run.summary["f1/c5_trn'"] = c5_f1_best
    # print(f'f1/c5_trn = {c5_f1_best}')

    # val
    run.summary["f1/avg_val'"] = avg_f1_val_best
    run.summary["mAP/avg_val'"] = avg_mAP_val_best
    run.summary["recall/avg_val'"] = avg_recall_val_best
    run.summary["prec/avg_val'"] = avg_prec_val_best

    run.summary["mAP/c1_val'"] = c1_mAP_val_best
    run.summary["mAP/c2_val'"] = c2_mAP_val_best
    run.summary["mAP/c3_val'"] = c3_mAP_val_best
    run.summary["mAP/c4_val'"] = c4_mAP_val_best
    run.summary["mAP/c5_val'"] = c5_mAP_val_best

    run.summary["prec/c1_val'"] = c1_prec_val_best
    run.summary["prec/c2_val'"] = c2_prec_val_best
    run.summary["prec/c3_val'"] = c3_prec_val_best
    run.summary["prec/c4_val'"] = c4_prec_val_best
    run.summary["prec/c5_val'"] = c5_prec_val_best

    run.summary["recall/c1_val'"] = c1_recall_val_best
    run.summary["recall/c2_val'"] = c2_recall_val_best
    run.summary["recall/c3_val'"] = c3_recall_val_best
    run.summary["recall/c4_val'"] = c4_recall_val_best
    run.summary["recall/c5_val'"] = c5_recall_val_best

    run.summary["f1/c1_val'"] = c1_f1_val_best
    run.summary["f1/c2_val'"] = c2_f1_val_best
    run.summary["f1/c3_val'"] = c3_f1_val_best
    run.summary["f1/c4_val'"] = c4_f1_val_best
    run.summary["f1/c5_val'"] = c5_f1_val_best

    for j in range(SAM_AMOUNT):
        run.summary[f"sam_att/{j + 1}_trn'"] = sam_att_best[j]
        run.summary[f"sam_att/{j + 1}_val'"] = sam_att_val_best[j]

        run.summary[f"IoU/{j + 1}_trn'"] = iou_best[j]
        run.summary[f"IoU/{j + 1}_val'"] = iou_val_best[j]

    run.summary["gradcam_miss_trn'"] = gradcam_miss_att_best
    run.summary["gradcam_miss_val'"] = gradcam_miss_att_val_best

    run.summary["gradcam_direct_trn'"] = gradcam_direct_att_best
    run.summary["gradcam_direct_val'"] = gradcam_direct_att_val_best


def print_metrics():
    c1_mAP, c2_mAP, c3_mAP, c4_mAP, c5_mAP, avg_mAP = count_mAP()
    c1_prec, c2_prec, c3_prec, c4_prec, c5_prec, avg_prec = count_precision()
    c1_recall, c2_recall, c3_recall, c4_recall, c5_recall, avg_recall = count_recall()
    c1_f1, c2_f1, c3_f1, c4_f1, c5_f1, avg_f1 = count_f1()
    print(f'mAP {c1_mAP:.3f} {c2_mAP:.3f} {c3_mAP:.3f} {c4_mAP:.3f} {c5_mAP:.3f} ({avg_mAP:.3f})\n'
          f'precision {c1_prec:.3f} {c2_prec:.3f} {c3_prec:.3f} {c4_prec:.3f} {c5_prec:.3f} ({avg_prec:.3f})\n'
          f'recall {c1_recall:.3f} {c2_recall:.3f} {c3_recall:.3f} {c4_recall:.3f} {c5_recall:.3f} ({avg_recall:.3f})\n'
          f'f1 {c1_f1:.3f} {c2_f1:.3f} {c3_f1:.3f} {c4_f1:.3f} {c5_f1:.3f} ({avg_f1:.3f})\n'
          )


def measure_accuracy(output, target):
    th = 0.5
    sigmoid = nn.Sigmoid()
    activated_output = sigmoid(output)
    activated_output = (activated_output > th).float()
    write_expected_predicted(target, activated_output)


def save_checkpoint(state, prefix):
    filename = f'./checkpoints/{prefix}_checkpoint.pth'
    torch.save(state, filename)
    # if is_best:
    #     shutil.copyfile(filename, './checkpoints/%s_model_best.pth.tar' % prefix)


def iou_numpy(labels: np.array, outputs: np.array):
    assert outputs.shape == labels.shape
    SMOOTH = 1e-8
    outputs = outputs.squeeze()
    labels = labels.squeeze()


    intersection = (outputs & labels).sum()
    union = (outputs | labels).sum()

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
