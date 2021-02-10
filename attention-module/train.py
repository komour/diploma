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

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt

from custom_dataset import DatasetISIC2018

import wandb

ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='PyTorch ResNet+CBAM ISIC2018 Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet')
parser.add_argument('--depth', default=50, type=int, metavar='D', help='model depth')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', type=int, default=1234, metavar='BS')
parser.add_argument('--prefix', type=str, default='', metavar='PFX',
                    help='prefix for logging & checkpoint saving')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluation only')
parser.add_argument('--cuda-device', type=int, default=0)
parser.add_argument('--run-name', type=str, default='noname run', help='run name on the W&B service')
parser.add_argument('--is-server', type=int, choices=[0, 1], default=1)
parser.add_argument("--tags", nargs='+', default=['default-tag'])

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

avg_f1_best = 0
avg_f1_test_best = 0
avg_mAP_best = 0
avg_mAP_test_best = 0
avg_precision_best = 0
avg_precision_test_best = 0
avg_recall_best = 0
avg_recall_test_best = 0

c1_f1_best = 0
c1_f1_test_best = 0
c2_f1_best = 0
c2_f1_test_best = 0
c3_f1_best = 0
c3_f1_test_best = 0
c4_f1_best = 0
c4_f1_test_best = 0
c5_f1_best = 0
c5_f1_test_best = 0

c1_mAP_best = 0
c1_mAP_test_best = 0
c2_mAP_best = 0
c2_mAP_test_best = 0
c3_mAP_best = 0
c3_mAP_test_best = 0
c4_mAP_best = 0
c4_mAP_test_best = 0
c5_mAP_best = 0
c5_mAP_test_best = 0

c1_precision_best = 0
c1_precision_test_best = 0
c2_precision_best = 0
c2_precision_test_best = 0
c3_precision_best = 0
c3_precision_test_best = 0
c4_precision_best = 0
c4_precision_test_best = 0
c5_precision_best = 0
c5_precision_test_best = 0

c1_recall_best = 0
c1_recall_test_best = 0
c2_recall_best = 0
c2_recall_test_best = 0
c3_recall_best = 0
c3_recall_test_best = 0
c4_recall_best = 0
c4_recall_test_best = 0
c5_recall_best = 0
c5_recall_test_best = 0


def main():
    if is_server:
        wandb.login()
    global args
    args = parser.parse_args()
    print("args", args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    # create model
    CLASS_AMOUNT = 5
    if args.arch == "resnet":
        model = ResidualNet('ImageNet', args.depth, CLASS_AMOUNT, 'CBAM')
    else:
        print('arch `resnet` expected, "', args.arch, '"found')
        return

    # define loss function (criterion) and optimizer
    if is_server:
        criterion = nn.BCEWithLogitsLoss().cuda(args.cuda_device)
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    config = dict(
        architecture=f"{args.arch}{args.depth}",
        learning_rate=args.lr,
        epochs=args.epochs,
        start_epoch=args.start_epoch,
        seed=args.seed,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        momentum=args.momentum,
        workers=args.workers,
        prefix=args.prefix,
        resume=args.resume,
        evaluate=args.evaluate
    )
    if is_server:
        wandb.init(config=config, project="vol.3", name=args.run_name, tags=args.tags)

    if is_server:
        model = model.cuda(args.cuda_device)
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
        validate(val_loader, model, criterion, 0)
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

    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        clear_expected_predicted()
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        clear_expected_predicted()
        validate(val_loader, model, criterion, epoch)
    save_summary()


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    for i, dictionary in enumerate(train_loader):
        input_img = dictionary['image']
        target = dictionary['label']
        segm = dictionary['segm']
        if is_server:
            input_img = input_img.cuda(args.cuda_device)
            target = target.cuda(args.cuda_device)
            segm = segm.cuda(args.cuda_device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output, sam_output = model(input_img)
        # sam_output shapes:
        # [1, 1, 56, 56]x3 , [1, 1, 28, 28]x4 [1, 1, 14, 14]x6 , [1, 1, 7, 7]x3

        # visualize mask/input_image/sam_output
        # np_sam = torch.squeeze(sam_output[0]).detach().numpy()
        # np_segm = np.moveaxis(torch.squeeze(segm).detach().numpy(), 0, -1)
        # np_input = np.moveaxis(torch.squeeze(input_img).detach().numpy(), 0, -1)
        # plt.imshow(np_segm.astype(np.uint8))
        # plt.imshow(np_sam)
        # plt.show()

        # initial segm size = [1, 3, 224, 224]
        # maxpool_segm1 = nn.MaxPool3d(kernel_size=(3, 4, 4))
        # maxpool_segm2 = nn.MaxPool3d(kernel_size=(3, 8, 8))
        # maxpool_segm3 = nn.MaxPool3d(kernel_size=(3, 16, 16))
        maxpool_segm4 = nn.MaxPool3d(kernel_size=(3, 32, 32))

        # processed_segm1 = maxpool_segm1(segm)
        # processed_segm2 = maxpool_segm2(segm)
        # processed_segm3 = maxpool_segm3(segm)
        processed_segm4 = maxpool_segm4(segm)

        loss0 = criterion(output, target)

        # loss1 = criterion(sam_output[0], processed_segm1)
        # loss2 = criterion(sam_output[1], processed_segm1)
        # loss3 = criterion(sam_output[2], processed_segm1)
        # loss4 = criterion(sam_output[3], processed_segm2)
        # loss5 = criterion(sam_output[4], processed_segm2)
        # loss6 = criterion(sam_output[5], processed_segm2)
        # loss7 = criterion(sam_output[6], processed_segm2)
        # loss8 = criterion(sam_output[7], processed_segm3)
        # loss9 = criterion(sam_output[8], processed_segm3)
        # loss10 = criterion(sam_output[9], processed_segm3)
        # loss11 = criterion(sam_output[10], processed_segm3)
        # loss12 = criterion(sam_output[11], processed_segm3)
        # loss13 = criterion(sam_output[12], processed_segm3)
        # loss14 = criterion(sam_output[13], processed_segm4)
        # loss15 = criterion(sam_output[14], processed_segm4)
        loss16 = criterion(sam_output[15], processed_segm4)
        #
        # loss_comb = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14 + loss15 + loss16
        loss_comb = loss0 + loss16
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

        # step -- number of examples seen
        if i % args.print_freq == 0:
            print(f'\nEpoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})')
            if i > 0:
                print_metrics()
    wandb_log_train(epoch, losses.avg)


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, dictionary in enumerate(val_loader):
        input_img = dictionary['image']
        target = dictionary['label']
        if is_server:
            input_img = input_img.cuda(args.cuda_device)
            target = target.cuda(args.cuda_device)

        # compute output
        with torch.no_grad():
            output, sam_output = model(input_img)
            loss = criterion(output, target)

        # measure accuracy and record loss
        measure_accuracy(output.data, target)
        # losses.update(loss.data[0], input.size(0))
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
    wandb_log_test(epoch, losses.avg)


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
    t = target[0]
    c1_expected.append(t[0])
    c2_expected.append(t[1])
    c3_expected.append(t[2])
    c4_expected.append(t[3])
    c5_expected.append(t[4])

    o = output[0]
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

    c1_precision = precision_score(c1_exp, c1_pred, average="binary")
    c2_precision = precision_score(c2_exp, c2_pred, average="binary")
    c3_precision = precision_score(c3_exp, c3_pred, average="binary")
    c4_precision = precision_score(c4_exp, c4_pred, average="binary")
    c5_precision = precision_score(c5_exp, c5_pred, average="binary")
    avg_precision = (c1_precision + c2_precision + c3_precision + c4_precision + c5_precision) / 5
    return c1_precision, c2_precision, c3_precision, c4_precision, c5_precision, avg_precision


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
    global c1_precision_best, c2_precision_best, c3_precision_best, c4_precision_best, c5_precision_best
    global avg_f1_best, avg_mAP_best, avg_recall_best, avg_precision_best

    c1_mAP, c2_mAP, c3_mAP, c4_mAP, c5_mAP, avg_mAP = count_mAP()
    c1_precision, c2_precision, c3_precision, c4_precision, c5_precision, avg_precision = count_precision()
    c1_recall, c2_recall, c3_recall, c4_recall, c5_recall, avg_recall = count_recall()
    c1_f1, c2_f1, c3_f1, c4_f1, c5_f1, avg_f1 = count_f1()

    if avg_f1 > avg_f1_best:
        avg_f1_best = avg_f1
    if avg_mAP > avg_mAP_best:
        avg_mAP_best = avg_mAP
    if avg_recall > avg_recall_best:
        avg_recall_best = avg_recall
    if avg_precision > avg_precision_best:
        avg_precision_best = avg_precision

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

    if c1_precision > c1_precision_best:
        c1_precision_best = c1_precision
    if c2_precision > c2_precision_best:
        c2_precision_best = c2_precision
    if c3_precision > c3_precision_best:
        c3_precision_best = c3_precision
    if c4_precision > c4_precision_best:
        c4_precision_best = c4_precision
    if c5_precision > c5_precision_best:
        c5_precision_best = c5_precision

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

    wandb.log({"loss_avg": loss_avg,
               "c1_mAP": c1_mAP, "c2_mAP": c2_mAP, "c3_mAP": c3_mAP, "c4_mAP": c4_mAP, "c5_mAP": c5_mAP,
               "avg_mAP": avg_mAP,
               "c1_precision": c1_precision, "c2_precision": c2_precision, "c3_precision": c3_precision,
               "c4_precision": c4_precision, "c5_precision": c5_precision, "avg_precision": avg_precision,
               "c1_recall": c1_recall, "c2_recall": c2_recall, "c3_recall": c3_recall, "c4_recall": c4_recall,
               "c5_recall": c5_recall, "avg_recall": avg_recall,
               "c1_f1": c1_f1, "c2_f1": c2_f1, "c3_f1": c3_f1, "c4_f1": c4_f1, "c5_f1": c5_f1, "avg_f1": avg_f1
               },
              step=epoch)


def wandb_log_test(epoch, loss_avg):
    if not is_server:
        return

    global c1_f1_test_best, c2_f1_test_best, c3_f1_test_best, c4_f1_test_best, c5_f1_test_best
    global c1_mAP_test_best, c2_mAP_test_best, c3_mAP_test_best, c4_mAP_test_best, c5_mAP_test_best
    global c1_recall_test_best, c2_recall_test_best, c3_recall_test_best, c4_recall_test_best, c5_recall_test_best
    global c1_precision_test_best, c2_precision_test_best, c3_precision_test_best, c4_precision_test_best, c5_precision_test_best
    global avg_f1_test_best, avg_mAP_test_best, avg_recall_test_best, avg_precision_test_best

    c1_mAP, c2_mAP, c3_mAP, c4_mAP, c5_mAP, avg_mAP = count_mAP()
    c1_precision, c2_precision, c3_precision, c4_precision, c5_precision, avg_precision = count_precision()
    c1_recall, c2_recall, c3_recall, c4_recall, c5_recall, avg_recall = count_recall()
    c1_f1, c2_f1, c3_f1, c4_f1, c5_f1, avg_f1 = count_f1()

    if avg_f1 > avg_f1_test_best:
        avg_f1_test_best = avg_f1
    if avg_mAP > avg_mAP_test_best:
        avg_mAP_test_best = avg_mAP
    if avg_recall > avg_recall_test_best:
        avg_recall_test_best = avg_recall
    if avg_precision > avg_precision_test_best:
        avg_precision_test_best = avg_precision

    if c1_mAP > c1_mAP_test_best:
        c1_mAP_test_best = c1_mAP
    if c2_mAP > c2_mAP_test_best:
        c2_mAP_test_best = c2_mAP
    if c3_mAP > c3_mAP_test_best:
        c3_mAP_test_best = c3_mAP
    if c4_mAP > c4_mAP_test_best:
        c4_mAP_test_best = c4_mAP
    if c5_mAP > c5_mAP_test_best:
        c5_mAP_test_best = c5_mAP

    if c1_precision > c1_precision_test_best:
        c1_precision_test_best = c1_precision
    if c2_precision > c2_precision_test_best:
        c2_precision_test_best = c2_precision
    if c3_precision > c3_precision_test_best:
        c3_precision_test_best = c3_precision
    if c4_precision > c4_precision_test_best:
        c4_precision_test_best = c4_precision
    if c5_precision > c5_precision_test_best:
        c5_precision_test_best = c5_precision

    if c1_recall > c1_recall_test_best:
        c1_recall_test_best = c1_recall
    if c2_recall > c2_recall_test_best:
        c2_recall_test_best = c2_recall
    if c3_recall > c3_recall_test_best:
        c3_recall_test_best = c3_recall
    if c4_recall > c4_recall_test_best:
        c4_recall_test_best = c4_recall
    if c5_recall > c5_recall_test_best:
        c5_recall_test_best = c5_recall

    if c1_f1 > c1_f1_test_best:
        c1_f1_test_best = c1_f1
    if c2_f1 > c2_f1_test_best:
        c2_f1_test_best = c2_f1
    if c3_f1 > c3_f1_test_best:
        c3_f1_test_best = c3_f1
    if c4_f1 > c4_f1_test_best:
        c4_f1_test_best = c4_f1
    if c5_f1 > c5_f1_test_best:
        c5_f1_test_best = c5_f1

    wandb.log({"loss_avg_test": loss_avg,
               "c1_mAP_test": c1_mAP, "c2_mAP_test": c2_mAP, "c3_mAP_test": c3_mAP, "c4_mAP_test": c4_mAP,
               "c5_mAP_test": c5_mAP,
               "avg_mAP_test": avg_mAP,
               "c1_precision_test": c1_precision, "c2_precision_test": c2_precision, "c3_precision_test": c3_precision,
               "c4_precision_test": c4_precision, "c5_precision_test": c5_precision,
               "avg_precision_test": avg_precision,
               "c1_recall_test": c1_recall, "c2_recall_test": c2_recall, "c3_recall_test": c3_recall,
               "c4_recall_test": c4_recall,
               "c5_recall_test": c5_recall, "avg_recall_test": avg_recall,
               "c1_f1_test": c1_f1, "c2_f1_test": c2_f1, "c3_f1_test": c3_f1, "c4_f1_test": c4_f1, "c5_f1_test": c5_f1,
               "avg_f1_test": avg_f1
               },
              step=epoch)


def save_summary():
    print("saving summary..")
    # train
    wandb.run.summary["avg_f1'"] = avg_f1_best
    wandb.run.summary["avg_mAP'"] = avg_mAP_best
    wandb.run.summary["avg_recall'"] = avg_recall_best
    wandb.run.summary["avg_precision'"] = avg_precision_best

    wandb.run.summary["mAP_с1"] = c1_mAP_best
    wandb.run.summary["mAP_с2"] = c2_mAP_best
    wandb.run.summary["mAP_с3"] = c3_mAP_best
    wandb.run.summary["mAP_с4"] = c4_mAP_best
    wandb.run.summary["mAP_с5"] = c5_mAP_best

    wandb.run.summary["precision_c1"] = c1_precision_best
    wandb.run.summary["precision_c2"] = c2_precision_best
    wandb.run.summary["precision_c3"] = c3_precision_best
    wandb.run.summary["precision_c4"] = c4_precision_best
    wandb.run.summary["precision_c5"] = c5_precision_best

    wandb.run.summary["recall_c1"] = c1_recall_best
    wandb.run.summary["recall_c2"] = c2_recall_best
    wandb.run.summary["recall_c3"] = c3_recall_best
    wandb.run.summary["recall_c4"] = c4_recall_best
    wandb.run.summary["recall_c5"] = c5_recall_best

    wandb.run.summary["f1_c1"] = c1_f1_best
    wandb.run.summary["f1_c2"] = c2_f1_best
    wandb.run.summary["f1_c3"] = c3_f1_best
    wandb.run.summary["f1_c4"] = c4_f1_best
    wandb.run.summary["f1_c5"] = c5_f1_best

    # test
    wandb.run.summary["avg_f1_test'"] = avg_f1_test_best
    wandb.run.summary["avg_mAP_test'"] = avg_mAP_test_best
    wandb.run.summary["avg_recall_test'"] = avg_recall_test_best
    wandb.run.summary["avg_precision_test'"] = avg_precision_test_best

    wandb.run.summary["mAP_test_c1"] = c1_mAP_test_best
    wandb.run.summary["mAP_test_c2"] = c2_mAP_test_best
    wandb.run.summary["mAP_test_c3"] = c3_mAP_test_best
    wandb.run.summary["mAP_test_c4"] = c4_mAP_test_best
    wandb.run.summary["mAP_test_c5"] = c5_mAP_test_best

    wandb.run.summary["precision_test_c1"] = c1_precision_test_best
    wandb.run.summary["precision_test_c2"] = c2_precision_test_best
    wandb.run.summary["precision_test_c3"] = c3_precision_test_best
    wandb.run.summary["precision_test_c4"] = c4_precision_test_best
    wandb.run.summary["precision_test_c5"] = c5_precision_test_best

    wandb.run.summary["recall_test_c1"] = c1_recall_test_best
    wandb.run.summary["recall_test_c2"] = c2_recall_test_best
    wandb.run.summary["recall_test_c3"] = c3_recall_test_best
    wandb.run.summary["recall_test_c4"] = c4_recall_test_best
    wandb.run.summary["recall_test_c5"] = c5_recall_test_best

    wandb.run.summary["f1_test_c1"] = c1_f1_test_best
    wandb.run.summary["f1_test_c2"] = c2_f1_test_best
    wandb.run.summary["f1_test_c3"] = c3_f1_test_best
    wandb.run.summary["f1_test_c4"] = c4_f1_test_best
    wandb.run.summary["f1_test_c5"] = c5_f1_test_best


def print_metrics():
    c1_mAP, c2_mAP, c3_mAP, c4_mAP, c5_mAP, avg_mAP = count_mAP()
    c1_precision, c2_precision, c3_precision, c4_precision, c5_precision, avg_precision = count_precision()
    c1_recall, c2_recall, c3_recall, c4_recall, c5_recall, avg_recall = count_recall()
    c1_f1, c2_f1, c3_f1, c4_f1, c5_f1, avg_f1 = count_f1()
    print(f'mAP {c1_mAP:.3f} {c2_mAP:.3f} {c3_mAP:.3f} {c4_mAP:.3f} {c5_mAP:.3f} ({avg_mAP:.3f})\n'
          f'precision {c1_precision:.3f} {c2_precision:.3f} {c3_precision:.3f} {c4_precision:.3f} {c5_precision:.3f} ({avg_precision:.3f})\n'
          f'recall {c1_recall:.3f} {c2_recall:.3f} {c3_recall:.3f} {c4_recall:.3f} {c5_recall:.3f} ({avg_recall:.3f})\n'
          f'f1 {c1_f1:.3f} {c2_f1:.3f} {c3_f1:.3f} {c4_f1:.3f} {c5_f1:.3f} ({avg_f1:.3f})\n'
          )


def measure_accuracy(output, target):
    th = 0.5
    sigmoid = nn.Sigmoid()
    activated_output = sigmoid(output)
    activated_output = (activated_output > th).float()
    write_expected_predicted(target, activated_output)


if __name__ == '__main__':
    main()
