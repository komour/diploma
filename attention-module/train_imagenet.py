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

from custom_dataset import DatasetISIC2018

ImageFile.LOAD_TRUNCATED_IMAGES = True
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--depth', default=50, type=int, metavar='D',
                    help='model depth')
parser.add_argument('--ngpu', default=4, type=int, metavar='G',
                    help='number of gpus to use')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
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
parser.add_argument("--seed", type=int, default=1234, metavar='BS', help='input batch size for training (default: 64)')
parser.add_argument("--prefix", type=str, required=True, metavar='PFX', help='prefix for logging & checkpoint saving')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluation only')
parser.add_argument('--att-type', type=str, choices=['BAM', 'CBAM'], default=None)
best_prec1 = 0

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

avg_f1_best = 0
avg_precision_best = 0
avg_recall_best = 0
avg_mAP_best = 0


def main():
    global args, avg_f1_best, avg_precision_best, avg_recall_best, avg_mAP_best
    global viz, train_lot, test_lot
    global c1_expected, c1_predicted, c2_expected, c2_predicted, c3_expected, c3_predicted, c4_expected, c4_predicted
    global c5_expected, c5_predicted, avg_f1_best, avg_precision_best, avg_recall_best, avg_mAP_best
    args = parser.parse_args()
    print("args", args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # create model
    if args.arch == "resnet":
        model = ResidualNet('ImageNet', args.depth, 5, args.att_type)
    else:
        print('arch `resnet` expected, "', args.arch, '"found')
        return

    # define loss function (criterion) and optimizer

    criterion = nn.BCEWithLogitsLoss()

    # optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))  # cuda_here

    model = model
    # print("model")
    # print(model)
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    # optionally resume from a checkpoint
    # if args.resume: TODO
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         args.start_epoch = checkpoint['epoch']
    #         best_prec1 = checkpoint['best_prec1']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         if 'optimizer' in checkpoint:
    #             optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(args.resume, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    train_labels = os.path.join(args.data, 'train', 'images_onehot_train.txt')
    valdir = os.path.join(args.data, 'val')
    val_labels = os.path.join(args.data, 'val', 'images_onehot_val.txt')
    # testdir = os.path.join(args.data, 'test') TODO
    # test_labels = os.path.join(args.data, 'test', 'images_onehot_test.txt')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # import pdb
    # pdb.set_trace()
    val_dataset = DatasetISIC2018(
        val_labels,
        valdir,
        False,
        transforms.Compose([
            # transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    size0 = 224
    train_dataset = DatasetISIC2018(
        train_labels,
        traindir,
        True,  # perform flips
        transforms.Compose([
            transforms.RandomResizedCrop(size0),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    # test_dataset = DatasetISIC2018( TODO
    #     test_labels,
    #     testdir,
    #     True,
    #     transforms.Compose([
    #         # transforms.RandomResizedCrop(size0),
    #         # transforms.RandomHorizontalFlip(),
    #         # transforms.RandomVerticalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    # )
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    for epoch in range(args.start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        clear_expected_predicted()
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        clear_expected_predicted()
        validate(val_loader, model, criterion)

        # save checkpoint
        c1_mAP, c2_mAP, c3_mAP, c4_mAP, c5_mAP, avg_mAP = count_mAP()
        c1_precision, c2_precision, c3_precision, c4_precision, c5_precision, avg_precision = count_precision()
        c1_recall, c2_recall, c3_recall, c4_recall, c5_recall, avg_recall = count_recall()
        c1_f1, c2_f1, c3_f1, c4_f1, c5_f1, avg_f1 = count_f1()
        is_best = avg_f1 > avg_f1_best
        if is_best:
            avg_f1_best = avg_f1
            avg_mAP_best = avg_mAP
            avg_recall_best = avg_recall
            avg_precision_best = avg_precision
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'arch': args.arch,
        #     'state_dict': model.state_dict(),
        #     'c1_mAP': c1_mAP,
        #     'c2_mAP': c2_mAP,
        #     'c3_mAP': c3_mAP,
        #     'c4_mAP': c4_mAP,
        #     'c5_mAP': c5_mAP,
        #     'avg_mAP': avg_mAP,
        #     'c1_precision': c1_precision,
        #     'c2_precision': c2_precision,
        #     'c3_precision': c3_precision,
        #     'c4_precision': c4_precision,
        #     'c5_precision': c5_precision,
        #     'avg_precision': avg_precision,
        #     'c1_recall': c1_recall,
        #     'c2_recall': c2_recall,
        #     'c3_recall': c3_recall,
        #     'c4_recall': c4_recall,
        #     'c5_recall': c5_recall,
        #     'avg_recall': avg_recall,
        #     'c1_f1': c1_f1,
        #     'c2_f1': c2_f1,
        #     'c3_f1': c3_f1,
        #     'c4_f1': c4_f1,
        #     'c5_f1': c5_f1,
        #     'avg_f1': avg_f1,
        #     'best_f1': avg_f1_best,
        #     'optimizer': optimizer.state_dict(),
        # }, is_best, args.prefix)


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
        target = target
        segm = dictionary['segm']
        # print("target = ", target)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output, cbam_output = model(input_img)
        # output, cbam = model(input_var)  # TODO
        # print("output = ", output)
        # output.data[0] = torch.Tensor([0.5, 0.5, 0.5, 0.5, 0.5])
        loss1 = criterion(output, target)
        print(segm.size())
        print(cbam_output[0].size(), cbam_output[1].size(), cbam_output[2].size(), cbam_output[3].size())
        # loss2 = criterion(cbam_output[-1], segm)
        # loss_comb = loss1 + loss2

        # measure accuracy and record loss
        measure_accuracy(output.data, target)

        # losses.update(loss.data[0], input.size(0))
        losses.update(loss1.item(), input_img.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('\nEpoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            if i > 0:
                print_metrics()


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, dictionary in enumerate(val_loader):
        input_img = dictionary['image']
        target = dictionary['label']
        target = target
        # segm = dictionary['segm'] TODO

        # compute output
        with torch.no_grad():
            output, _ = model(input_img)
            loss = criterion(output, target)

        # measure accuracy and record loss
        measure_accuracy(output.data, target)
        # losses.update(loss.data[0], input.size(0))
        losses.update(loss.item(), input_img.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Validate: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(i, len(val_loader), batch_time=batch_time,
                                                                loss=losses))
            if i != 0:
                print_metrics()
    print_metrics()


def save_checkpoint(state, is_best, prefix):
    filename = './checkpoints/%s_checkpoint.pth.tar' % prefix
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './checkpoints/%s_model_best.pth.tar' % prefix)


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

    c1_mAP = average_precision_score(c1_exp, c1_pred, average='binary')
    c2_mAP = average_precision_score(c2_exp, c2_pred, average='binary')
    c3_mAP = average_precision_score(c3_exp, c3_pred, average='binary')
    c4_mAP = average_precision_score(c4_exp, c4_pred, average='binary')
    c5_mAP = average_precision_score(c5_exp, c5_pred, average='binary')
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

    c1_precision = precision_score(c1_exp, c1_pred, average='binary')
    c2_precision = precision_score(c2_exp, c2_pred, average='binary')
    c3_precision = precision_score(c3_exp, c3_pred, average='binary')
    c4_precision = precision_score(c4_exp, c4_pred, average='binary')
    c5_precision = precision_score(c5_exp, c5_pred, average='binary')
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

    c1_recall = recall_score(c1_exp, c1_pred, average='binary')
    c2_recall = recall_score(c2_exp, c2_pred, average='binary')
    c3_recall = recall_score(c3_exp, c3_pred, average='binary')
    c4_recall = recall_score(c4_exp, c4_pred, average='binary')
    c5_recall = recall_score(c5_exp, c5_pred, average='binary')
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

    c1_f1 = f1_score(c1_exp, c1_pred, average='binary')
    c2_f1 = f1_score(c2_exp, c2_pred, average='binary')
    c3_f1 = f1_score(c3_exp, c3_pred, average='binary')
    c4_f1 = f1_score(c4_exp, c4_pred, average='binary')
    c5_f1 = f1_score(c5_exp, c5_pred, average='binary')
    avg_f1 = (c1_f1 + c2_f1 + c3_f1 + c4_f1 + c5_f1) / 5
    return c1_f1, c2_f1, c3_f1, c4_f1, c5_f1, avg_f1


def print_metrics():
    c1_mAP, c2_mAP, c3_mAP, c4_mAP, c5_mAP, avg_mAP = count_mAP()
    c1_precision, c2_precision, c3_precision, c4_precision, c5_precision, avg_precision = count_precision()
    c1_recall, c2_recall, c3_recall, c4_recall, c5_recall, avg_recall = count_recall()
    c1_f1, c2_f1, c3_f1, c4_f1, c5_f1, avg_f1 = count_f1()
    print('mAP {c1_mAP:.3f} {c2_mAP:.3f} {c3_mAP:.3f} {c4_mAP:.3f} {c5_mAP:.3f} ({avg_mAP:.3f})\n'
          'precision {c1_precision:.3f} {c2_precision:.3f} {c3_precision:.3f} {c4_precision:.3f} {c5_precision:.3f} ({avg_precision:.3f})\n'
          'recall {c1_recall:.3f} {c2_recall:.3f} {c3_recall:.3f} {c4_recall:.3f} {c5_recall:.3f} ({avg_recall:.3f})\n'
          'f1 {c1_f1:.3f} {c2_f1:.3f} {c3_f1:.3f} {c4_f1:.3f} {c5_f1:.3f} ({avg_f1:.3f})\n'.format(
        c1_mAP=c1_mAP, c2_mAP=c2_mAP, c3_mAP=c3_mAP, c4_mAP=c4_mAP, c5_mAP=c5_mAP, avg_mAP=avg_mAP,
        c1_precision=c1_precision, c2_precision=c2_precision, c3_precision=c3_precision, c4_precision=c4_precision,
        c5_precision=c5_precision, avg_precision=avg_precision,
        c1_recall=c1_recall, c2_recall=c2_recall, c3_recall=c3_recall, c4_recall=c4_recall, c5_recall=c5_recall,
        avg_recall=avg_recall,
        c1_f1=c1_f1, c2_f1=c2_f1, c3_f1=c3_f1, c4_f1=c4_f1, c5_f1=c5_f1, avg_f1=avg_f1)
    )


def measure_accuracy(output, target):
    th = 0.5
    sigmoid = nn.Sigmoid()
    activated_output = sigmoid(output)
    activated_output = (activated_output > th).float()
    write_expected_predicted(target, activated_output)
    print('output =', activated_output, ', target =', target, ' , f1 =', avg_f1_best)


if __name__ == '__main__':
    main()
