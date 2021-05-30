import argparse
import enum
import math
import os
import random
from collections import OrderedDict
from typing import List
import gc

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms
import wandb
from PIL import ImageFile
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

from MODELS.model_resnet import *
from MODELS.resnet18_bam import ResNet18BAM
from custom_dataset import DatasetISIC2018
from gradcam import GradCAM

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DataType(enum.Enum):
    ISIC256 = "ISIC256"
    ISIC512 = "ISIC512"
    HAM256 = "HAM256"


class RunType(enum.Enum):
    BASELINE = "baseline"
    SAM_1 = "SAM-1"
    SAM_2 = "SAM-2"
    SAM_3 = "SAM-3"
    SAM_ALL = "SAM-all"

    OUTER_SAM_1 = "outer-SAM-1"
    OUTER_SAM_2 = "outer-SAM-2"
    OUTER_SAM_3 = "outer-SAM-3"
    OUTER_SAM_ALL = "outer-SAM-all"

    INV_SAM_1 = "inv-SAM-1"
    INV_SAM_2 = "inv-SAM-2"
    INV_SAM_3 = "inv-SAM-3"
    INV_SAM_ALL = "inv-SAM-all"

    INV_OUTER_SAM_1 = "inv-outer-SAM-1"
    INV_OUTER_SAM_2 = "inv-outer-SAM-2"
    INV_OUTER_SAM_3 = "inv-outer-SAM-3"
    INV_OUTER_SAM_ALL = "inv-outer-SAM-all"


parser = argparse.ArgumentParser(description='PyTorch ResNet+BAM ISIC2018 Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet')
parser.add_argument('--depth', default=50, type=int, metavar='D', help='model depth')
parser.add_argument('--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
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
parser.add_argument('--print-freq', '-p', default=25, type=int,
                    metavar='N', help='print frequency (default: 25)')
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
parser.add_argument('--run-type', type=RunType, default=RunType.BASELINE, help='type of the current run')
parser.add_argument('--lmbd', type=int, default=1, help='coefficient for additional loss')
parser.add_argument('--data-type', type=DataType, default=DataType.ISIC256, help='choice of the dataset')
args = parser.parse_args()
is_server = args.is_server == 1


class MetricsHolder:
    """
    Holds all metrics for the current epoch.
    After logging resets all metrics.
    """

    def __init__(self, objects_amount):
        self.objects_amount = objects_amount

        # Classification metrics
        # +1 for average value on the last slot
        self.f1 = [0.] * (CLASS_AMOUNT + 1)
        self.mAP = [0.] * (CLASS_AMOUNT + 1)
        self.prec = [0.] * (CLASS_AMOUNT + 1)
        self.recall = [0.] * (CLASS_AMOUNT + 1)
        self.accuracy = [0.] * (CLASS_AMOUNT + 1)

        # SAM attention metrcis
        self.sam_miss_rel = [math.inf] * SAM_AMOUNT
        self.sam_direct_rel = [-math.inf] * SAM_AMOUNT
        self.sam_miss = [math.inf] * SAM_AMOUNT
        self.sam_direct = [-math.inf] * SAM_AMOUNT
        self.iou = [-math.inf] * SAM_AMOUNT

        # GradCam attention metric
        self.gc_miss_rel = math.inf
        self.gc_direct_rel = -math.inf
        self.gc_miss = math.inf
        self.gc_direct = -math.inf

        self.loss_add = -1.
        self.loss_main = -1.
        self.loss_comb = -1.

        self.__expected = [[] for _ in range(CLASS_AMOUNT)]
        self.__predicted = [[] for _ in range(CLASS_AMOUNT)]

        self.__gc_miss_rel_sum = 0.
        self.__gc_direct_rel_sum = 0.
        self.__gc_miss_sum = 0.
        self.__gc_direct_sum = 0.

        self.__iou_sum = [0.] * SAM_AMOUNT
        self.__sam_miss_rel_sum = [0.] * SAM_AMOUNT
        self.__sam_direct_rel_sum = [0.] * SAM_AMOUNT
        self.__sam_miss_sum = [0.] * SAM_AMOUNT
        self.__sam_direct_sum = [0.] * SAM_AMOUNT

        self.__loss_add_sum = 0.
        self.__loss_main_sum = 0.
        self.__loss_comb_sum = 0.

    def calculate_all_metrcis(self):
        """
        Set actual value for the every metric
        @return: void
        """
        self.__calculate_gc_metrcis()
        self.__calculate_sam_metrics()
        self.__calculate_classification_metrics()
        self.__calculate_losses()

    def update_losses(self, loss_add, loss_main, loss_comb):
        self.__loss_add_sum += loss_add
        self.__loss_main_sum += loss_main
        self.__loss_comb_sum += loss_comb

    def __calculate_losses(self):
        batch_amount = math.ceil(self.objects_amount / args.batch_size)
        self.loss_add = self.__loss_add_sum / batch_amount
        self.loss_main = self.__loss_main_sum / batch_amount
        self.loss_comb = self.__loss_comb_sum / batch_amount

    def update_expected_predicted(self, target, output):
        # iterate over batch
        assert target.size() == output.size()
        for i in range(target.size(0)):
            cur_target = target[i]
            cur_output = output[i]

            # iterate over classes
            assert cur_target.size(0) == CLASS_AMOUNT
            for j in range(CLASS_AMOUNT):
                self.__expected[j].append(cur_target[j])
                self.__predicted[j].append(cur_output[j])

    def update_sam_metrics(self, iou: List[float], sam_miss_rel: List[float], sam_direct_rel: List[float],
                           sam_miss: List[float], sam_direct: List[float]):
        for i in range(SAM_AMOUNT):
            self.__iou_sum[i] += iou[i]
            self.__sam_miss_rel_sum[i] += sam_miss_rel[i]
            self.__sam_direct_rel_sum[i] += sam_direct_rel[i]
            self.__sam_miss_sum[i] += sam_miss[i]
            self.__sam_direct_sum[i] += sam_direct[i]

    def __calculate_sam_metrics(self):
        for i in range(SAM_AMOUNT):
            self.iou[i] = self.__iou_sum[i] / self.objects_amount
            self.sam_miss_rel[i] = self.__sam_miss_rel_sum[i] / self.objects_amount
            self.sam_direct_rel[i] = self.__sam_direct_rel_sum[i] / self.objects_amount
            self.sam_miss[i] = self.__sam_miss_sum[i] / self.objects_amount
            self.sam_direct[i] = self.__sam_direct_sum[i] / self.objects_amount

    def update_gradcam_metrics(self, gc_miss_rel_sum: float, gc_direct_rel_sum: float, gc_miss_sum: float,
                               gc_direct_sum: float):
        self.__gc_miss_rel_sum += gc_miss_rel_sum
        self.__gc_direct_rel_sum += gc_direct_rel_sum
        self.__gc_miss_sum += gc_miss_sum
        self.__gc_direct_sum += gc_direct_sum

    def __calculate_gc_metrcis(self):
        self.gc_miss_rel = self.__gc_miss_rel_sum / self.objects_amount
        self.gc_direct_rel = self.__gc_direct_rel_sum / self.objects_amount
        self.gc_miss = self.__gc_miss_sum / self.objects_amount
        self.gc_direct = self.__gc_direct_sum / self.objects_amount

    def __calculate_classification_metrics(self):
        # strange things here, but it doesn't work w/o them
        expected = [np.empty([1])] * CLASS_AMOUNT
        predicted = [np.empty([1])] * CLASS_AMOUNT
        for i in range(CLASS_AMOUNT):
            expected[i] = np.asarray(self.__expected[i]).astype(float)
            predicted[i] = np.asarray(self.__predicted[i]).astype(float)
        for i in range(CLASS_AMOUNT):
            self.f1[i] = f1_score(expected[i], predicted[i], average="binary")
            self.mAP[i] = average_precision_score(expected[i], predicted[i])
            self.prec[i] = precision_score(expected[i], predicted[i], average="binary")
            self.recall[i] = recall_score(expected[i], predicted[i], average="binary")
            self.accuracy[i] = accuracy_score(expected[i], predicted[i])

        # reminder: average value is in the last element of the list
        self.f1[-1] = sum([x for i, x in enumerate(self.f1) if i != CLASS_AMOUNT]) / CLASS_AMOUNT
        self.mAP[-1] = sum([x for i, x in enumerate(self.mAP) if i != CLASS_AMOUNT]) / CLASS_AMOUNT
        self.prec[-1] = sum([x for i, x in enumerate(self.prec) if i != CLASS_AMOUNT]) / CLASS_AMOUNT
        self.recall[-1] = sum([x for i, x in enumerate(self.recall) if i != CLASS_AMOUNT]) / CLASS_AMOUNT
        self.accuracy[-1] = sum([x for i, x in enumerate(self.accuracy) if i != CLASS_AMOUNT]) / CLASS_AMOUNT


class BestMetricsHolder(MetricsHolder):
    def __init__(self, objects_amount):
        super().__init__(objects_amount)

    def update(self, mh: MetricsHolder):
        for i in range(CLASS_AMOUNT + 1):
            self.f1[i] = max(self.f1[i], mh.f1[i])
            self.mAP[i] = max(self.mAP[i], mh.mAP[i])
            self.prec[i] = max(self.prec[i], mh.prec[i])
            self.recall[i] = max(self.recall[i], mh.recall[i])
            self.accuracy[i] = max(self.accuracy[i], mh.accuracy[i])

        for i in range(SAM_AMOUNT):
            self.sam_miss_rel[i] = min(self.sam_miss_rel[i], mh.sam_miss_rel[i])
            self.sam_direct_rel[i] = max(self.sam_direct_rel[i], mh.sam_direct_rel[i])
            self.sam_miss[i] = min(self.sam_miss[i], mh.sam_miss[i])
            self.sam_direct[i] = max(self.sam_direct[i], mh.sam_direct[i])
            self.iou[i] = max(self.iou[i], mh.iou[i])

        self.gc_miss_rel = min(self.gc_miss_rel, mh.gc_miss_rel)
        self.gc_direct_rel = max(self.gc_direct_rel, mh.gc_direct_rel)
        self.gc_miss = min(self.gc_miss, mh.gc_miss)
        self.gc_direct = max(self.gc_direct, mh.gc_direct)


# SAM_AMOUNT = 3
SAM_AMOUNT = 1
CLASS_AMOUNT = 7 if args.data_type == DataType.HAM256 else 5
TRAIN_AMOUNT = 1600
# TRAIN_AMOUNT = 6195
VAL_AMOUNT = 400
# VAL_AMOUNT = 1550
TEST_AMOUNT = 594
# TEST_AMOUNT = 2270

best_metrics_val = BestMetricsHolder(VAL_AMOUNT)
best_metrics_test = BestMetricsHolder(TEST_AMOUNT)
best_metrics_train = BestMetricsHolder(TRAIN_AMOUNT)

run = None


def main():
    torch.set_num_threads(2)
    if is_server:
        wandb.login()
    global args, run
    image_size = 256 if args.data_type == DataType.ISIC256 or args.data_type == DataType.HAM256 else 512
    # parse args and set seed
    args = parser.parse_args()
    print("args", args)
    if args.resume is None:
        print("Run w/o checkpoint!")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # choose model
    if args.arch == "resnet34":
        model = models.resnet34(pretrained=True)
        model.fc = nn.Linear(512, CLASS_AMOUNT)
    elif args.arch == "ResNet18BAM":
        model = ResNet18BAM(pretrained=True, sam_instead_bam=True)
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
        model = ResidualNet('ImageNet', args.depth, CLASS_AMOUNT, 'BAM', image_size)
    else:
        model = ResidualNet('ImageNet', args.depth, CLASS_AMOUNT, 'CBAM', image_size)
    pos_weight_train = torch.Tensor(
        [[3.27807486631016, 2.7735849056603774, 12.91304347826087, 0.6859852476290832, 25.229508196721312]])
    if is_server:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_train).cuda(args.cuda_device)
        # criterion = nn.CrossEntropyLoss().cuda(args.cuda_device)
        sam_criterion_outer = nn.BCELoss(reduction='none').cuda(args.cuda_device)
        sam_criterion = nn.BCELoss().cuda(args.cuda_device)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_train)
        # criterion = nn.CrossEntropyLoss()
        sam_criterion_outer = nn.BCELoss(reduction='none')
        sam_criterion = nn.BCELoss()
    if is_server:
        model = model.cuda(args.cuda_device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    # load checkpoint
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            # load ImageNet checkpoints:
            load_foreign_checkpoint(model)

            # load my own checkpoints:
            # start_epoch = load_checkpoint(model)
        else:
            print(f"=> no checkpoint found at '{args.resume}'")
            return -1

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
        run = wandb.init(config=config, project="vol.12", name=args.run_name, tags=args.tags)
    if is_server:
        model = model.cuda(args.cuda_device)
    if is_server:
        wandb.watch(model, criterion, log="all", log_freq=args.print_freq)

    print(f'Number of model parameters: {sum([p.data.nelement() for p in model.parameters()])}')
    cudnn.benchmark = True

    # Data loading code
    if args.data_type == DataType.ISIC256:
        root_dir = 'data/'
        segm_dir = "images/256ISIC2018_Task1_Training_GroundTruth/"
        size0 = 224
    elif args.data_type == DataType.ISIC512:
        root_dir = 'data512/'
        segm_dir = "images/512ISIC2018_Task1_Training_GroundTruth/"
        size0 = 448
    else:
        root_dir = 'ham_data/'
        segm_dir = "images/256HAM_SEGM/"
        size0 = 224

    traindir = os.path.join(root_dir, 'train')
    train_labels = os.path.join(root_dir, 'train', 'images_onehot_train.txt')
    valdir = os.path.join(root_dir, 'val')
    val_labels = os.path.join(root_dir, 'val', 'images_onehot_val.txt')
    # testdir = os.path.join(root_dir, 'test')
    # test_labels = os.path.join(root_dir, 'test', 'images_onehot_test.txt')

    train_dataset = DatasetISIC2018(
        label_file=train_labels,
        root_dir=traindir,
        segm_dir=segm_dir,
        size0=size0,
        perform_flips=True,  # perform flips
        perform_crop=True,  # perform random resized crop with size = 224
        perform_rotate=True,
        perform_jitter=True,
        perform_gaussian_noise=False,
        perform_iaa_augs=True,
        transform=None
    )

    val_dataset = DatasetISIC2018(
        label_file=val_labels,
        root_dir=valdir,
        segm_dir=segm_dir,
        size0=size0,
        perform_flips=False,
        perform_crop=False,
        perform_rotate=False,
        perform_jitter=False,
        perform_gaussian_noise=False,
        perform_iaa_augs=False,
        transform=transforms.CenterCrop(size0)
    )

    # test_dataset = DatasetISIC2018(
    #     label_file=test_labels,
    #     root_dir=testdir,
    #     segm_dir=segm_dir,
    #     size0=size0,
    #     perform_flips=False,
    #     perform_crop=False,
    #     perform_rotate=False,
    #     perform_jitter=False,
    #     perform_gaussian_noise=False,
    #     perform_iaa_augs=False,
    #     transform=transforms.CenterCrop(size0)
    # )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True
    # )
    epoch_number = 0
    for epoch in range(start_epoch, args.epochs):
        # if epoch_number != 0:
        #     checkpoint_dict = {
        #         'epoch': epoch,
        #         'state_dict': model.state_dict(),
        #         'optimizer': optimizer.state_dict()
        #     }
        #     save_checkpoint_to_folder(checkpoint_dict, args.run_name)
        train(train_loader, model, criterion, sam_criterion, sam_criterion_outer, epoch, optimizer)
        validate(val_loader, model, criterion, sam_criterion, sam_criterion_outer, epoch)
        # test(test_loader, model, criterion, sam_criterion, sam_criterion_outer, epoch, optimizer)
        epoch_number += 1
        gc.collect()

    save_summary()
    if run is not None:
        run.finish()


def train(train_loader, model, criterion, sam_criterion, sam_criterion_outer, epoch, optimizer):
    global best_metrics_train
    metrics_holder = MetricsHolder(TRAIN_AMOUNT)
    # switch to train mode
    model.train()

    th = 0.5
    sigmoid = nn.Sigmoid()
    for i, dictionary in enumerate(train_loader):
        input_img = dictionary['image']
        target = dictionary['label']
        segm = dictionary['segm']
        if is_server:
            input_img = input_img.cuda(args.cuda_device)
            target = target.cuda(args.cuda_device)
            segm = segm.cuda(args.cuda_device)
        # get gradcam mask + compute output
        gradcam = GradCAM(model, target_layer=model.layer4)
        gc_mask, no_norm_gc_mask, output, sam_output = gradcam(input_img, retain_graph=True)

        # calculate loss
        loss_comb = calculate_and_update_loss(segm, target, output, sam_output, criterion, sam_criterion,
                                              sam_criterion_outer,
                                              metrics_holder)

        # update classification metrics
        activated_output = (sigmoid(output.data) > th).float()
        metrics_holder.update_expected_predicted(target=target, output=activated_output.detach())

        # calculate and update SAM and gradcam metrics
        metrics_holder.update_gradcam_metrics(*calculate_gradcam_metrics(no_norm_gc_mask.detach(), segm))
        metrics_holder.update_sam_metrics(*calculate_sam_metrics(sam_output, segm))

        optimizer.zero_grad()
        loss_comb.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print(f'Train: [{epoch}][{i}/{len(train_loader)}]')
    metrics_holder.calculate_all_metrcis()
    best_metrics_train.update(metrics_holder)
    wandb_log("trn", epoch, metrics_holder)


def validate(val_loader, model, criterion, sam_criterion, sam_criterion_outer, epoch):
    global best_metrics_val
    metrics_holder = MetricsHolder(VAL_AMOUNT)

    th = 0.5
    sigmoid = nn.Sigmoid()

    # switch to evaluate mode
    model.eval()
    for i, dictionary in enumerate(val_loader):
        input_img = dictionary['image']
        target = dictionary['label']
        segm = dictionary['segm']
        if is_server:
            input_img = input_img.cuda(args.cuda_device)
            target = target.cuda(args.cuda_device)
            segm = segm.cuda(args.cuda_device)

        # get gradcam mask + compute output
        gradcam = GradCAM(model, target_layer=model.layer4)
        gc_mask, no_norm_gc_mask, output, sam_output = gradcam(input_img)

        calculate_and_update_loss(segm, target, output, sam_output, criterion, sam_criterion, sam_criterion_outer,
                                  metrics_holder)

        # update classification metrics
        activated_output = (sigmoid(output.data) > th).float()
        metrics_holder.update_expected_predicted(target=target, output=activated_output.detach())

        # calculate and update SAM and gradcam metrics
        metrics_holder.update_gradcam_metrics(*calculate_gradcam_metrics(no_norm_gc_mask.detach(), segm))
        metrics_holder.update_sam_metrics(*calculate_sam_metrics(sam_output, segm))

        if i % args.print_freq == 0:
            print(f'Validate: [{epoch}][{i}/{len(val_loader)}]')

    metrics_holder.calculate_all_metrcis()
    best_metrics_val.update(metrics_holder)
    wandb_log("val", epoch, metrics_holder)


def test(test_loader, model, criterion, sam_criterion, sam_criterion_outer, epoch):
    global best_metrics_test
    metrics_holder = MetricsHolder(TEST_AMOUNT)

    th = 0.5
    sigmoid = nn.Sigmoid()

    # switch to evaluate mode
    model.eval()
    for i, dictionary in enumerate(test_loader):
        input_img = dictionary['image']
        target = dictionary['label']
        segm = dictionary['segm']
        if is_server:
            input_img = input_img.cuda(args.cuda_device)
            target = target.cuda(args.cuda_device)
            segm = segm.cuda(args.cuda_device)

        # get gradcam mask + compute output
        gradcam = GradCAM(model, target_layer=model.layer4)
        gc_mask, no_norm_gc_mask, output, sam_output = gradcam(input_img)

        # calculate loss and update its metrics
        calculate_and_update_loss(segm, target, output, sam_output, criterion, sam_criterion, sam_criterion_outer,
                                  metrics_holder)

        # update classification metrics
        activated_output = (sigmoid(output.data) > th).float()
        metrics_holder.update_expected_predicted(target=target, output=activated_output.detach())

        # calculate and update SAM and gradcam metrics
        metrics_holder.update_gradcam_metrics(*calculate_gradcam_metrics(no_norm_gc_mask.detach(), segm))
        metrics_holder.update_sam_metrics(*calculate_sam_metrics(sam_output, segm))

        if i % args.print_freq == 0:
            print(f'Test: [{epoch}][{i}/{len(test_loader)}]')

    metrics_holder.calculate_all_metrcis()
    best_metrics_test.update(metrics_holder)
    wandb_log("test", epoch, metrics_holder)


def calculate_and_update_loss(segm, target, output, sam_output, criterion, sam_criterion, sam_criterion_outer,
                              metrics_holder):
    loss_main = criterion(output, target)
    loss_add = calculate_and_choose_additional_loss(segm, sam_output, sam_criterion, sam_criterion_outer)
    loss_comb = loss_main + loss_add
    metrics_holder.update_losses(loss_add=loss_add.detach().item() if args.run_type != RunType.BASELINE else loss_add,
                                 loss_main=loss_main.detach().item(), loss_comb=loss_comb.detach().item())
    return loss_comb


def calculate_gradcam_metrics(no_norm_gc_mask_numpy: torch.Tensor, segm: torch.Tensor):
    """
    Measure gradcam metric

    @param segm - true mask, shape B x 3 x H x W
    @param no_norm_gc_mask_numpy - Grad-CAM output w/o normalization, shape B x 1 x H x W
    @return sum of gradcam attention of every image in the batch (miss and direct)
    """

    # initial segm size = [1, 3, 224, 224]
    maxpool = nn.MaxPool3d(kernel_size=(3, 1, 1))
    true_mask = maxpool(segm)
    true_mask_invert = 1 - true_mask

    true_mask_invert = true_mask_invert.detach().clone().cpu()
    true_mask = true_mask.detach().clone().cpu()
    gradcam_mask = no_norm_gc_mask_numpy.detach().clone().cpu()

    gc_miss_rel_sum = 0.
    gc_direct_rel_sum = 0.
    gc_miss_sum = 0.
    gc_direct_sum = 0.
    # iterate over batch to calculate metrics on each image of the batch
    assert gradcam_mask.size() == true_mask.size() == true_mask_invert.size()
    for i in range(gradcam_mask.size(0)):
        cur_gc = gradcam_mask[i]
        cur_mask = true_mask[i]
        cur_mask_inv = true_mask_invert[i]

        gc_miss_rel_sum += safe_division(torch.sum(cur_gc * cur_mask_inv), torch.sum(cur_gc))
        gc_direct_rel_sum += safe_division(torch.sum(cur_gc * cur_mask), torch.sum(cur_gc))
        gc_miss_sum += safe_division(torch.sum(cur_gc * cur_mask_inv), torch.sum(cur_mask_inv))
        gc_direct_sum += safe_division(torch.sum(cur_gc * cur_mask), torch.sum(cur_mask))
    return gc_miss_rel_sum, gc_direct_rel_sum, gc_miss_sum, gc_direct_sum


def calculate_sam_metrics(sam_output: List[torch.Tensor], segm: torch.Tensor):
    """
    Measure SAM attention metrics and IoU

    @param sam_output[i] - SAM-output #i
    @param segm - true mask, shape B x 3 x H x W
    @return sum for the every metric of every image in the batch
    """

    true_masks, invert_masks = get_processed_masks(segm)

    iou_sum = [0.] * SAM_AMOUNT
    sam_miss_rel_sum = [0.] * SAM_AMOUNT
    sam_direct_rel_sum = [0.] * SAM_AMOUNT
    sam_miss_sum = [0.] * SAM_AMOUNT
    sam_direct_sum = [0.] * SAM_AMOUNT

    # measure SAM attention metrics
    for i in range(SAM_AMOUNT):
        cur_sam_batch = sam_output[i].detach().clone().cpu()
        cur_mask_batch = true_masks[i].detach().clone().cpu()
        cur_mask_inv_batch = invert_masks[i].detach().clone().cpu()

        # iterate over batch to calculate metrics on each image of the batch
        assert cur_sam_batch.size(0) == cur_mask_batch.size(0)
        for j in range(cur_sam_batch.size(0)):
            cur_sam = cur_sam_batch[j]
            # cur_mask = cur_mask_batch[j].expand_as(cur_sam)
            cur_mask = cur_mask_batch[j]
            # cur_mask_inv = cur_mask_inv_batch[j].expand_as(cur_sam)
            cur_mask_inv = cur_mask_inv_batch[j]

            sam_miss_rel_sum[i] += safe_division(torch.sum(cur_sam * cur_mask_inv),
                                                 torch.sum(cur_sam))
            sam_direct_rel_sum[i] += safe_division(torch.sum(cur_sam * cur_mask),
                                                   torch.sum(cur_sam))
            sam_miss_sum[i] += safe_division(torch.sum(cur_sam * cur_mask_inv),
                                             torch.sum(cur_mask_inv))
            sam_direct_sum[i] += safe_division(torch.sum(cur_sam * cur_mask),
                                               torch.sum(cur_mask))
            iou_sum[i] += calculate_iou((cur_sam > 0.5).int(), cur_mask.int())
    return iou_sum, sam_miss_rel_sum, sam_direct_rel_sum, sam_miss_sum, sam_direct_sum


def calculate_additional_loss(segm: torch.Tensor, sam_output: torch.Tensor, sam_criterion, sam_criterion_outer):
    """
    @param segm: true mask, shape B x 3 x H x W
    @param sam_output[i] - SAM-output #i
    @param sam_criterion - nn.BCELoss() (witch or w/o CUDA)
    @param sam_criterion_outer - nn.BCELoss(reduction='none') (witch or w/o CUDA)
    """

    true_masks, invert_masks = get_processed_masks(segm)

    loss_outer_sum = [0 for _ in range(SAM_AMOUNT)]
    loss_inv_sum = [0 for _ in range(SAM_AMOUNT)]

    # assert len(true_masks) == SAM_AMOUNT
    # iterate over SAM number
    for i in range(SAM_AMOUNT):
        # iterate over batch
        for j in range(len(true_masks[i])):
            cur_sam_output = sam_output[i][j]
            # cur_mask = true_masks[i][j].expand_as(cur_sam_output)
            cur_mask = true_masks[i][j]
            # cur_mask_inv = invert_masks[i][j].expand_as(cur_sam_output)
            cur_mask_inv = invert_masks[i][j]

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
        # loss[i] = args.lmbd * sam_criterion(sam_output[i], true_masks[i].expand_as(sam_output[i]))
        loss[i] = args.lmbd * sam_criterion(sam_output[i], true_masks[i])
        # loss_inv[i] = args.lmbd * sam_criterion(sam_output[i], invert_masks[i].expand_as(sam_output[i]))
        loss_inv[i] = args.lmbd * sam_criterion(sam_output[i], invert_masks[i])

        loss_outer[i] = args.lmbd * loss_outer_sum[i] / args.batch_size
        loss_outer_inv[i] = args.lmbd * loss_inv_sum[i] / args.batch_size
    return loss, loss_inv, loss_outer, loss_outer_inv


def get_processed_masks(segm: torch.Tensor):
    """
    @param segm has shape B x 3 x H x W
    @return two lists with true_masks and invert masks for every SAM output
    """
    maxpool_segm1 = nn.MaxPool3d(kernel_size=(3, 4, 4))
    maxpool_segm2 = nn.MaxPool3d(kernel_size=(3, 8, 8))
    maxpool_segm3 = nn.MaxPool3d(kernel_size=(3, 16, 16))
    # maxpool_segm4 = nn.MaxPool3d(kernel_size=(3, 32, 32))

    true_mask1 = maxpool_segm1(segm)
    true_mask2 = maxpool_segm2(segm)
    true_mask3 = maxpool_segm3(segm)
    # true_mask4 = maxpool_segm4(segm)

    true_mask_inv1 = 1 - true_mask1
    true_mask_inv2 = 1 - true_mask2
    true_mask_inv3 = 1 - true_mask3
    # true_mask_inv4 = 1 - true_mask4

    true_masks = [true_mask1]
    # true_masks = [true_mask1, true_mask2, true_mask3]
    invert_masks = [true_mask_inv1]
    # invert_masks = [true_mask_inv1, true_mask_inv2, true_mask_inv3]

    return true_masks, invert_masks


def choose_add_loss(loss: list, loss_inv: list, loss_outer: list, loss_outer_inv: list):
    """
    The choice of additional loss depends on args.run_type.
    Arguments - lists of already calculated losses, len of the list equals SAM_AMOUNT

    @param loss - default SAM-loss
    @param loss_inv: SAM-loss calculated with invert mask
    @param loss_outer: outer SAM-loss
    @param loss_outer_inv: outer SAM-loss calculated with invert mask
    """
    if args.run_type == RunType.BASELINE:
        return 0
    assert len(loss) == len(loss_inv) == len(loss_outer) == len(loss_outer_inv) == SAM_AMOUNT

    loss_add = None
    if args.run_type == RunType.SAM_1:
        loss_add = loss[0]
    elif args.run_type == RunType.INV_SAM_1:
        loss_add = loss_inv[0]
    elif args.run_type == RunType.SAM_2:
        loss_add = loss[1]
    elif args.run_type == RunType.INV_SAM_2:
        loss_add = loss_inv[1]
    elif args.run_type == RunType.SAM_3:
        loss_add = loss[2]
    elif args.run_type == RunType.INV_SAM_3:
        loss_add = loss_inv[2]
    elif args.run_type == RunType.SAM_ALL:
        loss_add = sum(loss)
    elif args.run_type == RunType.INV_SAM_ALL:
        loss_add = sum(loss_inv)
    elif args.run_type == RunType.OUTER_SAM_1:
        loss_add = loss_outer[0]
    elif args.run_type == RunType.INV_OUTER_SAM_1:
        loss_add = loss_outer_inv[0]
    elif args.run_type == RunType.OUTER_SAM_2:
        loss_add = loss_outer[1]
    elif args.run_type == RunType.INV_OUTER_SAM_2:
        loss_add = loss_outer_inv[1]
    elif args.run_type == RunType.OUTER_SAM_3:
        loss_add = loss_outer[2]
    elif args.run_type == RunType.INV_OUTER_SAM_3:
        loss_add = loss_outer_inv[2]
    elif args.run_type == RunType.OUTER_SAM_ALL:
        loss_add = sum(loss_outer)
    elif args.run_type == RunType.INV_OUTER_SAM_ALL:
        loss_add = sum(loss_outer_inv)
    return loss_add


def calculate_and_choose_additional_loss(segm: torch.Tensor, sam_output: torch.Tensor, sam_criterion,
                                         sam_criterion_outer):
    """
    Calculate all add loss and select required one for the current run. The choice depends on the args.run_type.

    @param segm: true mask, shape B x 3 x H x W
    @param sam_output[i] - SAM-output #i
    @param sam_criterion - nn.BCELoss() (witch or w/o CUDA)
    @param sam_criterion_outer - nn.BCELoss(reduction='none') (witch or w/o CUDA)
    """
    return choose_add_loss(*calculate_additional_loss(segm, sam_output, sam_criterion, sam_criterion_outer))


def wandb_log(suffix: str, epoch: int, metrics_holder: MetricsHolder):
    """
    Log all metrcis of the current epoch to the W&B
    @param suffix: trn, val or test
    @param epoch: epoch number to log
    @param metrics_holder: MetricsHolder object which contains ALL ALREADY CALCULATED METRICS
    @return: void
    """
    if not is_server:
        return
    dict_for_log = make_dict_for_log(suffix=suffix, mh=metrics_holder)
    wandb.log(dict_for_log, step=epoch)


def make_dict_for_log(suffix: str, mh: MetricsHolder):
    """
    Takes all metrics via MetrcisHolder object and makes dictionary for wandb_log.
    Or converts MetricsHolder to dict.
    """

    log_dict = {f'loss/comb_{suffix}': mh.loss_comb, f'loss/add_{suffix}': mh.loss_add,
                f'loss/main_{suffix}': mh.loss_main,
                f'gradcam_rel/miss_{suffix}': mh.gc_miss_rel, f'gradcam_rel/direct_{suffix}': mh.gc_direct_rel,
                f'gradcam/miss_{suffix}': mh.gc_miss, f'gradcam/direct_{suffix}': mh.gc_direct}

    assert len(mh.f1) == len(mh.recall) == len(mh.prec) == len(mh.mAP) == CLASS_AMOUNT + 1
    for i in range(CLASS_AMOUNT):
        log_dict[f'f1/c{i + 1}_{suffix}'] = mh.f1[i]
        log_dict[f'mAP/c{i + 1}_{suffix}'] = mh.mAP[i]
        log_dict[f'prec/c{i + 1}_{suffix}'] = mh.prec[i]
        log_dict[f'recall/c{i + 1}_{suffix}'] = mh.recall[i]
        log_dict[f'accuracy/c{i + 1}_{suffix}'] = mh.accuracy[i]
    log_dict[f'f1/avg_{suffix}'] = mh.f1[-1]
    log_dict[f'mAP/avg_{suffix}'] = mh.mAP[-1]
    log_dict[f'prec/avg_{suffix}'] = mh.prec[-1]
    log_dict[f'recall/avg_{suffix}'] = mh.recall[-1]
    log_dict[f'accuracy/avg_{suffix}'] = mh.accuracy[-1]

    assert len(mh.iou) == len(mh.sam_miss_rel) == len(mh.sam_direct_rel) == SAM_AMOUNT == len(mh.sam_miss) == \
           len(mh.sam_direct)
    for i in range(SAM_AMOUNT):
        log_dict[f'IoU/{i + 1}_{suffix}'] = mh.iou[i]
        log_dict[f'sam_miss_rel/{i + 1}_{suffix}'] = mh.sam_miss_rel[i]
        log_dict[f'sam_direct_rel/{i + 1}_{suffix}'] = mh.sam_direct_rel[i]
        log_dict[f'sam_miss/{i + 1}_{suffix}'] = mh.sam_miss[i]
        log_dict[f'sam_direct/{i + 1}_{suffix}'] = mh.sam_direct[i]
    return log_dict


# noinspection PyUnresolvedReferences
def save_summary():
    if not is_server:
        return
    print("saving summary..")

    global run
    global best_metrics_train, best_metrics_val, best_metrics_test
    summary_dict_train = make_dict_for_log("trn'", best_metrics_train)
    summary_dict_val = make_dict_for_log("val'", best_metrics_val)
    summary_dict_test = make_dict_for_log("test'", best_metrics_test)
    run.summary.update(summary_dict_train)
    run.summary.update(summary_dict_val)
    run.summary.update(summary_dict_test)


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


def save_checkpoint_to_folder(state, run_name):
    if not os.path.exists(f'./checkpoints/my_checkpoints/'):
        os.mkdir(f'./checkpoints/my_checkpoints/')
    filename = f'./checkpoints/my_checkpoints/{run_name}.pth'
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


def load_foreign_checkpoint(model):
    print(f"=> loading checkpoint '{args.resume}'")

    # create dummy layer to init weights in the state_dict
    dummy_fc = torch.nn.Linear(512 * 4, CLASS_AMOUNT)
    torch.nn.init.xavier_uniform_(dummy_fc.weight)

    if is_server:
        checkpoint = torch.load(args.resume, map_location=f'cuda:{args.cuda_device}')
    else:
        checkpoint = torch.load(args.resume, map_location='cpu')
    state_dict = checkpoint['state_dict']

    state_dict['module.fc.weight'] = dummy_fc.weight
    state_dict['module.fc.bias'] = dummy_fc.bias

    # remove `module.` prefix because we don't use torch.nn.DataParallel
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print(f"=> loaded checkpoint '{args.resume}'")


def load_checkpoint(model, optimizer):
    print(f"=> loading checkpoint '{args.resume}'")
    if is_server:
        checkpoint = torch.load(args.resume, map_location=f'cuda:{args.cuda_device}')
    else:
        checkpoint = torch.load(args.resume, map_location='cpu')
    state_dict = checkpoint['state_dict']
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("optimizer state loaded")
    model.load_state_dict(state_dict)
    print(f"=> loaded checkpoint '{args.resume}'")
    print(f"epoch = {checkpoint['epoch']}")
    return checkpoint['epoch']


if __name__ == '__main__':
    main()
