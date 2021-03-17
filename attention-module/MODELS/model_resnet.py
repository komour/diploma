from torch.nn import init
from .cbam import *
from .bam import *


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False, first_launch=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM(planes, 16)
        else:
            self.cbam = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.cbam is not None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, num=None, downsample=None, use_cbam=False, first_launch=True):
        super(Bottleneck, self).__init__()
        self.num = num
        self.first_launch = first_launch
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM(planes * 4, 16)
        else:
            self.cbam = None

    def forward(self, x_init):
        if not self.first_launch:
            x = x_init[0]
            sam_output_list = x_init[1]
        else:
            x = x_init
            sam_output_list = []

        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        sam_output = None
        if self.cbam is not None:
            out, sam_output = self.cbam(out)
        out += residual
        out = self.relu(out)
        sam_output_list.append(sam_output)
        return out, sam_output_list


class ResNet(nn.Module):
    # block = BottleNeck
    # layers = [3, 4, 6, 3] (depth = 50)
    # network_type = ImageNet
    # num_classes = 5
    # att_type = CBAM
    def __init__(self, block, layers, network_type, num_classes, att_type=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.network_type = network_type
        # different model config between ImageNet and CIFAR 
        if network_type == "ImageNet":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.avgpool = nn.AvgPool2d(7)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        if att_type == 'BAM':
            self.bam1 = BAM(64 * block.expansion)
            self.bam2 = BAM(128 * block.expansion)
            self.bam3 = BAM(256 * block.expansion)
        else:
            self.bam1, self.bam2, self.bam3 = None, None, None

        self.layer1 = self._make_layer(block, 64, layers[0], att_type=att_type)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, att_type=att_type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, att_type=att_type)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, att_type=att_type)

        # self.l0 = self.layerList1[0]
        # self.l1 = self.layerList1[1]
        # self.l2 = self.layerList1[2]
        #
        # self.l3 = self.layerList2[0]
        # self.l4 = self.layerList2[1]
        # self.l5 = self.layerList2[2]
        # self.l6 = self.layerList2[3]
        #
        # self.l7 = self.layerList3[0]
        # self.l8 = self.layerList3[1]
        # self.l9 = self.layerList3[2]
        # self.l10 = self.layerList3[3]
        # self.l11 = self.layerList3[4]
        # self.l12 = self.layerList3[5]
        #
        # self.l13 = self.layerList4[0]
        # self.l14 = self.layerList4[1]
        # self.l15 = self.layerList4[2]

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        init.kaiming_normal_(self.fc.weight)
        for key in self.state_dict():
            if key.split('.')[-1] == "weight":
                if "conv" in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, blocks, stride=1, att_type=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, num=0, downsample=downsample, use_cbam=att_type == 'CBAM', first_launch=True)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, num=i, use_cbam=att_type == 'CBAM', first_launch=False))

        return nn.Sequential(*layers)
        # return layers

    def forward(self, x):
        sam_output = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.network_type == "ImageNet":  # true
            x = self.maxpool(x)

        # x, sam0 = self.layer1[0](x, segm[0])
        # sam_output.append(sam0)
        # x, sam1 = self.layer1[1](x, segm[0])
        # sam_output.append(sam1)
        # x, sam2 = self.layer1[2](x, segm[0])
        # sam_output.append(sam2)
        x, sam1 = self.layer1(x)

        if self.bam1 is not None:  # false
            x, _ = self.bam1(x)

        # x, sam3 = self.layer2[0](x, segm[1])
        # sam_output.append(sam3)
        # x, sam4 = self.layer2[1](x, segm[1])
        # sam_output.append(sam4)
        # x, sam5 = self.layer2[2](x, segm[1])
        # sam_output.append(sam5)
        # x, sam6 = self.layer2[3](x, segm[1])
        # sam_output.append(sam6)

        x, sam2 = self.layer2(x)

        if self.bam2 is not None:  # false
            x, _ = self.bam2(x)

        # x, sam7 = self.layer3[0](x, segm[2])
        # sam_output.append(sam7)
        # x, sam8 = self.layer3[1](x, segm[2])
        # sam_output.append(sam8)
        # x, sam9 = self.layer3[2](x, segm[2])
        # sam_output.append(sam9)
        # x, sam10 = self.layer3[3](x, segm[2])
        # sam_output.append(sam10)
        # x, sam11 = self.layer3[4](x, segm[2])
        # sam_output.append(sam11)
        # x, sam12 = self.layer3[5](x, segm[2])
        # sam_output.append(sam12)

        x, sam3 = self.layer3(x)

        if self.bam3 is not None:  # false
            x, _ = self.bam3(x)

        # x, sam13 = self.layer4[0](x, segm[3])
        # sam_output.append(sam13)
        # x, sam14 = self.layer4[1](x, segm[3])
        # sam_output.append(sam14)
        # x, sam15 = self.layer4[2](x, segm[3])
        # sam_output.append(sam15)

        x, sam4 = self.layer4(x)

        if self.network_type == "ImageNet":
            x = self.avgpool(x)
        else:
            x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        sam_output += sam1
        sam_output += sam2
        sam_output += sam3
        sam_output += sam4
        return x, sam_output


# att_type = CBAM
# network_type = ImageNet
def ResidualNet(network_type, depth, num_classes, att_type):
    assert network_type in ["ImageNet", "CIFAR10", "CIFAR100"], "network type should be ImageNet or CIFAR10 / CIFAR100"
    assert depth in [18, 34, 50, 101], 'network depth should be 18, 34, 50 or 101'
    model = None
    if depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], network_type, num_classes, att_type)

    elif depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], network_type, num_classes, att_type)

    elif depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], network_type, num_classes, att_type)

    elif depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], network_type, num_classes, att_type)

    return model
