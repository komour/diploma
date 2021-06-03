import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock
from .bam import *
from .cbam import SpatialGate as CBAM_SAM


class ResNet18BAM(ResNet):

    def __init__(self, pretrained=True, sam_instead_bam=False):
        self.sam_instead_bam = sam_instead_bam
        super(ResNet18BAM, self).__init__(BasicBlock, [2, 2, 2, 2])
        self.load_state_dict(models.resnet18(pretrained=pretrained).state_dict())

        CLASS_AMOUNT = 5
        self.fc = nn.Linear(512, CLASS_AMOUNT)
        if self.sam_instead_bam:
            self.sam0 = None
            self.sam1 = None
            # self.sam1 = SpatialGate(64 * BasicBlock.expansion)
            self.sam2 = None
            # self.sam3 = SpatialGate(256 * BasicBlock.expansion)
            self.sam3 = None
            self.sam4 = None
        else:
            self.bam1 = None
            # self.bam1 = BAM(64 * BasicBlock.expansion)
            self.bam2 = None
            # self.bam2 = BAM(128 * BasicBlock.expansion)
            # self.bam3 = None
            self.bam3 = BAM(256 * BasicBlock.expansion)
            self.bam4 = None

    def forward(self, x):
        sam_output = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if self.sam_instead_bam:
            if self.sam0 is not None:
                x, sam_o0 = self.sam0(x)
                # print(sam_o0.size())
                sam_output.append(sam_o0)
        x = self.layer1(x)
        if self.sam_instead_bam:
            if self.sam1 is not None:
                x, sam_o1 = self.sam1(x)
                # x = x * sam_o1
                sam_output.append(sam_o1)
        else:
            if self.bam1 is not None:
                x, sam_o1 = self.bam1(x)
                sam_output.append(sam_o1)

        x = self.layer2(x)
        if self.sam_instead_bam:
            if self.sam2 is not None:
                x, sam_o2 = self.sam2(x)
                # x = x * (1 + sam_o2)
                sam_output.append(sam_o2)
        else:
            if self.bam2 is not None:
                x, sam_o2 = self.bam2(x)
                sam_output.append(sam_o2)

        x = self.layer3(x)
        if self.sam_instead_bam:
            if self.sam3 is not None:
                _, sam_o3 = self.sam3(x)
                x = x * sam_o3
                sam_output.append(sam_o3)
        else:
            if self.bam3 is not None:
                x, sam_o3 = self.bam3(x)
                sam_output.append(sam_o3)

        x = self.layer4(x)

        if self.sam_instead_bam:
            if self.sam4 is not None:
                _, sam_o4 = self.sam4(x)
                x = x * (1 + sam_o4)
                sam_output.append(sam_o4)
        else:
            if self.bam4 is not None:
                x, sam_o4 = self.bam4(x)
                sam_output.append(sam_o4)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, sam_output