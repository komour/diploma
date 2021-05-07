import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock
from .bam import *


class ResNet18BAM(ResNet):

    def __init__(self, pretrained=True):
        super(ResNet18BAM, self).__init__(BasicBlock, [2, 2, 2, 2])
        self.load_state_dict(models.resnet18(pretrained=pretrained).state_dict())

        CLASS_AMOUNT = 5
        self.fc = nn.Linear(512, CLASS_AMOUNT)
        self.bam1 = BAM(64 * BasicBlock.expansion)
        self.bam2 = BAM(128 * BasicBlock.expansion)
        self.bam3 = BAM(256 * BasicBlock.expansion)

    def forward(self, x):
        sam_output = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x, sam1 = self.bam1(x)
        sam_output.append(sam1)

        x = self.layer2(x)
        x, sam2 = self.bam2(x)
        sam_output.append(sam2)

        x = self.layer3(x)
        x, sam3 = self.bam3(x)
        sam_output.append(sam3)

        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, sam_output
