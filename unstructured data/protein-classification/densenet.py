import torch
import torch.nn as nn
import torchvision
    
DENSENET_ENCODERS = {
    121: torchvision.models.densenet121,
    201: torchvision.models.densenet201
}
    
class Densenet4Channel(nn.Module):
    def __init__(self, encoder_depth=121, pretrained=True, num_classes=28):
        super().__init__()
        
        encoder = DENSENET_ENCODERS[encoder_depth](pretrained=pretrained)
        
        # we initialize this conv to take in 4 channels instead of 3
        # we keeping corresponding weights and initializing new weights with 1 of the trained channels
        # this trick taken from https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
        w = encoder.features.conv0.weight
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(torch.cat((w,w[:,:1,:,:]),dim=1))
        
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True) 
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.dense1 = encoder.features.denseblock1
        self.trans1 = encoder.features.transition1
        self.dense2 = encoder.features.denseblock2
        self.trans2 = encoder.features.transition2
        self.dense3 = encoder.features.denseblock3
        self.trans3 = encoder.features.transition3
        self.dense4 = encoder.features.denseblock4
        
        self.norm5 = encoder.features.norm5
        
        self.relu2 = nn.ReLU(inplace=True) 
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = encoder.classifier
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        x = self.trans3(x)
        x = self.dense4(x)

        x = self.norm5(x)
        x = self.relu2(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x