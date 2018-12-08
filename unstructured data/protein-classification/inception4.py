import torch
import torch.nn as nn
import torchvision
import pretrainedmodels
    
class Inception4Channel(nn.Module):
    def __init__(self, num_classes=28):
        super().__init__()
        
        encoder = pretrainedmodels.models.inceptionv4(pretrained='imagenet')
        
        # we initialize this conv to take in 4 channels instead of 3
        # we keeping corresponding weights and initializing new weights with 1 of the trained channels
        # this trick taken from https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
        w = encoder.features[0].conv.weight
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(torch.cat((w,w[:,:1,:,:]),dim=1))
        
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True) 
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception = encoder.features 
        
        self.avg_pool = encoder.avg_pool 
        self.last_linear = encoder.last_linear
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.inception(x)

        x = self.avg_pool(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)

        return x