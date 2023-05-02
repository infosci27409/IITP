import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
# from detect_gram import G_p


#########################################
# LeNet-5
#########################################

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size = 5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size = 5, stride=1)
        self.conv3 = nn.Conv2d(16, 120, kernel_size = 5 , stride=1)
        self.fc1 = nn.Linear(120, 84)
        self.linear = torch.nn.Parameter(torch.rand(10, 84))
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))

        x = torch.flatten(x, 1)
        embedd = self.fc1(x)
        logit = F.linear(embedd, self.linear)

        return embedd, logit

#########################################
# ResNet
#########################################

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, feature_num = 512):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = torch.nn.Parameter(torch.rand(num_classes, feature_num))

        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def feature_list(self, x):
        out_list = []
        out = self.bn1(self.conv1(x))
        out_list.append(out)
        out = self.layer1(F.relu(out))
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = self.layer3(out)
        out_list.append(out)
        out = self.layer4(out)
        out_list.append(out)
        out = F.avg_pool2d(out, 4)
        embedd = out.view(out.size(0), -1)
        logit = F.linear(embedd, self.linear)
        return logit, out_list

    
    def intermediate_forward(self, x, layer_index):
        out = self.bn1(self.conv1(x))
        if layer_index == 1:
            out = self.layer1(F.relu(out))       
        elif layer_index == 2:
            out = self.layer1(F.relu(out))       
            out = self.layer2(out)
        elif layer_index == 3:
            out = self.layer1(F.relu(out))       
            out = self.layer2(out)
            out = self.layer3(out)
        elif layer_index == 4:
            out = self.layer1(F.relu(out))       
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
        return out

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        embedd = out.view(out.size(0), -1)
        logit = F.linear(embedd, self.linear)
        return embedd, logit

    
    def forward_threshold(self, x, threshold):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.clip(max=threshold)
        embedd = out.view(out.size(0), -1)
        logit = F.linear(embedd, self.linear)
        return embedd, logit
    
class tResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, feature_num = 512):
        super(tResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(feature_num, num_classes, bias=False)
        self.linear.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal(m.weight.data)
                
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        embedd = out.view(out.size(0), -1)
        tmp = F.normalize(embedd, dim=1, p=2)
        logit = torch.abs(self.linear(tmp))
        
        return embedd, logit

def tResNet18():
    return tResNet(BasicBlock, [2,2,2,2])

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])


class ResNetCosine(ResNet):
    def __init__(self, block, num_blocks, num_classes=10, feature_num = 512):
        super(ResNetCosine, self).__init__(block, num_blocks, num_classes=10, feature_num = 512)
        self.fc = nn.Linear(512, num_classes, bias=False)
        self.fc_w = nn.Parameter(self.fc.weight)
        self.bn_scale = nn.BatchNorm1d(1)
        self.fc_scale = nn.Linear(512, 1)

    def forward(
        self, x, y=None, mixup=None, alpha=None, all_pred=False,
        candidate_layers=[0, 1, 2, 3],):
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        embedd = out.view(out.size(0), -1)
        
        scale = torch.exp(self.bn_scale(self.fc_scale(embedd)))
        x_norm = F.normalize(embedd)
        w_norm = F.normalize(self.fc_w)

        cos_sim = F.linear(x_norm, w_norm) # cos_theta
        scaled_cosine = cos_sim * scale

        return scaled_cosine, scale, cos_sim
    
    
class entropic_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, loss_first_part=None):
        super(entropic_ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = loss_first_part(512, 10)

        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        embedd = out.view(out.size(0), -1)
        logit = self.linear(embedd)
        return embedd, logit
#########################################
# AlexNet
#########################################

class AlexNet(nn.Module) :
    def __init__(self) :
        super(AlexNet, self).__init__()
        self.name = "AlexNet"
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=(4, 4)),
            nn.ReLU(inplace=True),
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fc_layer1 = nn.Sequential(
            nn.Dropout(p = 0),
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0),
            nn.Linear(4096, 4096),
        )
        self.fc2 = torch.nn.Parameter(torch.rand(10, 4096))

    
    def forward(self, x) :
        output = self.conv_layer1(x)
        output = self.conv_layer2(output)
        output = self.conv_layer3(output)
        output = output.view(-1, 9216)
        
        embedd = self.fc_layer1(output)
        logit = F.linear(F.relu(embedd), self.fc2)
        return embedd, logit


    def forward_threshold(self, x, threshold=1e10) :
        output = self.conv_layer1(x)
        output = self.conv_layer2(output)
        output = self.conv_layer3(output)
        output = output.clip(max=threshold)
        output = output.view(-1, 9216)
        embedd = self.fc_layer1(output)
        logit = F.linear(F.relu(embedd), self.fc2)
        return embedd, logit


class tAlexNet(nn.Module) :
    def __init__(self) :
        super(AlexNet, self).__init__()
        self.name = "AlexNet"
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=(4, 4)),
            nn.ReLU(inplace=True),
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fc_layer1 = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(4096, 4096),
        )

        self.fc2 = nn.Linear(4096, 10, bias=False)
        self.fc2.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal(m.weight.data)

        
    def forward(self, x) :
        output = self.conv_layer1(x)
        output = self.conv_layer2(output)
        output = self.conv_layer3(output)
        output = output.view(-1, 9216)
        embedd = self.fc_layer1(output)
        tmp = F.normalize(embedd, dim=1, p=2)
        logit = torch.abs(self.fc2(tmp))

        return embedd, logit


    def forward_threshold(self, x, threshold=1e10) :
        output = self.conv_layer1(x)
        output = self.conv_layer2(output)
        output = self.conv_layer3(output)
        output = output.clip(max=threshold)
        output = output.view(-1, 9216)
        embedd = self.fc_layer1(output)
        logit = F.linear(F.relu(embedd), self.fc2)
        return embedd, logit
    
    
class AlexNetCosine(AlexNet):
    def __init__(self, num_classes=10, feature_num = 4096):
        super(AlexNetCosine, self).__init__()
        self.fc = nn.Linear(4096, num_classes, bias=False)
        self.fc_w = nn.Parameter(self.fc.weight)
        self.bn_scale = nn.BatchNorm1d(1)
        self.fc_scale = nn.Linear(4096, 1)

    def forward(self, x) :
        output = self.conv_layer1(x)
        output = self.conv_layer2(output)
        output = self.conv_layer3(output)
        output = output.view(-1, 9216)
        
        embedd = self.fc_layer1(output)
        scale = torch.exp(self.bn_scale(self.fc_scale(embedd)))
        x_norm = F.normalize(embedd)
        w_norm = F.normalize(self.fc_w)
        
        cos_sim = F.linear(x_norm, w_norm)
        scaled_cosine = cos_sim * scale
        
        return scaled_cosine, scale, cos_sim
    
    
class entropic_AlexNet(nn.Module) :
    def __init__(self, loss_first_part=None) :
        super(entropic_AlexNet, self).__init__()
        self.name = "AlexNet"
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=(4, 4)),
            nn.ReLU(inplace=True),
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fc_layer1 = nn.Sequential(
            nn.Dropout(p = 0),
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0),
            nn.Linear(4096, 4096),
        )
        self.fc2 = loss_first_part(4096, 10)

    def forward(self, x) :
        output = self.conv_layer1(x)
        output = self.conv_layer2(output)
        output = self.conv_layer3(output)
        output = output.view(-1, 9216)
        
        embedd = self.fc_layer1(output)
        logit = self.fc2(embedd)
        return embedd, logit
