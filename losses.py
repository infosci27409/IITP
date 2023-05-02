import torch
import math
from torch.nn.functional import linear, normalize
import torch.nn.functional as F
import torch.nn as nn


class ArcFace(torch.nn.Module):
    def __init__(self, feature_num, class_num, s=32.0, m=0.5):
        super().__init__()
        self.s = torch.tensor(s)
        self.m = torch.tensor(m)
        self.feature_num = feature_num
        self.class_num = class_num
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.theta = math.cos(math.pi - m)
        self.sinmm = math.sin(math.pi - m) * m
        self.easy_margin = True
        self.w = torch.nn.Parameter(torch.rand(class_num, feature_num))


    def forward(self, feature: torch.Tensor, labels: torch.Tensor):
        feature = normalize(feature)
        w = normalize(self.w)
        logits = linear(feature, w)
        index = torch.where(labels != -1)[0]
        cos_theta = logits[index, labels[index].view(-1)]

        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))

        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        if self.easy_margin:
            final_target_logit = torch.where(
                cos_theta > 0, cos_theta_m, cos_theta)
        else:
            final_target_logit = torch.where(
                cos_theta > self.theta, cos_theta_m, cos_theta - self.sinmm)

        logits[index, labels[index].view(-1)] = final_target_logit
        arc_logits = logits * self.s

        return arc_logits


class CosFace(torch.nn.Module):
    def __init__(self, feature_num, class_num,s=32.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m
        self.feature_num = feature_num
        self.class_num = class_num
        self.w = torch.nn.Parameter(torch.rand(class_num, feature_num))

    def forward(self, feature: torch.Tensor, labels: torch.Tensor):
        feature = normalize(feature)
        w = normalize(self.w)
        logits = linear(feature, w)
        index = torch.where(labels != -1)[0]
        cos_theta = logits[index, labels[index].view(-1)]
        final_target_logit = cos_theta - self.m
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.s

        return logits


class SphereFace(torch.nn.Module):
    def __init__(self, feature_num, class_num, s=32., m=1.5):
        super(SphereFace, self).__init__()
        self.feature_num = feature_num
        self.class_num = class_num
        self.s = s
        self.m = m
        self.w = torch.nn.Parameter(torch.rand(class_num, feature_num))

    def forward(self, feature: torch.Tensor, labels: torch.Tensor):
        feature = normalize(feature)
        w = normalize(self.w)
        logits = linear(feature, w)
        
        with torch.no_grad():
            m_theta = torch.acos(logits.clamp(-1.+1e-5, 1.-1e-5))
            m_theta.scatter_(
                1, labels.view(-1, 1), self.m, reduce='multiply',
            )
            k = (m_theta / math.pi).floor()
            sign = -2 * torch.remainder(k, 2) + 1  # (-1)**k
            phi_theta = sign * torch.cos(m_theta) - 2. * k
            d_theta = phi_theta - logits

        logits = self.s * (logits + d_theta)
        return logits
    
class IsoMaxLossFirstPart(nn.Module):
    """This part replaces the model classifier output layer nn.Linear()"""
    def __init__(self, num_features, num_classes, temperature=1.0):
        super(IsoMaxLossFirstPart, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.temperature = temperature        
        self.prototypes = nn.Parameter(torch.Tensor(num_classes, num_features))
        nn.init.constant_(self.prototypes, 0.0)

    def forward(self, features):
        distances = torch.cdist(features, self.prototypes, p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
        logits = -distances
        # The temperature may be calibrated after training to improve uncertainty estimation.
        return logits / self.temperature


class IsoMaxLossSecondPart(nn.Module):
    """This part replaces the nn.CrossEntropyLoss()"""
    def __init__(self, entropic_scale=10.0):
        super(IsoMaxLossSecondPart, self).__init__()
        self.entropic_scale = entropic_scale

    def forward(self, logits, targets, debug=False):

        distances = -logits
        probabilities_for_training = nn.Softmax(dim=1)(-self.entropic_scale * distances)
        probabilities_at_targets = probabilities_for_training[range(distances.size(0)), targets]
        loss = -torch.log(probabilities_at_targets).mean()
        if not debug:
            return loss
        else:
            targets_one_hot = torch.eye(distances.size(1))[targets].long().cuda()
            intra_inter_distances = torch.where(targets_one_hot != 0, distances, torch.Tensor([float('Inf')]).cuda())
            inter_intra_distances = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), distances)
            intra_distances = intra_inter_distances[intra_inter_distances != float('Inf')]
            inter_distances = inter_intra_distances[inter_intra_distances != float('Inf')]
            return loss, 1.0, intra_distances, 