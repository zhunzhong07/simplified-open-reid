from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import pdb
from ..evaluation_metrics import accuracy

class TripletLoss_biu(nn.Module):
    def __init__(self, margin=0, num_instances=4, alpha=1.0, beta=0.5, gamma =0.5):
        super(TripletLoss_biu, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.num_instances = num_instances
        self.alpha = alpha 
        self.beta = beta
        self.gamma = gamma
        #print (self.alpha,self.beta)
        self.xentropy_loss = nn.CrossEntropyLoss()
        


    def forward(self, inputs, targets):

        n = inputs.size(0)
        num_person=n // self.num_instances
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(1).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
       
        dist_ap, dist_an = [], []
        cov_p=[]
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        
        TripletLoss = self.ranking_loss(dist_an, dist_ap, y)
        xentropy = self.xentropy_loss(inputs,targets) 

        loss = self.alpha*TripletLoss +  self.gamma *xentropy
        #pdb.set_trace()
        #print('Triplet-Loss is :{},  Cross-Entropy-Loss is :{}'.format(TripletLoss,xentropy))
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        #pdb.set_trace()
        accuracy_val,  = accuracy(inputs.data, targets.data)
        accuracy_val = accuracy_val[0]
        prec2 = max(prec,accuracy_val)

        return loss, prec2 
