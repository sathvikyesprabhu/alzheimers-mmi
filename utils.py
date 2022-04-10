import torch
import torch.nn as nn
import torch.nn.functional as F

import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix

def reweight(cls_num_list, beta=0.9999):
    '''
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    '''
    per_cls_weights = [(1-beta)/(1-beta**cls_num_list[i]) for i in range(len(cls_num_list))]

    return per_cls_weights


# class FocalLoss(nn.Module):
#     def __init__(self, weight=None, gamma=0.):
#         super(FocalLoss, self).__init__()
#         assert gamma >= 0
#         self.gamma = gamma
#         self.weight = weight

#     def forward(self, input, target):
#         '''
#         Implement forward of focal loss
#         :param input: input predictions
#         :param target: labels
#         :return: tensor of focal loss in scalar
#         '''
#         # print(input.shape)
#         # print(target.shape) 
#         input = F.log_softmax(input, dim=1)
        
#         loss = 0.0
#         for n in range(len(target)):
#             yn = target[n]
#             pt = input[n,yn]
#             loss = loss + self.weight[yn]*((1-torch.exp(pt))**self.gamma)*(-pt)
            
#         return loss

class FocalLoss(nn.Module):
    
    def __init__(self, weight=None, 
                 gamma=2., reduction='sum'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob, 
            target_tensor, 
            weight=self.weight,
            reduction = self.reduction
        )
    
    
def get_entropy(logits: torch.Tensor) -> torch.Tensor:
    r"""Compute entropy according to the definition.

    Args:
        logits: Unscaled log probabilities.

    Return:
        A tensor containing the Shannon entropy in the last dimension.
    """
    probs = F.softmax(logits, -1) + 1e-8
    entropy = - torch.sum(probs * torch.log(probs))
    return entropy

# Loss curves
def plot_loss(train_loss_list,test_loss_list):
    plt.plot(train_loss_list)
    plt.plot(test_loss_list)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.ylim(bottom = 0)
    plt.legend(["Train loss", "Test loss"])
    plt.show()

def print_report(y,yhat, class_names):
    y = np.hstack(y)
    yhat = np.hstack(yhat)
    report = classification_report(y, yhat, target_names=class_names, digits=4)
    print(report)

    cm=confusion_matrix(y,yhat)

    # Enhanced confusion matrix
    plt.figure(figsize=(6,4))
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names).astype(int)
    heatmap = sn.heatmap(df_cm, cmap="Blues", annot=True, fmt="d", annot_kws={'size':16})

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=12)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right',fontsize=12)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    return report