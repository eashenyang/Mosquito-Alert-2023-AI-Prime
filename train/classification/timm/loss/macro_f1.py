import torch
import torch.nn as nn

class MacroSoftF1(nn.Module):

    def __init__(self):
        super(MacroSoftF1, self).__init__()
    
    # @torch.cuda.amp.autocast(nn.Module)
    def forward(self, y_hat, y):
        
        y_hat = torch.sigmoid(y_hat)

        tp = torch.sum(y_hat * y, dim=0)
        fp = torch.sum(y_hat * (1 - y), dim=0)
        fn = torch.sum((1 - y_hat) * y, dim=0)

        macroSoft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
        cost = 1 - macroSoft_f1
        macroCost = torch.mean(cost)

        return macroCost