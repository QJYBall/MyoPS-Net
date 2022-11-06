import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import numpy as np


class GeneralizedDiceLoss(nn.Module):
    def __init__(self):
        super(GeneralizedDiceLoss, self).__init__()

    def forward(self, predict, target):

        w = 1 / ((einsum("bcwh->bc", target).type(torch.float32) + 1e-10) ** 2)
        intersection = w * einsum("bcwh,bcwh->bc", predict, target)
        union = w * (einsum("bcwh->bc", predict) + einsum("bcwh->bc", target))
        loss = 1 - 2 * (einsum("bc->b", intersection) + 1e-10) / (einsum("bc->b", union) + 1e-10)

        return loss.mean()


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, predict, target):

        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = (predict * target).sum(1)
        den = predict.sum(1) + target.sum(1)

        loss = 1 - (2 * num + 1e-10) / (den + 1e-10)

        return loss.mean()


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, predict, target):
        
        dice = BinaryDiceLoss()
        total_loss = 0

        for i in range(target.shape[1]):
            dice_loss = dice(predict[:, i], target[:, i])
            total_loss += dice_loss

        return total_loss/target.shape[1]


class WCELoss(nn.Module):
    def __init__(self):
        super(WCELoss, self).__init__()
    
    def weight_function(self, target):

        mask = torch.argmax(target, dim=1)
        voxels_sum = mask.shape[0] * mask.shape[1] * mask.shape[2]
        weights = []
        for i in range(mask.max()+1):
            voxels_i = [mask==i][0].sum().cpu().numpy()
            w_i = np.log(voxels_sum/voxels_i).astype(np.float32)
            weights.append(w_i)
        weights = torch.from_numpy(np.array(weights)).cuda()
        
        return weights

    def forward(self, predict, target):

        ce_loss = torch.mean(-target * torch.log(predict + 1e-10), dim=(0,2,3))
        weights = self.weight_function(target)
        loss = weights * ce_loss
        
        return loss.sum()


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self, predict, target):

        ce_loss = -target * torch.log(predict + 1e-10)
        
        return ce_loss.mean()


class InvariantLoss(nn.Module):
    def __init__(self):
        super(InvariantLoss, self).__init__()

    def forward(self, seg_1, seg_2):

        invariant_loss = 1 - F.cosine_similarity(seg_1, seg_2, dim=1)
        
        return invariant_loss.mean()


class InclusiveLossScar(nn.Module):
    def __init__(self):
        super(InclusiveLossScar, self).__init__()

    def forward(self, scar, gd_edema):

        inclusive_loss_scar = -1 * (1 - gd_edema) * torch.log((1 - scar) + 1e-10)
        inclusive_loss_scar = inclusive_loss_scar.sum()/((1-gd_edema).sum() + 1e-10)
        
        return inclusive_loss_scar


class InclusiveLossEdema(nn.Module):
    def __init__(self):
        super(InclusiveLossEdema, self).__init__()

    def forward(self, edema, gd_scar):

        inclusive_loss_edema = -1 * gd_scar * torch.log(edema + 1e-10)
        inclusive_loss_edema = inclusive_loss_edema.sum()/(gd_scar.sum() + 1e-10)
        
        return inclusive_loss_edema


class SegmentationLoss(nn.Module):
    def __init__(self):
        super(SegmentationLoss, self).__init__()

        self.dice_loss = DiceLoss()
        self.wce_loss = WCELoss()

    def forward(self, predict, target):

        dice_loss = self.dice_loss(predict, target)
        wce_loss = self.wce_loss(predict, target)
        loss = wce_loss + dice_loss

        return loss


class MLSCLoss(nn.Module):
    def __init__(self):
        super(MLSCLoss, self).__init__()

        self.seg_loss = SegmentationLoss()
        self.invariant_loss = InvariantLoss()
        self.inclusive_scar = InclusiveLossScar()
        self.inclusive_edema = InclusiveLossEdema()

    def forward(self, seg, label):

        # seg = {'C0': seg_C0, 'LGE': seg_LGE, 'T2': seg_T2, 'mapping': seg_mapping}
        # label = {'cardiac': cardiac_gd, 'scar': scar_gd, 'edema': edema_gd}

        # seg_loss
        loss_seg = self.seg_loss(seg['C0'], label['cardiac']) + \
                2 * self.seg_loss(seg['LGE'], label['scar']) + \
                2 * self.seg_loss(seg['T2'], label['edema']) + \
                2 * self.seg_loss(seg['mapping'], label['scar'])

        # invariant loss
        myo_C0 = torch.cat([seg['C0'][:,0:1,:,:]+seg['C0'][:,2:3,:,:], seg['C0'][:,1:2,:,:]], dim=1)
        myo_LGE = torch.cat([seg['LGE'][:,0:1,:,:], seg['LGE'][:,1:2,:,:]+seg['LGE'][:,2:3,:,:]], dim=1)
        myo_T2 = torch.cat([seg['T2'][:,0:1,:,:], seg['T2'][:,1:2,:,:]+seg['T2'][:,2:3,:,:]], dim=1)
        myo_mapping = torch.cat([seg['mapping'][:,0:1,:,:], seg['mapping'][:,1:2,:,:]+seg['mapping'][:,2:3,:,:]], dim=1)
        loss_invariant = self.invariant_loss(myo_C0, myo_LGE) + \
                        self.invariant_loss(myo_C0, myo_T2) + \
                        self.invariant_loss(myo_C0, myo_mapping)

        # inclusive loss
        loss_inclusive = self.inclusive_scar(seg['LGE'][:,2,:,:], label['edema'][:,2,:,:]) + \
                        self.inclusive_edema(seg['T2'][:,2,:,:], label['scar'][:,2,:,:]) + \
                        self.inclusive_scar(seg['mapping'][:,2,:,:], label['edema'][:,2,:,:])

        loss = loss_seg + loss_invariant + loss_inclusive

        return loss_seg, loss_invariant, loss_inclusive, loss
