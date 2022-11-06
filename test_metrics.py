import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from medpy import metric
from utils.config import config
from process import PostProcess


def img_load(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

#extract ROI
def get_ROI(img, index):
    for key, value in index.items():
        img = np.where(img == key, value, img)
    return img 


def calculate(seg_path, gd_path, index):
    labels = np.array([0, 200, 500, 600,1220,2221])
    labels_ = np.array([0, 200, 500, 600,1220,2221])
    if index == 1220: #edema
        trans_path = np.array([0, 0, 0, 0, 1, 1])
        trans_myo = np.array([0, 1, 0, 0, 0, 0])
    if index == 2221: #scar
        trans_path = np.array([0, 0, 0, 0, 0, 1])
        trans_myo = np.array([0, 1, 0, 0, 1, 0])

    seg = img_load(seg_path)
    gd = img_load(gd_path)
    
    # ph refers to pathological
    seg_path = get_ROI(seg,dict(zip(labels, trans_path)))
    gd_path = get_ROI(gd,dict(zip(labels_,trans_path)))
    gd_myo = get_ROI(gd,dict(zip(labels_, trans_myo)))

    #全集
    all = seg_path + gd_path + gd_myo

    P = np.where((seg_path+gd_path)==2, 3, 0)
    if np.sum(P)==0:
        TP = 0
        FN = np.sum(gd_path)
        FP = np.sum(seg_path)
    else:
        TP = np.sum(np.where((seg_path+P)==4, 1, 0))
        FN = np.sum(np.where((gd_path-P)==1, 1, 0))
        FP = np.sum(np.where((seg_path-P)==1, 1, 0))
    
    All = np.where(all>0, 1, 0)
    
    TN = np.sum(All) - TP - FN - FP
    
    if (TP+TN)==0 and (TP+FP+FN+TN)==0:
        acc = 1
    else:
        acc = float(TP+TN)/float(TP+FP+FN+TN)  
    
    if TP ==(TP+FN)==0:
        sen = 1
    else:
        sen = float(TP)/float(TP+FN)

    if TN ==(TP+FP)==0:
        spe = 1
    else:
        spe = float(TN)/float(TN+FP)

    if (TP + FN) == 0:
        dice = 'Null'
    else:
        dice = 2*TP/((TP + FP) + (TP +FN))
    
    try:
        hd = metric.hd(seg_path, gd_path)
    except:
        hd = 0
    
    return acc, sen, spe, dice, hd


def main(args):

    post_process = PostProcess()
    post_process(args.test_path, 60, 200)
    
    result = []

    for i in range(2031,2051):

        file = args.test_path + '/PP_Case'+str(i)+'_result.nii.gz'
        gd = 'Data/test/Case'+str(i)+'/'+'Case'+str(i)+'_gd.nii.gz'

        try:
            acc_s, sen_s, spe_s, dice_s, hd_s = calculate(file, gd, 2221) #scar
        except:
            acc_s, sen_s, spe_s, dice_s, hd_s = 'Null', 'Null', 'Null', 'Null', 'Null'

        try:
            acc_e, sen_e, spe_e, dice_e, hd_e = calculate(file, gd, 1220) #edema
        except:
            acc_e, sen_e, spe_e, dice_e, hd_e = 'Null', 'Null', 'Null', 'Null', 'Null'
        
        result.append((dice_s, hd_s, acc_s, sen_s, spe_s, dice_e, hd_e, acc_e, sen_e, spe_e))

    result_df = pd.DataFrame(result, columns=['scar_dice', 'scar_hd', 'scar_acc', 'scar_sen', 'scar_spe', 'edema_dice', 'edema_hd', 'edema_acc', 'edema_sen', 'edema_spe'])
    result_df.to_csv('result.csv')
    print("Done!")


if __name__ == '__main__':
    args = config()
    main(args)
    
    