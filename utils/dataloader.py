import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from utils.tools import Normalization, ImageTransform, LabelTransform
    

class CrossModalDataLoader(Dataset):

    def __init__(self, path, file_name, dim, max_iters = None, stage = 'Train'):

        self.path = path
        self.crop_size = dim
        self.stage = stage

        self.Img = [item.strip().split() for item in open(self.path + file_name)]
        
        if max_iters != None:
            self.Img = self.Img * int(np.ceil(float(max_iters) / len(self.Img)))
        
        self.files = []

        for item in self.Img:
            
            img_path, gt_path, imgidx = item
            
            C0_path = img_path + '_C0.nii.gz'
            LGE_path = img_path + '_LGE.nii.gz'
            T2_path = img_path + '_T2.nii.gz'
            T1m_path = img_path + '_T1m.nii.gz'
            T2starm_path = img_path + '_T2starm.nii.gz'
            label_path = gt_path + '_gd.nii.gz'

            C0_file = os.path.join(self.path, C0_path)
            LGE_file = os.path.join(self.path, LGE_path)
            T2_file = os.path.join(self.path, T2_path)
            T1m_file = os.path.join(self.path, T1m_path)
            T2starm_file = os.path.join(self.path, T2starm_path)
            label_file = os.path.join(self.path, label_path)
            
            self.files.append({
                "C0": C0_file,
                "LGE": LGE_file,
                "T2": T2_file,
                "T1m": T1m_file,
                "T2starm": T2starm_file,
                "label": label_file,
                "index": int(imgidx)
            })

        self.normalize = Normalization()
        self.image_transform = ImageTransform(self.crop_size, self.stage)
        self.label_transform = LabelTransform(self.stage)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        file_path = self.files[index]

        # get raw data
        C0_raw = nib.load(file_path["C0"])
        LGE_raw = nib.load(file_path["LGE"])
        T2_raw = nib.load(file_path["T2"])
        T1m_raw = nib.load(file_path["T1m"])
        T2starm_raw = nib.load(file_path["T2starm"])
        gd_raw = nib.load(file_path["label"])
        imgidx = file_path["index"]

        # get data [x,y,z] & normalize
        C0_img = self.normalize(C0_raw.get_fdata(),'Truncate')
        LGE_img = self.normalize(LGE_raw.get_fdata(),'Truncate')
        T2_img = self.normalize(T2_raw.get_fdata(),'Truncate')
        T1m_img = T1m_raw.get_fdata()
        if int(T1m_img.max()) != 0:
            T1m_img = self.normalize(T1m_img,'Truncate')
        T2starm_img = self.normalize(T2starm_raw.get_fdata(),'Truncate')
        gd_img = gd_raw.get_fdata()
        
        # cut slice [x,y,1] -> [x,y,5]
        C0_slice = C0_img[:,:,imgidx:imgidx+1].astype(np.float32)
        LGE_slice = LGE_img[:,:,imgidx:imgidx+1].astype(np.float32)
        T2_slice = T2_img[:,:,imgidx:imgidx+1].astype(np.float32)
        T1m_slice = T1m_img[:,:,imgidx:imgidx+1].astype(np.float32)
        T2starm_slice = T2starm_img[:,:,imgidx:imgidx+1].astype(np.float32)
        label_slice = gd_img[:,:,imgidx:imgidx+1].astype(np.float32)
        image = np.concatenate([C0_slice,LGE_slice,T2_slice,T1m_slice,T2starm_slice,label_slice], axis=2)
        img_C0, img_LGE, img_T2, img_T1m, img_T2starm, label = torch.chunk(self.image_transform(image), chunks=6, dim=0)

        img_C0 = self.normalize(img_C0, 'Zero_Mean_Unit_Std')
        img_LGE = self.normalize(img_LGE, 'Zero_Mean_Unit_Std')
        img_T2 = self.normalize(img_T2, 'Zero_Mean_Unit_Std')
        img_T1m = self.normalize(img_T1m, 'Zero_Mean_Unit_Std')
        img_T2starm = self.normalize(img_T2starm, 'Zero_Mean_Unit_Std')

        # label transform [class,H,W]
        label_cardiac, label_scar, label_edema, label_pathology = self.label_transform(label)

        return img_C0, img_LGE, img_T2, img_T1m, img_T2starm, label_cardiac, label_scar, label_edema, label_pathology
