import torch
import random
import numpy as np
import torch.nn as nn
from torchvision import transforms
from skimage.transform import resize


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class Normalization(object):
    def __call__(self, image, mode):

        if mode == 'Max_Min':
            eps = 1e-8
            mn = image.min()
            mx = image.max()
            image_normalized = (image - mn) / (mx - mn + eps)
        
        if mode == 'Zero_Mean_Unit_Std':
            eps = 1e-8
            mean = image.mean()
            std = image.std()
            image_normalized = (image - mean) / (std + eps)

        if mode == 'Truncate':
            # truncate
            
            Hist, _ = np.histogram(image, bins=int(image.max()))

            idexs = np.argwhere(Hist >= 20)
            idex_min = np.float32(0)
            idex_max = np.float32(idexs[-1, 0])

            image[np.where(image <= idex_min)] = idex_min
            image[np.where(image >= idex_max)] = idex_max

            # normalize
            sig = image[0, 0, 0]
            image = np.where(image != sig, image - np.mean(image[image != sig]), 0 * image)
            image_normalized = np.where(image != sig, image / np.std(image[image != sig] + 1e-8), 0 * image)
        
        return image_normalized


class RandomSizedCrop(object):
    def __init__(self, dim):
        self.crop_size = dim

    def __call__(self, image):
        # RandomCrop
        scaler = np.random.uniform(0.9, 1.1)
        scale_size = int(self.crop_size * scaler)
        h_off = random.randint(0, image.shape[1] - 0 - scale_size)
        w_off = random.randint(0, image.shape[2] - 0 - scale_size)
        image = image[:, h_off:h_off+scale_size, w_off:w_off+scale_size]

        # Resize
        image = image.numpy() 

        C0_slice = image[:1,...]
        LGE_slice = image[1:2,...]
        T2_slice = image[2:3,...]
        T1m_slice = image[3:4,...]
        T2starm_slice = image[4:5,...]
        label_slice = image[5:,...]
        
        output_shape = (1, self.crop_size, self.crop_size)

        C0_resized = resize(C0_slice, output_shape, order=1, mode='constant', preserve_range=True)
        LGE_resized = resize(LGE_slice, output_shape, order=1, mode='constant', preserve_range=True)
        T2_resized = resize(T2_slice, output_shape, order=1, mode='constant', preserve_range=True)
        T1m_resized = resize(T1m_slice, output_shape, order=1, mode='constant', preserve_range=True)
        T2starm_resized = resize(T2starm_slice, output_shape, order=1, mode='constant', preserve_range=True)
        label_resized = resize(label_slice, output_shape, order=0, mode='edge', preserve_range=True)

        image = np.concatenate([C0_resized, LGE_resized, T2_resized, T1m_resized, T2starm_resized, label_resized], axis=0)
        image = torch.from_numpy(image).float()    
        return image


class ToTensor(object):
    def __call__(self, image):
        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        image = torch.from_numpy(image).float()
        return image


# image transform
class ImageTransform(object):
    def __init__(self, dim, stage):  
        self.dim = dim
        self.stage = stage

    def __call__(self, image):

        if self.stage == 'Train':
            transform = transforms.Compose([
                ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(180),
                RandomSizedCrop(self.dim)
            ])     
        
        if self.stage == 'Valid' or self.stage == 'Test':
            transform = transforms.Compose([
                ToTensor(),
                transforms.CenterCrop(self.dim)
            ])

        return transform(image)


# label transform
class LabelTransform(object):
    def __init__(self, stage):
        self.stage = stage

    def __call__(self, label):
        
        label = label.numpy()
        cardiac_gd = self.label_transform(label, 'cardiac')
        scar_gd = self.label_transform(label, 'scar')
        edema_gd = self.label_transform(label, 'edema')
        pathology_gd = self.label_transform(label, 'pathology')

        if self.stage == 'Train':
            cardiac_gd = self.convert_onehot(cardiac_gd, 3) # bg myo lv
            scar_gd = self.convert_onehot(scar_gd, 3) # bg myo scar
            edema_gd = self.convert_onehot(edema_gd, 3) # bg myo edema
            pathology_gd = self.convert_onehot(pathology_gd, 3) # bg scar edema

        return cardiac_gd, scar_gd, edema_gd, pathology_gd

    def convert_onehot(self, label, num_class):
        label = label.long()
        label_onehot = torch.zeros((num_class, label.shape[1], label.shape[2]))
        label_onehot.scatter_(0, label, 1).float()
        return label_onehot 

    def label_transform(self, label, mode):

        # 0 bg, 1 myo, 2 lv
        if mode == 'cardiac':
            label = np.where(label == 200, 1, label)
            label = np.where(label == 500, 2, label)
            label = np.where(label == 600, 0, label)
            label = np.where(label == 1220, 1, label)
            label = np.where(label == 2221, 1, label)
    
        # 0 bg, 1 myo, 2 scar
        if mode == 'scar':
            label = np.where(label == 200, 1, label)
            label = np.where(label == 500, 0, label)
            label = np.where(label == 600, 0, label)
            label = np.where(label == 1220, 1, label)
            label = np.where(label == 2221, 2, label)
    
        # 0 bg, 1 myo-edema, 2 edema
        if mode == 'edema':
            label = np.where(label == 200, 1, label)
            label = np.where(label == 500, 0, label)
            label = np.where(label == 600, 0, label)
            label = np.where(label == 1220, 2, label)
            label = np.where(label == 2221, 2, label)
    
        # 0 bg, 1 edema, 2 scar
        if mode == 'pathology':
            label = np.where(label == 200, 0, label)
            label = np.where(label == 500, 0, label)
            label = np.where(label == 600, 0, label)
            label = np.where(label == 1220, 1, label)
            label = np.where(label == 2221, 2, label)    
        
        label = torch.from_numpy(label).float()
        return label


# result transform
class ResultTransform(object):
    def __init__(self, ToOriginal = False):      
        self.flag = ToOriginal

    def __call__(self, seg_scar_LGE, seg_scar_mapping, seg_edema):

        seg_scar_LGE = seg_scar_LGE.numpy()
        seg_scar_mapping = seg_scar_mapping.numpy()
        seg_edema = seg_edema.numpy()

        # transform scar
        seg_scar_LGE = np.where(seg_scar_LGE == 1, 0, seg_scar_LGE) # 0 bg, 2 scar
        seg_scar_mapping = np.where(seg_scar_mapping == 1, 0, seg_scar_mapping) # 0 bg, 2 scar
        seg_scar = seg_scar_LGE + seg_scar_mapping # 0 bg, 2&4 scar
        seg_scar = np.where(seg_scar == 4, 2, seg_scar) # 0 bg, 2 scar

        # transform edema
        seg_edema = np.where(seg_edema == 1, 0, seg_edema) # 0 bg, 2 edema
        seg_edema = np.where(seg_edema == 2, 1, seg_edema) # 0 bg, 1 edema
        
        # transform pathology
        seg_pathology = seg_scar + seg_edema # 0 bg, 1 edema, 2+3 scar
        seg_pathology = np.where(seg_pathology == 3, 2, seg_pathology) # 0 bg, 1 edema, 2 scar

        if self.flag == True:
            seg_pathology = np.where(seg_pathology == 1, 1220, seg_pathology) # 1 - edema - 1220
            seg_pathology = np.where(seg_pathology == 2, 2221, seg_pathology) # 2 - scar - 2221
        
        seg_pathology = torch.from_numpy(seg_pathology)
        
        return seg_pathology
