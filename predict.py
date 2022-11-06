import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
import nibabel as nib
from utils.config import config
from network.model import CrossModalSegNet
from process import LargestConnectedComponents
from utils.tools import Normalization, ImageTransform, ResultTransform


def predict(args, model_path, epoch):

    # model definition
    model = CrossModalSegNet(in_chs=(5,2,2,3), out_chs=(3,3,3,3))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    if not os.path.exists('test/test_' + str(epoch)):
        os.makedirs('test/test_' + str(epoch))
    
    test_img = pd.read_csv(args.path + 'Zhongshan/test.csv')

    normalize = Normalization()
    keepLCC = LargestConnectedComponents()
    image_transform = ImageTransform(args.dim, 'Test')
    result_transform = ResultTransform(ToOriginal=True)

    for i in range(int(len(test_img))):

        prefix_data = os.path.join(args.path + 'Zhongshan/' + test_img.iloc[i]["stage"], test_img.iloc[i]["file_name"])
        dim_x, dim_y, dim_z = test_img.iloc[i]["dx"], test_img.iloc[i]["dy"], test_img.iloc[i]["dz"]
        result = torch.zeros([dim_z, dim_x, dim_y])
        
        # get data [x,y,z]
        C0_raw = nib.load(prefix_data + '_C0.nii.gz')
        LGE_raw = nib.load(prefix_data + '_LGE.nii.gz')
        T2_raw = nib.load(prefix_data + '_T2.nii.gz')
        T1m_raw = nib.load(prefix_data + '_T1m.nii.gz')
        T2starm_raw = nib.load(prefix_data + '_T2starm.nii.gz')
        img_affine = C0_raw.affine

        C0_img = normalize(C0_raw.get_fdata(), 'Truncate').astype(np.float32)
        LGE_img = normalize(LGE_raw.get_fdata(), 'Truncate').astype(np.float32)
        T2_img = normalize(T2_raw.get_fdata(), 'Truncate').astype(np.float32)
        T1m_img = normalize(T1m_raw.get_fdata(), 'Truncate').astype(np.float32)
        T2starm_img = normalize(T2starm_raw.get_fdata(), 'Truncate').astype(np.float32)

        oridata = np.concatenate([C0_img, LGE_img, T2_img, T1m_img, T2starm_img], axis=2)
        img_C0, img_LGE, img_T2, img_T1m, img_T2starm = torch.chunk(image_transform(oridata), chunks=5, dim=0)
    
        test_C0 = torch.FloatTensor(1, 1, args.dim, args.dim)
        test_LGE = torch.FloatTensor(1, 1, args.dim, args.dim)
        test_T2 = torch.FloatTensor(1, 1, args.dim, args.dim)
        test_T1m = torch.FloatTensor(1, 1, args.dim, args.dim)
        test_T2starm = torch.FloatTensor(1, 1, args.dim, args.dim)

        seg_LGE = torch.FloatTensor(dim_z, args.dim, args.dim)
        seg_T2 = torch.FloatTensor(dim_z, args.dim, args.dim)
        seg_mapping = torch.FloatTensor(dim_z, args.dim, args.dim)

        for j in range(dim_z):

            img_C0_slice = normalize(img_C0[j:j+1,...], 'Zero_Mean_Unit_Std')
            img_LGE_slice = normalize(img_LGE[j:j+1,...], 'Zero_Mean_Unit_Std')
            img_T2_slice = normalize(img_T2[j:j+1,...], 'Zero_Mean_Unit_Std')
            img_T1m_slice = normalize(img_T1m[j:j+1,...], 'Zero_Mean_Unit_Std')
            img_T2starm_slice = normalize(img_T2starm[j:j+1,...], 'Zero_Mean_Unit_Std')

            test_C0.copy_(img_C0_slice.unsqueeze(0))
            test_LGE.copy_(img_LGE_slice.unsqueeze(0))
            test_T2.copy_(img_T2_slice.unsqueeze(0))
            test_T1m.copy_(img_T1m_slice.unsqueeze(0))
            test_T2starm.copy_(img_T2starm_slice.unsqueeze(0))

            _, res_LGE, res_T2, res_mapping = model(test_C0, test_LGE, test_T2, test_T1m, test_T2starm)

            seg_LGE[j:j+1,:,:].copy_(torch.argmax(res_LGE, dim=1))
            seg_T2[j:j+1,:,:].copy_(torch.argmax(res_T2, dim=1))
            seg_mapping[j:j+1,:,:].copy_(torch.argmax(res_mapping, dim=1))
      
        # post process
        seg_LGE = keepLCC(seg_LGE, 'scar')
        seg_T2 = keepLCC(seg_T2, 'edema')
        seg_mapping = keepLCC(seg_mapping, 'scar')
        seg_pathology = result_transform(seg_LGE, seg_mapping, seg_T2)

        result[:, dim_x//2-args.dim//2:dim_x//2+args.dim//2, dim_y//2-args.dim//2:dim_y//2+args.dim//2].copy_(seg_pathology)
        result = result.numpy().transpose(1,2,0)
        seg_map = nib.Nifti1Image(result, img_affine) 
        nib.save(seg_map, 'test/test_' + str(epoch) + '/' + test_img.iloc[i]["file_name"] + '_result.nii.gz')
        print(test_img.iloc[i]["file_name"] + "_Successfully saved!")


def predict_multiple(args):

    if not os.path.exists('test'):
        os.makedirs('test')

    for root, folder, files in os.walk(args.load_path):
        for file in files:
            model_path = os.path.join(args.load_path, file)
            dice = float(file.split('[')[0])
            epoch = int((file.split('[')[1]).split(']')[0])
            if args.predict_mode == 'single':
                if dice == args.threshold:
                    print('--- Start predicting epoch ' + str(epoch) + ' ---') 
                    predict(args, model_path, epoch)  
                    print('--- Test done for epoch ' + str(epoch) + ' ---')
            if args.predict_mode == 'multiple':
                if dice > args.threshold:
                    print('--- Start predicting epoch ' + str(epoch) + ' ---') 
                    predict(args, model_path, epoch)  
                    print('--- Test done for epoch ' + str(epoch) + ' ---')
    

if __name__ == '__main__':
    args = config()
    predict_multiple(args)
    