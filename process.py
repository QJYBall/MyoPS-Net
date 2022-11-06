import os
import torch
import numpy as np
import nibabel as nib
from skimage import measure


class LargestConnectedComponents(object):
    def __call__(self, mask, mode):
        
        mask = mask.numpy()
        
        # keep a heart connectivity 
        heart_slice = np.where((mask>0), 1, 0)
        out_heart = np.zeros(heart_slice.shape, dtype=np.uint8)
        for struc_id in [1]:
            binary_img = (heart_slice == struc_id)
            blobs = measure.label(binary_img, connectivity=1)
            props = measure.regionprops(blobs)
            if not props:
                continue
            area = [ele.area for ele in props]
            largest_blob_ind = np.argmax(area)
            largest_blob_label = props[largest_blob_ind].label
            out_heart[blobs == largest_blob_label] = struc_id
        
        # keep LV/MYO connectivity
        if mode == 'cardiac':
            out_img = np.zeros(mask.shape, dtype=np.uint8)
            for struc_id in [1,2]:
                binary_img = mask == struc_id
                blobs = measure.label(binary_img, connectivity=1)
                props = measure.regionprops(blobs)
                if not props:
                    continue
                area = [ele.area for ele in props]
                largest_blob_ind = np.argmax(area)
                largest_blob_label = props[largest_blob_ind].label
                out_img[blobs == largest_blob_label] = struc_id
            final_img = out_heart * out_img
        
        if mode == 'scar' or mode == 'edema':
            final_img = out_heart * mask

        final_img = torch.from_numpy(final_img).float()
    
        return final_img


class PostProcess(object):
    def __init__(self):
        super(PostProcess, self).__init__()

    def continues_region_extract_scar(self, label, threshold_1):
        numbers = []
        label_pp = label.copy()
        for i in range(label_pp.shape[2]):
            label_i = label_pp[:,:,i]
            regions = np.where(label_i==2221, np.ones_like(label_i), np.zeros_like(label_i))
            L_i, n_i = measure.label(regions, background=0, connectivity=1, return_num=True)

            for j in range(1, n_i + 1):
                num_j = np.sum(L_i == j)
                numbers.append(num_j)
                if num_j < threshold_1:
                    bbx_h, bbx_w = np.where(L_i==j)
                    bbx_h_min = bbx_h.min()
                    bbx_h_max = bbx_h.max()
                    bbx_w_min = bbx_w.min()
                    bbx_w_max = bbx_w.max()
                    roi = label_i[bbx_h_min-1:bbx_h_max+2, bbx_w_min-1:bbx_w_max+2]
                    replace_lable = np.argmax(np.bincount(roi[roi!=2221].flatten()))

                    label_pp[:,:,i] = np.where(L_i==j, replace_lable*np.ones_like(label_i), label_i)

        return numbers, label_pp
    
    def continues_region_extract_edema(self, label, threshold_2):
        numbers = []
        label_pp = label.copy()
        for i in range(label_pp.shape[2]):
            label_i = label_pp[:,:,i]
            regions = np.where(label_i>=1220, np.ones_like(label_i), np.zeros_like(label_i))
            L_i, n_i = measure.label(regions, background=0, connectivity=1, return_num=True)

            for j in range(1, n_i + 1):
                num_j = np.sum(L_i == j)
                numbers.append(num_j)
                if num_j < threshold_2:
                    bbx_h, bbx_w = np.where(L_i==j)
                    bbx_h_min = bbx_h.min()
                    bbx_h_max = bbx_h.max()
                    bbx_w_min = bbx_w.min()
                    bbx_w_max = bbx_w.max()
                    roi = label_i[bbx_h_min-1:bbx_h_max+2, bbx_w_min-1:bbx_w_max+2]
                    replace_lable = np.argmax(np.bincount(roi[roi<1220].flatten()))

                    label_pp[:,:,i] = np.where(L_i==j, replace_lable*np.ones_like(label_i), label_i)

        return numbers, label_pp

    def __call__(self, img_path, thre_1, thre_2):
        for root, _, files in os.walk(img_path):
            for i in sorted(files):
                if i[-2:] != 'gz' or i[0]=='P':
                    continue
                i_file = root +'/'+ i
                predNII = nib.load(i_file)
                label = predNII.get_fdata().astype('int64')

                _, label_pp = self.continues_region_extract_scar(label, thre_1)
                _, label_pp = self.continues_region_extract_edema(label_pp, thre_2)

                label_pp = label_pp.astype(np.int16)

                label_pp = nib.Nifti1Image(label_pp, affine=predNII.affine)
                name = img_path + '/PP_' + i
                seg_save_p = os.path.join(name)
                nib.save(label_pp, seg_save_p)
                pass
