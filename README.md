# MyoPS-Net
PyTorch implemetation for our MedIA paper "MyoPS-Net: Myocardial Pathology Segmentation with Flexible Combination of Multi-Sequence CMR Images" (https://doi.org/10.1016/j.media.2022.102694).

<img decoding="async" src="structure.png">

## Introduction
Myocardial pathology segmentation (MyoPS) can be a prerequisite for the accurate diagnosis and treatment planning of myocardial infarction. However, achieving this segmentation is challenging, mainly due to the inadequate and indistinct information from an image. In this work, we develop an end-to-end deep neural network, referred to as MyoPS-Net, to flexibly combine five-sequence cardiac magnetic resonance (CMR) images for MyoPS. 

The proposed MyoPS-Net were evaluated on two datasets, i.e., a private one consisting of 50 paired multi-sequence CMR images and a public one from MICCAI2020 MyoPS Challenge (https://zmiclab.github.io/zxh/0/myops20/). Experimental results showed that MyoPS-Net could achieve state-of-the-art performance in various scenarios. Note that in practical clinics, the subjects may not have full sequences, such as missing LGE CMR or mapping CMR scans. We therefore conducted extensive experiments to investigate the performance of the proposed method in dealing with such complex combinations of different CMR sequences. Results proved the superiority and generalizability of MyoPS-Net, and more importantly, indicated a practical clinical application.

## Usage
### Dataset
Please organize the dataset as the following structure:
```
data/
  -- train_set/
     -- train_image/
     -- train_gd/
  -- val_set/     
     -- val_image/
     -- val_gd/
  -- test_set/
     -- test_image/
  -- train.txt (with each line: image_path gd_path z_index)
  -- validation.txt (with each line: image_path gd_path z_index)
  -- test.csv (with each line: image_path stage dx dy dz)
```
### Train
```
python main.py --path "data_path" --batch_size 16 --dim 192 --lr 1e-4 --threshold 0.50 --end_epoch 200
```

### Predict
```
python predict.py --load_path checkpoints/xxx.pth --predict_mode single --threshold 0.50 --dim 192
```

## Citation and Acknowledge
If you make use of the code, or if you found the code useful, please cite this paper in any resulting publications.
```
@article{qiu2022myops,
title = {MyoPS-Net: Myocardial pathology segmentation with flexible combination of multi-sequence CMR images},
journal = {Medical Image Analysis},
pages = {102694},
year = {2022},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2022.102694},
author = {Junyi Qiu and Lei Li and Sihan Wang and Ke Zhang and Yinyin Chen and Shan Yang and Xiahai Zhuang}
}
```
J. Qiu, L. Li, S. Wang et al., MyoPS-Net: Myocardial pathology segmentation with flexible combination of multi-sequence CMR images. Medical Image Analysis(2022), doi: https://doi.org/10.1016/j.media.2022.102694.

