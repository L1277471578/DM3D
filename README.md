# DM3D: Deformable Mamba via Offset-Guided Gaussian Sequencing for Point Cloud Understanding

## Will do
- ⏳ The paper will be released soon
- ⏳ release the code...
  
We will release the paper and code soon!

## Overview




<div align="center">
  <img 
    src="https://github.com/user-attachments/assets/fec2c940-c6bd-48be-9c5e-9c90bc454ec1"
    alt="居中图片" 
    width="576" 
    height="465"
  />
</div>


##   Requirements

Tested on:   
PyTorch == 1.13.1   
python == 3.8    
CUDA == 11.7

```
pip install -r requirements.txt
```

```
# Chamfer Distance & emd
cd ./extensions/chamfer_dist
python setup.py install --user
cd ./extensions/emd
python setup.py install --user
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

# Mamba install
pip install causal-conv1d==1.1.1
pip install mamba-ssm==1.1.1
```

## Training
###  Training from scratch
To train DM3D from scratch, run:
```
## Classification on ModelNet40
python main.py --config cfgs/finetune_modelnet.yaml --test False --ckpts None --finetune_model False --scratch_model True 

## Classification on ScanObjectNN
python main.py --config cfgs/finetune_scan_objbg.yaml --test False --ckpts None --finetune_model False --scratch_model True

## Classification on ModelNet40-fow-shot
python main.py --config cfgs/finetune_modelnet.yaml --test False --ckpts None --finetune_model False --scratch_model False

## Partseg on Shapenetpart
python part_segmentation/train_partseg.py --config cfgs/config.yaml --pretrain_weight None 
```

###  Training from fine-tuning
 Like Scratch, only the parameters need to be modified,  For example: Classification on ModelNet40
```
python main.py --config cfgs/finetune_modelnet.yaml --test False  --ckpts cfgs/pretrain_pointmae.pth --finetune_model True --scratch_model False 
```

##  Testing 
To test DM3D on ModelNet40, run:
```
python main.py --config cfgs/finetune_modelnet.yaml --test True  --ckpts output/finetune_scan_objonly/0929-2059/ckpt-best.pth 
```

## Datasets
The overall directory structure should be:
```
│Point-MAE/
├──cfgs/
├──data/
│   ├──ModelNet/
│   ├──ModelNetFewshot/
│   ├──ScanObjectNN/
│   ├──ShapeNet55-34/
│   ├──shapenetcore_partanno_segmentation_benchmark_v0_normal/
├──datasets/
├──.......
```

### ModelNet40 Dataset: 
```
│ModelNet/
├──modelnet40_normal_resampled/
│  ├── modelnet40_shape_names.txt
│  ├── modelnet40_train.txt
│  ├── modelnet40_test.txt
│  ├── modelnet40_train_8192pts_fps.dat
│  ├── modelnet40_test_8192pts_fps.dat
```
Download: You can download the processed data from [Point-BERT repo](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md), or download from the [official website](https://modelnet.cs.princeton.edu/#) and process it by yourself.

### ModelNet Few-shot Dataset:
```
│ModelNetFewshot/
├──5way10shot/
│  ├── 0.pkl
│  ├── ...
│  ├── 9.pkl
├──5way20shot/
│  ├── ...
├──10way10shot/
│  ├── ...
├──10way20shot/
│  ├── ...
```
Download: Please download the data from [Point-BERT repo](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md). We use the same data split as theirs.

### ScanObjectNN Dataset:
```
│ScanObjectNN/
├──main_split/
│  ├── training_objectdataset_augmentedrot_scale75.h5
│  ├── test_objectdataset_augmentedrot_scale75.h5
│  ├── training_objectdataset.h5
│  ├── test_objectdataset.h5
├──main_split_nobg/
│  ├── training_objectdataset.h5
│  ├── test_objectdataset.h5
```
Download: Please download the data from the [official website](https://hkust-vgd.github.io/scanobjectnn/).

### ShapeNetPart Dataset:
```
|shapenetcore_partanno_segmentation_benchmark_v0_normal/
├──02691156/
│  ├── 1a04e3eab45ca15dd86060f189eb133.txt
│  ├── .......
│── .......
│──train_test_split/
│──synsetoffset2category.txt
```
Download: Please download the data from [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip). 


##  Acknowledgement
We would like to thank the authors of [PointMamba](https://github.com/LMD0311/PointMamba), [Mamba3D](https://github.com/xhanxu/Mamba3D), and [Point-MAE](https://github.com/Pang-Yatian/Point-MAE) for their great works and repos.






