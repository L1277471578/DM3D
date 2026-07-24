# [DM3D: Dynamic Mamba via Offset-Guided Feature Resampling for Point Cloud Understanding](https://arxiv.org/abs/2512.03424)

## 	📝 To do 
- ✅ Release the [Paper](https://arxiv.org/abs/2512.03424)
- ✅ Release the code
- ⏳ Release the pre-training weights
- ⏳ Update code...
  
If this project is helpful to you, please support it by giving it a ⭐ star and and a 💬 citation:
```
@article{DM3D,
      title={DM3D: Deformable Mamba via Offset-Guided Gaussian Sequencing for Point Cloud Understanding}, 
      author={Bin Liu and Chunyang Wang and Xuelian Liu},
      year={2025},
      journal={arXiv preprint arXiv:2512.03424},
}
```

## Abstract


State Space Models (SSMs) model long token sequences of point cloud with linear complexity, but require an unordered point
cloud to be serialized. Existing methods mainly address this requirement by designing or learning a better token order. Even
a well-constructed order, however, cannot preserve every local relation on an irregular 3D surface: a fixed sequence may still
mix points that are close in index but distant in 3D or belong to different object parts. We propose DM3D, a dynamic Mamba
architecture that preserves the base token order while adapting local feature support and state propagation. First, according to local feature context, DM3D learns spatial and sequence offsets without constructing a global permutation. Then, spatial offsets adjust the sampling anchors in 3D space, whereas sequence offsets guide feature resampling within a local sequence window, which lets different slots draw from overlapping local supports while retaining their identities. This design preserves the global prior of the original traversal, allowing each token to aggregate a more suitable local context. Second, the state update is modulated by the 3D distance between points at adjacent sequence positions, thereby reducing information propagation when these points are spatially far apart. DM3D reaches 95.2% accuracy on the ModelNet40, 93.3% accuracy on the PB T50 RS split of ScanObjectNN, and 84.8% class mIoU on ShapeNetPart. Extensive experiments on benchmark datasets show that DM3D achieves strong and competitive performance, validating the effectiveness of local feature adaptation for point cloud understanding. 

## Overview 
<div align="center">
  <img 
    src="https://github.com/user-attachments/assets/a1f5ff69-40f5-4808-a133-4a56cde22e63" 
    alt="居中图片" 
    width="1036" 
    height="837"
  />
</div>


##  🌈 Requirements

Tested on:   
PyTorch == 1.13.1   python == 3.8    CUDA == 11.7

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

## 🚀 Training
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

##  🎯 Testing 
```
# To test DM3D on ModelNet40, run:
python main.py --config cfgs/finetune_modelnet.yaml --test True  --ckpts output/finetune_scan_objonly/0929-2059/ckpt-best.pth

# Visualization of Part segmentation
python part_segmentation/vis.py 
```

## 📁 Datasets
The overall directory structure should be:
```
DM3D/
├──cfgs/
├──data/
│   ├──ModelNet/
│   ├──ModelNetFewshot/
│   ├──ScanObjectNN/
│   ├──ShapeNet55-34/
├──.......
├──part_segmentation/
│   ├──cfgs/
│   ├──data/
│   ├──├──shapenetcore_partanno_segmentation_benchmark_v0_normal/
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


## 🤝 Acknowledgement
We would like to thank the authors of [PointMamba](https://github.com/LMD0311/PointMamba), [Mamba3D](https://github.com/xhanxu/Mamba3D), and [Point-MAE](https://github.com/Pang-Yatian/Point-MAE) for their great works and repos.

 ​✨ Make the open source world a better place💝




