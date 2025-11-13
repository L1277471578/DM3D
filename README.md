# DM3D: Deformable Mamba via Offset-Guided Gaussian Sequencing for Point Cloud Understanding

## Will do
- ⏳ The paper will be released soon
- ⏳ release the code...
  
We will release the paper and code soon!

## Overview
<img src="![幻灯片2](https://github.com/user-attachments/assets/b443f991-49c9-4856-bae1-97b7a218be31)" width="80%" />


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

##  Training from scratch

To train Mamba3D on ScanObjectNN/Modelnet40 from scratch, run:
```
# Note: change config files for different dataset
python main.py --config cfgs/finetune_modelnet.yaml --finetune_model False --scratch_model True
```

##  Acknowledgement
We would like to thank the authors of [PointMamba](https://github.com/LMD0311/PointMamba), [Mamba3D](https://github.com/xhanxu/Mamba3D), and [Point-MAE](https://github.com/Pang-Yatian/Point-MAE) for their great works and repos.



