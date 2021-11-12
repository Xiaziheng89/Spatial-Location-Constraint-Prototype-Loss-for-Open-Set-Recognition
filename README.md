# Spatial Location Constraint Prototype Loss for Open Set Recognition
Official PyTorch implementation of
["**Spatial Location Constraint Prototype Loss for Open Set Recognition**"](https://arxiv.org/abs/2110.11013). 

More our open set recognition article and code information can be seen in https://arxiv.org/abs/2108.04225.

## 1. Requirements
### Environments
These codes are supposed to be run with a Linux system. If you use Windows system to run them,
it may encounter some errors.

Currently, requires following packages
- python 3.6+
- torch 1.4+
- torchvision 0.5+
- CUDA 10.1+
- scikit-learn 0.22+

### Datasets
For Tiny-ImageNet, please download the following datasets to ```./data/tiny_imagenet``` and unzip it.
-   [tiny_imagenet](https://drive.google.com/file/d/1oJe95WxPqEIWiEo8BI_zwfXDo40tEuYa/view?usp=sharing)

## 2. Training 

### Open Set Recognition
To train open set recognition models in paper, run this command:
```train
python osr.py --dataset <DATASET> --loss <LOSS>
```
> Option 
> 
> --loss can be one of SLCPLoss/GCPLoss/Softmax.
> 
> --dataset is one of mnist/svhn/cifar10/cifar100/tiny_imagenet.



## 3. Results
### We visualize the deep feature of Softmax/GCPL/SLCPL as below.

<p align="center">
    <img src="/home/xiaziheng/Desktop/SLCPL code for Github/img/SLCPL_colorful_MNIST.png" width="400">
</p>

Before getting the figure above, you need to train the LeNet++ network, whose architecture is in "./models/model.py".


## Citation
- If you find our work or the code useful, please consider cite our paper using:
```bibtex
@misc{xia2021spatial,
      title={Spatial Location Constraint Prototype Loss for Open Set Recognition}, 
      author={Ziheng Xia and Ganggang Dong and Penghui Wang and Hongwei Liu},
      year={2021},
      eprint={2110.11013},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
