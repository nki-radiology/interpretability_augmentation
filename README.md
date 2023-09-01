# Interpretability-guided Data Augmentation

This repository contains the code of the paper **Interpretability-guided Data Augmentation for Robust Segmentation in Multi-centre Colonoscopy Data**, accepted at the [MLMI workshop at MICCAI 2023](https://sites.google.com/view/mlmi2023).

A pre-print of the paper is available at [this link](http://arxiv.org/abs/2308.15881).

### Graphical Abstract

![all_in_one_figure](NewSDNet/figures/all_in_one_figure.png)

### Requirements

All experiments were carried out using the following:
```
albumentations==1.3.0
captum==0.6.0
hydra-core==1.3.2
kornia==0.6.12
lightning==2.0.0
monai==1.1.0
numpy==1.23.0
omegaconf==2.3.0
pandas==1.4.3
psutil==5.9.4
scikit-image==0.19.2
scikit-learn==1.1.1
torch==1.13.1
torchmetrics==0.11.4
torchvision==0.14.1
wandb==0.13.9
```

### Dataset
All experiments were conducted on the open-source PolypGen dataset. You can ask for access to the dataset at this [link](https://www.synapse.org/#!Synapse:syn26376615/wiki/613312).

### Training

To launch the training of one of the available models run the following:

```
python /PATH_TO_REPO/train.py 
```
All configuration parameters are handles by [Hydra](https://hydra.cc/), to run different experiments modify the config files available in the folder `conf`. Please refer to Hydra docs to see how to over-ride configuration parameters directly from the command line. 

### To-do
* Remove artifact creation in W&B logging

### Aknowledgments

* Thanks to [spthermo](https://github.com/spthermo/SDNet/tree/master) for the original Pytorch implementation of [SDNet](https://www.sciencedirect.com/science/article/abs/pii/S1361841519300684).
* Thanks to [sharib-vision](https://github.com/sharib-vision/PolypGen-Benchmark) for the original Pytorch implementations of UNet and DeepLabV3+.
* Thanks to [DebeshJha](https://github.com/DebeshJha/PolypGen) for the original dataset repository. 
