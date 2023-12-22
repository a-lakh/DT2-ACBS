# Pytorch Implemetation of DT2-ACBS

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/a-lakh/DT2-ACBS/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.2.0-%237732a8)

This repository contains the source code of our ICCV 2021 paper, **Learning of Visual Relations: The Devil is in the Tails**.  

[[Project Website]](http://www.svcl.ucsd.edu/projects/DT2-ACBS/)
[[Paper]](https://arxiv.org/pdf/2108.09668.pdf)

## Contents

1. [Overview](#Overview)
2. [Install the Requirements](INSTALL.md)
3. [Prepare the Dataset](DATASET.md)
4. [Training on Scene Graph Generation](#perform-training-on-scene-graph-generation)
5. [Evaluation on Scene Graph Generation](#Evaluation)
6. [Citations](#Citations)

## Overview


## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.

## Dataset

Check [DATASET.md](DATASET.md) for instructions of dataset preprocessing.

## Pretrained Models
To keep the training process faster, we provide pretrained model weights of ResNet trained on Visual Genome dataset, which is used by default to start training the model in stage-1. This can be found in the pretrained_model folder under the name "resnet101_VG.pth". We also provided the model of DT2-ACBS which gave us the best results after the two-stage training. It can also be found in the pretrained_model folder under the name "updated_distill_KD_gnd_a0.2_t10_last_model.pth".

Both of these models can be downloaded from [Here](https://drive.google.com/drive/folders/1r5LShP-oomJx3z_xb1Y-xXMb-ZflRxyD?usp=sharing)


## Training on Scene Graph Generation

There are **three standard taks**: (1) Predicate Classification (PredCls): taking ground truth bounding boxes and entity labels as inputs, (2) Scene Graph Classification (SGCls) : using ground truth bounding boxes without entity labels, (3) Scene Graph Detection (SGDet): detecting SGs from scratch. 

However, the code-base doesn't perform objet detection as our paper deicded not to focus on the detection problem as it has been widely studied. Hence, there is no seperate training option for Scene Graph Detection (SGDet). The argument ```--train_task``` is used to select the task for training the model and ```--init_weight_path``` can be used to provide model weights to start the training.


### Predicate Classification (PredCls)
The network is trained in two stages. These stages follow diffrent sampling strategies. (1) Standard Random Sampling (SRS): sampling images uniformly, independent of their class labels - both feature extractor (φ) and output softmax layer parameters W are jointly learned (2) Class Balanced Sampling (CBS): sampling uniformly over classes such that all classes are present equally - the feature extractor (φ) is fixed and the output softmax layer parameters W are relearned. The argument ```--stage``` is used to select the sampling stage.

For **Standard Random Sampling (SRS)**:
``` bash
python train.py --train_task 1 --stage 1
```
For **Class Balanced Sampling (CBS)**:
``` bash
python train.py --train_task 1 --stage 2
```

### Scene Graph Classification (SGCls)
The network is trained in two stages. These stages follow diffrent sampling strategies. (1) Standard Random Sampling (SRS): sampling images uniformly, independent of their class labels - both feature extractors (θ,φ,ψ) and output softmax layer parameters W^e and W^p are learned for both predicate and object classification independently (2) Alternating Class Balanced Sampling (ACBS): uniform alternate sampling over classes from predicate distribtion and object distrbution - the feature extractors (θ,φ,ψ) are fixed and the output softmax layer parameters W^e and W^p are relearned through knowledge distilation between alternate steps. The argument ```--stage``` is used to select the sampling stage. We also provide the option of using the type of object labels you want to feed the predicate classifier while training. (1) Type=0: Use Object ground truth labels for predicate classification (2) Type=1: Use Object classifier labels for predicate classification. The argument ```--type``` is used to pick type of object labels.

In adittion, the ACBS stage requires a few hyper-parameters ```--alpha```, ```--beta```, ```--temperature``` which are by default set to the optimum values of 0.2, 1 and 10 respectively. You can also use weight matrix based knowledge transfer in ACBS stage by enabling ```--non_soft_label_kd```.


For **Standard Random Sampling (SRS)**:
``` bash
python train.py --train_task 2 --stage 1
```
For **Alternating Class Balanced Sampling (ACBS)** with object ground truth labels for predicate classification:
``` bash
python train.py --train_task 2 --stage 2 --type 0
```
For **Alternating Class Balanced Sampling (ACBS)** with object classifier labels for predicate classifion:
``` bash
python train.py --train_task 2 --stage 2 --type 1
```


### Other Training Arguments

#### General Training Arguments
Flags |  Description
-- | -- 
`--experiment_root`       |       root where to store models, losses and accuracies|
`--cuda`                  |       enables cuda    
`--batch_size`            |       number of samples within each batch, default=256
`--epochs`                |       number of epochs to train for, default=20
`--init_weight_path`      |       path to the initial model weights, default=None



#### Scheduler Arguments
Flags |  Description
-- | -- 
`--learning_rate`         |       learning rate for the model, default=0.001
`--lr_scheduler_step`     |       StepLR learning rate scheduler step, default=20
`--lr_scheduler_gamma`    |       StepLR learning rate scheduler gamma, default=0.5

#### Sampler Argumets
Flags |  Description
-- | -- 
`--iterations_train`      |       number of train episodes per epoch, default=100
`--iterations_val`        |       number of validation episodes per epoch, default=10
`--pred_classes_per_it`   |       number of random classes for predicate sampling based training, default=49 (max_val=49)
`--obj_classes_per_it`    |       number of random classes object sampling based training, default=150  (max_val=150)
`--pred_num_samples`      |       number of samples per class to use predicate sampling based training, default=5
`--obj_num_samples`       |       number of samples per class to use predicate sampling based training, default=2


## Evaluation
There are **three standard taks**: (1) Predicate Classification (PredCls): taking ground truth bounding boxes and entity labels as inputs, (2) Scene Graph Classification (SGCls) : using ground truth bounding boxes without entity labels, (3) Scene Graph Detection (SGDet): detecting SGs from scratch. 

However, the code-base doesn't perform object detection as our paper deicded not to focus on the detection problem as it has been widely studied. Hence, we use already detcted bounding boxes results stored as "object_detection_info.json" in the dataset folder. This file contains object bounding boxes from mask-rcnn trained on Visual Genome dataset. The argument ```--test_task``` is used to select the task for evaluating the model and ```--init_weight_path``` is used to provide model weights.


For **Predicate Classification (PredCls)**:
``` bash
python test.py --test_task 1
```
For **Scene Graph Classification (SGCls)**:
``` bash
python test.py --test_task 2
```
For **Scene Graph Detection (SGDet)**:
``` bash
python test.py --test_task 3
```

## Citations

Please kindly consider citing our project or paper in your publications, if this project helps your research.

```
@inproceedings{Desai_and_Wu_dt2-acbs_21,
	title={Learning of Visual Relations: The Devil is in the Tails},
	author={Alakh Desai and Tz-Ying Wu and Subarna Tripathi and Nuno Vasconcelos},
	booktitle={IEEE International Conference on Computer Vision (ICCV)},
	year={2021}
}
```
