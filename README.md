# [Temporal_Query_Networks for Fine-grained Video Understanding](https://www.robots.ox.ac.uk/~vgg/research/tqn/)

📋 This repository contains the implementation of CVPR2021 paper [Temporal_Query_Networks for Fine-grained Video Understanding](https://arxiv.org/pdf/2104.09496.pdf)

# Abstract

<p float="center">
  <img src="https://www.robots.ox.ac.uk/~vgg/research/tqn/teaser.jpg" />
</p>

Our objective in this work is fine-grained classification of actions in untrimmed videos, where the actions may be temporally extended or may span only a few frames of the video. We cast this into a query-response mechanism, where each query addresses a particular question, and has its own response label set.

We make the following four contributions: (i) We propose a new model — a Temporal Query Network — which enables the query-response functionality, and a structural undertanding of fine-grained actions. It attends to relevant segments for each query with a temporal attention mechanism, and can be trained using only the labels for each query. (ii) We propose a new way — stochastic feature bank update — to train a network on videos of various lengths with the dense sampling required to respond to fine-grained queries. (iii) we compare the TQN to other architectures and text supervision methods, and analyze their pros and cons. Finally, (iv) we evaluate the method extensively on the FineGym and Diving48 benchmarks for fine-grained action classification and surpass the state-of-the-art using only RGB features.

# Getting Started
1. Clone this repository
```
git clone https://github.com/Chuhanxx/Temporal_Query_Networks.git
```
2. Create conda virtual env and install the requirements  
(This implementation requires CUDA and python > 3.7)
```
cd Temporal_Query_Networks
source build_venv.sh
```

# Prepare Data and Weight Initialization

Please refer to [data.md](https://github.com/Chuhanxx/Temporal_Query_Networks/blob/main/data/data.md) for data preparation. 


# Training 
you can start training the model with the following steps, taking the Diving48 dataset as an example,:

1. First stage training:
Set the paths in the `Diving48_first_stage.yaml` config file first, and then run:

```
cd scripts
python train_1st_stage.py --name $EXP_NAME --dataset diving48 --dataset_config ../configs/Diving48_first_stage.yaml --gpus 0,1 --batch_size 16  
```
2. Construct stochastically updated feature banks:

```
python construct_SUFB.py --dataset diving48 --dataset_config ../configs/Diving48_first_stage.yaml \
--gpus 0  --resume_file  $PATH_TO_BEST_FILE_FROM_1ST_STAGE --out_dir $DIR_FOR_SAVING_FEATURES 
```

3. Second stage training:
Set the paths in the `Diving48_second_stage.yaml` config file first, and then run:

```
python train_2nd_stage.py --name $EXP_NAME  --dataset diving48  \
--dataset_config ../configs/Diving48_second_stage.yaml   \
--batch_size 16 --gpus 0,1
```

# Test

```
python test.py --name $EXP_NAME  --dataset diving48 --batch_size 1 \
--dataset_config ../configs/Diving48_second_stage.yaml 
```

# Citation

If you use this code etc., please cite the following paper:

```
@inproceedings{zhangtqn,
  title={Temporal Query Networks for Fine-grained Video Understanding},
  author={Chuhan Zhang and Ankush Gputa and Andrew Zisserman},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```

If you have any question, please contact czhang@robots.ox.ac.uk .
