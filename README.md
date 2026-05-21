## AdaptViG: Adaptive Vision GNN with Exponential Decay Gating

WACV 2026

[PDF](https://openaccess.thecvf.com/content/WACV2026/papers/Munir_AdaptViG_Adaptive_Vision_GNN_with_Exponential_Decay_Gating_WACV_2026_paper.pdf) | [Arxiv](https://arxiv.org/abs/2511.09942)

Mustafa Munir, Md Mostafijur Rahman, and Radu Marculescu

# Overview
This repository contains the source code for AdaptViG: Adaptive Vision GNN with Exponential Decay Gating


# Pretrained Models

Weights trained on ImageNet-1K can be downloaded [here](https://huggingface.co/SLDGroup/AdaptViG/tree/main). 

Weights trained on COCO 2017 Object Detection and Instance Segmentation can be downloaded [here](https://huggingface.co/SLDGroup/AdaptViG/tree/main/Detection). 

Weights trained on ADE20K Semantic Segmentation can be downloaded [here](https://huggingface.co/SLDGroup/AdaptViG/tree/main/Sem_Segmentation).

### detection
Contains all of the object detection and instance segmentation backbone code and config.

### segmentation
Contains all of the semantic segmentation backbone code and config.

### models
Contains the main model code.

### util
Contains utility scripts used.

# Usage

## Installation Image Classification

```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```
```
conda install mpi4py
```
```
pip install -r requirements.txt
```

## Image Classification

### Train image classification:
```
python -m torch.distributed.launch --nproc_per_node=num_GPUs --nnodes=num_nodes --use_env main.py --data-path /path/to/imagenet --model adaptvig_model --output_dir results
```

### Test image classification:
```
python -m torch.distributed.launch --nproc_per_node=num_GPUs --nnodes=num_nodes --use_env main.py --data-path /path/to/imagenet --model adaptvig_model --resume pretrained_model --eval
```


## Installation Object Detection and Instance Segmentation
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```
```
pip install timm
```
```
pip install submitit
```
```
pip install -U openmim
```
```
mim install mmcv-full
```
```
mim install mmdet==2.28
```
## Object Detection and Instance Segmentation

Detection and instance segmentation on MS COCO 2017 is implemented based on [MMDetection](https://github.com/open-mmlab/mmdetection). We follow settings and hyper-parameters of [PVT](https://github.com/whai362/PVT/tree/v2/segmentation), [PoolFormer](https://github.com/sail-sg/poolformer), and [EfficientFormer](https://github.com/snap-research/EfficientFormer) for comparison. 


### Data preparation

Prepare COCO 2017 dataset according to the [instructions in MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md#test-existing-models-on-standard-datasets).


### Train object detection and instance segmentation:
```
python -m torch.distributed.launch --nproc_per_node num_GPUs --nnodes=num_nodes --node_rank 0 main.py configs/mask_rcnn_adaptvig_model --model adaptvig_model --work-dir Output_Directory --launcher pytorch > Output_Directory/log_file.txt 
```


### Test object detection and instance segmentation:
```
python -m torch.distributed.launch --nproc_per_node=num_GPUs --nnodes=num_nodes --node_rank 0 test.py configs/mask_rcnn_adaptvig_model --checkpoint Pretrained_Model --eval {bbox or segm} --work-dir Output_Directory --launcher pytorch > log_file.txt
```



## Installation Semantic Segmentation
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```
```
pip install -U openmim
mim install mmengine
mim install mmcv-full
```
```
mim install "mmsegmentation <=0.30.0"
```

## Semantic Segmentation

Semantic segmentation on ADE20K is implemented based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). We follow settings and hyper-parameters of [PVT](https://github.com/whai362/PVT/tree/v2/segmentation), [PoolFormer](https://github.com/sail-sg/poolformer), and [EfficientFormer](https://github.com/snap-research/EfficientFormer) for comparison. 


### Train semantic segmentation:

8 GPUs, 40K Iterations
```
python -m torch.distributed.launch --nproc_per_node 8 --nnodes 1 --node_rank 0 train.py configs/sem_fpn/fpn_adaptvig_m_ade20k_40k.py --model adaptvig_model --work-dir semantic_results/ --launcher pytorch > semantic_results/run_semantic.txt
```


### Citation

```
@InProceedings{AdaptViG_WACV,
  title={AdaptViG: Adaptive Vision GNN with Exponential Decay Gating},
  author={Munir, Mustafa and Rahman, Md Mostafijur and Marculescu, Radu},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={440--450},
  year={2026}
}
```