# MViTs Excel at Class-agnostic Object Detection

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-modal-transformers-excel-at-class/class-agnostic-object-detection-on-pascal-voc)](https://paperswithcode.com/sota/class-agnostic-object-detection-on-pascal-voc?p=multi-modal-transformers-excel-at-class)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-modal-transformers-excel-at-class/class-agnostic-object-detection-on-coco)](https://paperswithcode.com/sota/class-agnostic-object-detection-on-coco?p=multi-modal-transformers-excel-at-class)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-modal-transformers-excel-at-class/class-agnostic-object-detection-on-kitti)](https://paperswithcode.com/sota/class-agnostic-object-detection-on-kitti?p=multi-modal-transformers-excel-at-class)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-modal-transformers-excel-at-class/class-agnostic-object-detection-on-kitchen)](https://paperswithcode.com/sota/class-agnostic-object-detection-on-kitchen?p=multi-modal-transformers-excel-at-class)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-modal-transformers-excel-at-class/class-agnostic-object-detection-on-comic2k)](https://paperswithcode.com/sota/class-agnostic-object-detection-on-comic2k?p=multi-modal-transformers-excel-at-class)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-modal-transformers-excel-at-class/open-world-object-detection-on-pascal-voc)](https://paperswithcode.com/sota/open-world-object-detection-on-pascal-voc?p=multi-modal-transformers-excel-at-class)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-modal-transformers-excel-at-class/open-world-object-detection-on-coco-2017)](https://paperswithcode.com/sota/open-world-object-detection-on-coco-2017?p=multi-modal-transformers-excel-at-class)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-modal-transformers-excel-at-class/open-world-object-detection-on-coco-2017-1)](https://paperswithcode.com/sota/open-world-object-detection-on-coco-2017-1?p=multi-modal-transformers-excel-at-class)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-modal-transformers-excel-at-class/open-world-object-detection-on-coco-2017-2)](https://paperswithcode.com/sota/open-world-object-detection-on-coco-2017-2?p=multi-modal-transformers-excel-at-class)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-modal-transformers-excel-at-class/object-detection-on-pascal-voc-10)](https://paperswithcode.com/sota/object-detection-on-pascal-voc-10?p=multi-modal-transformers-excel-at-class)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-modal-transformers-excel-at-class/object-detection-on-pascal-voc-2007)](https://paperswithcode.com/sota/object-detection-on-pascal-voc-2007?p=multi-modal-transformers-excel-at-class)

**Multi-modal Vision Transformers Excel at Class-agnostic Object Detection**

[Muhammad Maaz](https://scholar.google.com/citations?user=vTy9Te8AAAAJ&hl=en&authuser=1&oi=sra), [Hanoona Rasheed](https://scholar.google.com/citations?user=yhDdEuEAAAAJ&hl=en&authuser=1&oi=sra), [Salman Khan](https://salman-h-khan.github.io/), [Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en), [Rao Muhammad Anwer](https://scholar.google.com/citations?hl=en&authuser=1&user=_KlvMVoAAAAJ) and [Ming-Hsuan Yang](https://scholar.google.com/citations?user=p9-ohHsAAAAJ&hl=en)


**Paper**: https://arxiv.org/abs/2111.11430

<hr />

![main figure](images/main_figure.png)
> **Abstract:** *What constitutes an object? This has been a long-standing question in computer vision. Towards this goal, numerous learning-free and learning-based approaches have been developed to score objectness. However, they generally do not scale well across new domains and for unseen objects. In this paper, we advocate that existing methods lack a top-down supervision signal governed by human-understandable semantics. To bridge this gap, we explore recent Multi-modal Vision Transformers (MViT) that have been trained with aligned image-text pairs. Our extensive experiments across various domains and novel objects show the state-of-the-art performance of MViTs to localize generic objects in images. Based on these findings, we develop an efficient and flexible MViT architecture using multi-scale feature processing and deformable self-attention that can adaptively generate proposals given a  specific language query. We show the significance of MViT proposals in a diverse range of applications including open-world object detection, salient and camouflage object detection, supervised and self-supervised detection tasks. Further, MViTs offer enhanced interactability with intelligible text queries.* 
<hr />

## Architecture overview of MViTs used in this work

![Architecture overview](images/block_diag.png)

<hr />

## Results
<hr />

<strong>Class-agnostic OD performance of MViTs</strong> in comparison with uni-modal detector (RetinaNet) on several datasets. MViTs show consistently good results on all datasets.

![Results](images/table_results.png)

<hr />

<strong> Enhanced Interactability</strong>: Effect of using different <strong>intuitive text queries</strong> on the MDef-DETR class-agnostic OD performance.
Combining detections from multiple queries captures varying aspects of objectness.

![Results](images/combined_queries_results.png)

<hr />

<strong> Generalization to Rare/Novel Classes</strong>: MDef-DETR class-agnostic OD performance on rarely and frequently occurring categories in the pretraining captions.
The numbers on top of the bars indicate occurrences of the corresponding category in the training dataset.
The MViT achieves good recall values even for the classes with no or very few occurrences.

![Results](images/recall_rare_categories_results.png)

<hr />
<strong> Open-world Object Detection</strong>: Effect of using class-agnostic OD proposals from MDef-DETR for pseudo labelling of unknowns in Open World Detector (ORE).

![Results](images/OWOD_results.png)

<hr />
<strong> Pretraining for Class-aware Object Detection</strong>: Effect of using MDef-DETR proposals for pre-training of DETReg instead of Selective Search proposals.

![Results](images/DETReg_results.png)
<hr />

## Evaluation
The provided codebase contains the pre-computed detections for all datasets using ours MDef-DETR model. The provided directory structure is as follows,

```
-> README.md
-> LICENSE
-> get_eval_metrics.py
-> get_multi_dataset_eval_metrics.py
-> data
    -> voc2007
        -> combined.pkl
    -> coco
        -> combined.pkl
    -> kitti
        -> combined.pkl
    -> kitchen
        -> combined.pkl
    -> cliaprt
        -> combined.pkl
    -> comic
        -> combined.pkl
    -> watercolor
        -> combined.pkl
    -> dota
        -> combined.pkl
```

Where `combined.pkl` contains the combined detections from multiple intutive text queries for corresponding datasets. (Refer Section 5.1: Enhanced Interactability for more details)

Download the [annotations](https://drive.google.com/file/d/14NcjRrDF8AtYI3hwxknIm85fIWhsGdvK/view?usp=sharing) for all datasets and arrange them as shown below. Note that the script expect COCO annotations in standard COCO format & annotations of all other datasets in VOC format.

```
...
...
-> data
    -> voc2007
        -> combined.pkl
        -> Annotations
    -> coco
        -> combined.pkl
        -> instances_val2017_filt.json
    -> kitti
        -> combined.pkl
        -> Annotations
        ...
    -> kitchen
        -> combined.pkl
        -> Annotations
    -> cliaprt
        -> combined.pkl
        -> Annotations
    -> comic
        -> combined.pkl
        -> Annotations
    -> watercolor
        -> combined.pkl
        -> Annotations
    -> dota
        -> combined.pkl
        -> Annotations
```

Once the above mentioned directory structure is created, follow the following steps to calculate the metrics.

1. Install numpy
```
$ pip install numpy
```
2. Calculate metrics
```
$ python get_multi_dataset_eval_metrics.py
```

The calculated metrics will be stored in a `data.csv` file in the same directory.
<hr />

## Citation
If you use our work, please consider citing:

    @article{Maaz2021Multimodal,
        title={Multi-modal Transformers Excel at Class-agnostic Object Detection},
        author={Muhammad Maaz and Hanoona Rasheed and Salman Khan and Fahad Shahbaz Khan and Rao Muhammad Anwer and Ming-Hsuan Yang},
        journal={ArXiv 2111.11430},
        year={2021}
    }

## Contact
Should you have any question, please contact muhammad.maaz@mbzuai.ac.ae or hanoona.bangalath@mbzuai.ac.ae


### :rocket: Note: The repository contains the minimum evaluation code. The complete training and inference scripts along with pretrained models will be released soon. Stay Tuned!
