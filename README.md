# MViTs Excel at Class-agnostic Object Detection
Multi-modal Vision Transformers Excel at Class-agnostic Object Detection

## Evaluation
* The proided codebase contains the pre-computed detections for all datasets using ours MDef-DETR model. The provided directory structure is as follows,

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

Where `combined.pkl` contains the combined detections from multiple intutive text queries for corresponding datasets (refer to section 5.1 for details).

Download the annotations for all datasets and arrange them as shown below. In our work, we downloaded all these annotations from the [open-source](http://www.svcl.ucsd.edu/projects/universal-detection) resource. Note that the script expect COCO annotations in standard COCO format & annotations of all other datasets in VOC format.

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


### :rocket: Note: The repository contains the minimum evaluation code. The complete training and inference scripts along with pretrained models will be released soon. Stay Tuned!
