# Evaluation
We provide instructions to reproduce class-agnostic object detection results of 
MDef-DETR with and without language branch. 
Please refer to Tables 1, 2, 4 & 5 of our [paper](https://arxiv.org/abs/2111.11430) for more details.

[comment]: <> (The dataset, pretrained models and pre-computed predictions are available at [this link]&#40;https://shortest.link/1Rka&#41;.)
Download the datasets (annotations & images) and arrange them as,
```
code_root/
└─ data
    └─ voc2007
        ├─ Annotations
        ├─ JPEGImages
    └─ coco
        ├─ instances_val2017.json
        ├─ val2017
    └─ kitti
        ├─ Annotations
        ├─ JPEGImages
    └─ kitchen
        ├─ Annotations
        ├─ JPEGImages
    └─ cliaprt
        ├─ Annotations
        ├─ JPEGImages
    └─ comic
        ├─ Annotations
        ├─ JPEGImages
    └─ watercolor
        ├─ Annotations
        ├─ JPEGImages
    └─ dota
        ├─ Annotations
        ├─ JPEGImages
```

Once the above directory structure is created,
1. Download the pretrained weights from [this link](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/muhammad_maaz_mbzuai_ac_ae/Erw0P4D7bDdKtl4BstL8XFsBv-k2W7Ya9rKOaZBIOrTEcQ?e=pL2VgF).
2. Set the environment variable
```shell
export PYTHONPATH="./:$PYTHONPATH"
```
3. Run the following script to generate predictions and calculate metrics.
   1. MDef-DETR
    ```shell
    bash scripts/get_mvit_multi_query_metrics.sh <dataset root dir path> <model checkpoints path> 
    ```
   2. MDef-DETR w/o Language Branch (trained by maintaining the structure introduced by captions)
    ```shell
    bash scripts/get_mvit_minus_language_metrics.sh <dataset root dir path> <model checkpoints path> 
    ```

[comment]: <> (Alternatively, you can also download the pre-computed predictions from [this link]&#40;https://shortest.link/1Rka&#41; )

[comment]: <> (and run the following scripts to calculate metrics. )

[comment]: <> (```shell)

[comment]: <> (python evaluation/class_agnostic_od/get_multi_dataset_eval_metrics.py <model name>)

[comment]: <> (```)

The calculated evaluation metrics will be stored in a `*.csv` file in the same directory.