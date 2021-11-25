# The script generates the class-agnostic detections for 'MDef-DETR Minus Language' model.
# The arguments of this script are as follows,
# 1st Argument: path to directory containing the 8 evaluation datasets (Pascal VOC, COCO, Clipart, Comic, Kitchen, KITTI and DOTA)
# 2nd Argument: model checkpoints path

DATASET_BASE_DIR=$1
CHECKPOINTS_PATH=$2
MODEL_NAME=mdef_detr_minus_language

# Pascal VOC
echo "Running inference on Pascal VOC"
python inference/main.py -m "$MODEL_NAME" -i "$DATASET_BASE_DIR/voc2007/JPEGImages" -c "$CHECKPOINTS_PATH"

# COCO
echo "Running inference on COCO"
python inference/main.py -m "$MODEL_NAME" -i "$DATASET_BASE_DIR/coco/val2017" -c "$CHECKPOINTS_PATH"

# KITTI
echo "Running inference on KITTI"
python inference/main.py -m "$MODEL_NAME" -i "$DATASET_BASE_DIR/kitti/JPEGImages" -c "$CHECKPOINTS_PATH"

# Compute Class-agnostic object detection metrics
echo "Evaluating"
python evaluation/class_agnostic_od/get_multi_dataset_eval_metrics.py -m "$MODEL_NAME"
