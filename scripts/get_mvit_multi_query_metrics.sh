# The script generates the class-agnostic detections for 'MDef-DETR' model for different queries and then combines the detections from each query.
# The arguments of this script are as follows,
# 1st Argument: path to directory containing the 8 evaluation datasets (Pascal VOC, COCO, Clipart, Comic, Kitchen, KITTI and DOTA)
# 2nd Argument: model checkpoints path

DATASET_BASE_DIR=$1
CHECKPOINTS_PATH=$2
MODEL_NAME=mdef_detr
# For Pascal VOC, COCO, Clipart, Comic and Watercolor datasets
TEXT_QUERIES_SET_1='[all objects,all entities,all visible entities and objects,all obscure entities and objects]'
# For Kitchen, KITTI and DOTA datasets
TEXT_QUERIES_SET_2='[all objects,all entities,all visible entities and objects,all obscure entities and objects,all small objects]'

# Pascal VOC
echo "Running inference on Pascal VOC"
python inference/main_mvit_multi_query.py -m "$MODEL_NAME" -i "$DATASET_BASE_DIR/voc2007/JPEGImages" -c "$CHECKPOINTS_PATH" -tq_list "$TEXT_QUERIES_SET_1"
echo "Combining detections"
python utils/combine_detections.py -i "$DATASET_BASE_DIR/voc2007/$MODEL_NAME"

# COCO
echo "Running inference on COCO"
python inference/main_mvit_multi_query.py -m "$MODEL_NAME" -i "$DATASET_BASE_DIR/coco/val2017" -c "$CHECKPOINTS_PATH" -tq_list "$TEXT_QUERIES_SET_1"
echo "Combining detections"
python utils/combine_detections.py -i "$DATASET_BASE_DIR/coco/$MODEL_NAME"

# KITTI
echo "Running inference on KITTI"
python inference/main_mvit_multi_query.py -m "$MODEL_NAME" -i "$DATASET_BASE_DIR/kitti/JPEGImages" -c "$CHECKPOINTS_PATH" -tq_list "$TEXT_QUERIES_SET_2"
echo "Combining detections"
python utils/combine_detections.py -i "$DATASET_BASE_DIR/kitti/$MODEL_NAME"

# Compute Class-agnostic object detection metrics
echo "Evaluating"
python evaluation/class_agnostic_od/get_multi_dataset_eval_metrics.py -m "$MODEL_NAME"
