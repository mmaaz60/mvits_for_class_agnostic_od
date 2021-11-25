DATASET_PATH=/home/maaz/PycharmProjects/MViTS_for_class_agnostic_OD/data/class_agnostic_OD
MODEL_NAME=mdeformable_detr_r101

# Pascal VOC
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/VOC2007/$MODEL_NAME/all_objects" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/VOC2007/$MODEL_NAME/all_entities" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/VOC2007/$MODEL_NAME/all_visible_entities_and_objects" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/VOC2007/$MODEL_NAME/all_obscure_entities_and_objects" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/VOC2007/$MODEL_NAME/combined" -N 50

# MS COCO
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/coco/$MODEL_NAME/all_objects" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/coco/$MODEL_NAME/all_entities" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/coco/$MODEL_NAME/all_visible_entities_and_objects" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/coco/$MODEL_NAME/all_obscure_entities_and_objects" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/coco/$MODEL_NAME/combined" -N 50

# Clipart
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/clipart/$MODEL_NAME/all_objects" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/clipart/$MODEL_NAME/all_entities" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/clipart/$MODEL_NAME/all_visible_entities_and_objects" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/clipart/$MODEL_NAME/all_obscure_entities_and_objects" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/clipart/$MODEL_NAME/combined" -N 50

# Comic
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/comic/$MODEL_NAME/all_objects" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/comic/$MODEL_NAME/all_entities" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/comic/$MODEL_NAME/all_visible_entities_and_objects" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/comic/$MODEL_NAME/all_obscure_entities_and_objects" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/comic/$MODEL_NAME/combined" -N 50

# Dota
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/dota/$MODEL_NAME/all_objects" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/dota/$MODEL_NAME/all_entities" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/dota/$MODEL_NAME/all_visible_entities_and_objects" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/dota/$MODEL_NAME/all_obscure_entities_and_objects" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/dota/$MODEL_NAME/all_small_objects" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/dota/$MODEL_NAME/combined" -N 50

# Kitchen
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/kitchen/$MODEL_NAME/all_objects" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/kitchen/$MODEL_NAME/all_entities" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/kitchen/$MODEL_NAME/all_visible_entities_and_objects" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/kitchen/$MODEL_NAME/all_obscure_entities_and_objects" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/kitchen/$MODEL_NAME/all_small_objects" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/kitchen/$MODEL_NAME/combined" -N 50

# KITTI
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/KITTI/$MODEL_NAME/all_objects" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/KITTI/$MODEL_NAME/all_entities" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/KITTI/$MODEL_NAME/all_visible_entities_and_objects" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/KITTI/$MODEL_NAME/all_obscure_entities_and_objects" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/KITTI/$MODEL_NAME/all_small_objects" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/KITTI/$MODEL_NAME/combined" -N 50

# Watercolor
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/watercolor/$MODEL_NAME/all_objects" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/watercolor/$MODEL_NAME/all_entities" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/watercolor/$MODEL_NAME/all_visible_entities_and_objects" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/watercolor/$MODEL_NAME/all_obscure_entities_and_objects" -N 50
python scripts/prediction_txts_to_pkl.py -i "$DATASET_PATH/watercolor/$MODEL_NAME/combined" -N 50