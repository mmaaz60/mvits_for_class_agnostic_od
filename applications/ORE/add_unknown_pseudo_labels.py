"""
The script expects the MViT (MDef-DETR or MDETR) detections in .txt format. For example, there should be,
One .txt file for each image and each line in the file represents a detection.
The format of a single detection should be "<label> <confidence> <x1> <y1> <x2> <y2>

Please see the 'mvit_detections' for reference.
"""

import os
import argparse
import xml.etree.ElementTree as ET
from fvcore.common.file_io import PathManager
import numpy as np
import time
import cv2
from nms import nms

TASK1_TRAIN_LIST = "t1_train.txt"
TASK2_TRAIN_LIST = "t2_train.txt"
TASK3_TRAIN_LIST = "t3_train.txt"
TASK4_TRAIN_LIST = "t4_train.txt"


def read_image_list(path):
    with open(path, 'r') as f:
        lines = f.read()
    images = lines.split('\n')

    return images[:-1]


TASK1_TRAIN_IMAGES = read_image_list(TASK1_TRAIN_LIST)
TASK2_TRAIN_IMAGES = read_image_list(TASK2_TRAIN_LIST)
TASK3_TRAIN_IMAGES = read_image_list(TASK3_TRAIN_LIST)
TASK4_TRAIN_IMAGES = read_image_list(TASK4_TRAIN_LIST)

TASK1_KNOWN_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
                       "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
                       "pottedplant", "sheep", "sofa", "train", "tvmonitor", "airplane", "dining table", "motorcycle",
                       "potted plant", "couch", "tv"]
TASK2_KNOWN_CLASSES = TASK1_KNOWN_CLASSES + ["truck", "traffic light", "fire hydrant", "stop sign", "parking meter",
                                             "bench", "elephant", "bear", "zebra", "giraffe",
                                             "backpack", "umbrella", "handbag", "tie", "suitcase",
                                             "microwave", "oven", "toaster", "sink", "refrigerator"]
TASK3_KNOWN_CLASSES = TASK2_KNOWN_CLASSES + ["frisbee", "skis", "snowboard", "sports ball", "kite",
                                             "baseball bat", "baseball glove", "skateboard", "surfboard",
                                             "tennis racket",
                                             "banana", "apple", "sandwich", "orange", "broccoli",
                                             "carrot", "hot dog", "pizza", "donut", "cake"]
TASK4_KNOWN_CLASSES = TASK3_KNOWN_CLASSES + ["bed", "toilet", "laptop", "mouse",
                                             "remote", "keyboard", "cell phone", "book", "clock",
                                             "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
                                             "wine glass", "cup", "fork", "knife", "spoon", "bowl"]


def parse_arguments():
    """
    Parse the command line arguments
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-ann", "--annotations_dir_path", required=True,
                    help="Path to the directory containing the original annotations in pascal VOC format.")
    ap.add_argument("-det", "--detections_dir_path", required=True,
                    help="Path to the directory containing the detections generated using class agnostic object "
                         "detector. One .txt file for each image where each line in the file represents a detection."
                         "The format of a single detection should be "
                         "<label> <confidence> <x1> <y1> <x2> <y2>")
    ap.add_argument("-o", "--output_dir_path", required=True,
                    help="The output dir path to save the updated annotations.")
    ap.add_argument("-det_conf", "--detection_confidence_threshold", required=False, type=float, default=0.5,
                    help="The confidence threshold to filter potential detections at first step. All detections with "
                         "confidence less than this threshold value will be ignored.")
    ap.add_argument("-iou", "--iou_thresh_unk", required=False, type=float, default=0.5,
                    help="All detections, having an overlap greater than iou_thresh with any of the ground truths, "
                         "will be ignored.")
    ap.add_argument("-nms", "--apply_nms", required=False, type=bool, default=False,
                    help="Flag to decide either to apply NMS on detections before assigning them unknown/gt or not.")
    ap.add_argument("-iou_nms", "--iou_thresh_nms", required=False, type=float, default=0.2,
                    help="IOU threshold for NMS.")

    args = vars(ap.parse_args())

    return args


def parse_voc_gt_kn(path):
    image_name = os.path.basename(path).split('.')[0]
    if os.path.exists(path):
        with PathManager.open(path) as f:
            tree = ET.parse(f)
        boxes = []
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            if image_name in TASK1_TRAIN_IMAGES:
                if cls not in TASK1_KNOWN_CLASSES:
                    continue
            elif image_name in TASK2_TRAIN_IMAGES:
                if cls not in TASK2_KNOWN_CLASSES:
                    continue
            elif image_name in TASK3_TRAIN_IMAGES:
                if cls not in TASK3_KNOWN_CLASSES:
                    continue
            elif image_name in TASK4_TRAIN_IMAGES:
                if cls not in TASK4_KNOWN_CLASSES:
                    continue
            else:
                # Not a training image
                return boxes, tree, False
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            boxes.append(bbox)
    else:
        # No annotation file found, create an empty xml node and return
        image_name = f"{os.path.basename(path).split('.')[0]}.jpg"
        image_path = f"{os.path.dirname(os.path.dirname(path))}/JPEGImages/{image_name}"
        img = cv2.imread(image_path)
        h, w, c = img.shape
        node_root = ET.Element('annotation')
        node_folder = ET.SubElement(node_root, 'folder')
        node_folder.text = 'VOC2007'
        node_filename = ET.SubElement(node_root, 'filename')
        node_filename.text = image_name
        node_size = ET.SubElement(node_root, 'size')
        node_width = ET.SubElement(node_size, 'width')
        node_width.text = str(int(w))
        node_height = ET.SubElement(node_size, 'height')
        node_height.text = str(int(h))
        node_depth = ET.SubElement(node_size, 'depth')
        node_depth.text = str(int(c))
        tree = ET.ElementTree(node_root)
        boxes = []

    return boxes, tree, True


def parse_det_txt(path, conf_thresh=0.5):
    if os.path.exists(path):
        with open(path, "r") as f:
            lines = f.readlines()
        boxes = []
        scores = []
        for line in lines:
            content = line.rstrip().split(' ')
            bbox = content[2:]
            # Only keep the boxes with score >= conf_thresh
            det_conf = float(content[1])
            if det_conf >= conf_thresh:
                boxes.append([int(b) for b in bbox])
                scores.append(det_conf)
        return boxes, scores
    else:
        return [], []


def class_agnostic_nms(boxes, scores, iou=0.7):
    # boxes = non_max_suppression_fast(np.array(boxes), iou)
    boxes = nms(np.array(boxes), np.array(scores), iou)
    return list(boxes)


def get_unk_det(gt, det, iou):
    if not gt:
        return det
    gt = np.array(gt)
    unk_det = []
    for dl in det:
        d = np.array(dl)
        ixmin = np.maximum(gt[:, 0], d[0])
        iymin = np.maximum(gt[:, 1], d[1])
        ixmax = np.minimum(gt[:, 2], d[2])
        iymax = np.minimum(gt[:, 3], d[3])
        iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
        ih = np.maximum(iymax - iymin + 1.0, 0.0)
        inters = iw * ih
        uni = (
                (d[2] - d[0] + 1.0) * (d[3] - d[1] + 1.0)
                + (gt[:, 2] - gt[:, 0] + 1.0) * (gt[:, 3] - gt[:, 1] + 1.0)
                - inters
        )
        overlaps = inters / uni
        ov_max = np.max(overlaps)
        if ov_max < iou:
            unk_det.append(dl)
    return unk_det


def main(ann_dir, det_dir, out_dir, det_conf_thesh, iou_thresh, nms=False, iou_thresh_nms=0.7):
    files = os.listdir(det_dir)
    start = time.time()
    for i, file_name in enumerate(files):
        if i % 100 == 0:
            print(f"On image no. {i}. Time: {time.time() - start}")
            start = time.time()
        ann_file_path = f"{ann_dir}/{file_name.split('.')[0]}.xml"
        ref_det_file_path = f"{det_dir}/{file_name.split('.')[0]}.txt"
        out_ann_file_path = f"{out_dir}/{file_name.split('.')[0]}.xml"
        gt_boxes, ann_tree, train = parse_voc_gt_kn(ann_file_path)  # Read the ground truth bounding boxes
        # Only add the unknown detections if training image
        if not train:
            # Copy the original annotation file
            ann_tree.write(out_ann_file_path, encoding='latin-1')
            continue
        det_boxes, scores = parse_det_txt(ref_det_file_path, conf_thresh=det_conf_thesh)  # Read the detections
        if nms:
            det_boxes = class_agnostic_nms(det_boxes, scores, iou_thresh_nms)  # Apply NMS if prompted to do so
        det_unk = get_unk_det(gt_boxes, det_boxes, iou_thresh)  # Get the potential unknown detections
        # Create the updated annotation file
        for det in det_unk:
            object = ET.SubElement(ann_tree.getroot(), 'object')
            name = ET.SubElement(object, "name")
            name.text = "unknown"
            pose = ET.SubElement(object, "pose")
            pose.text = "Unspecified"
            truncated = ET.SubElement(object, "truncated")
            truncated.text = "2"
            difficult = ET.SubElement(object, "difficult")
            difficult.text = "0"
            bndbox = ET.SubElement(object, "bndbox")
            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text = str(int(det[0]))
            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text = str(int(det[1]))
            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text = str(int(det[2]))
            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(int(det[3]))
        # Save the updated annotations
        ann_tree.write(out_ann_file_path, encoding='latin-1')


if __name__ == "__main__":
    args = parse_arguments()
    annotations_dir = args["annotations_dir_path"]
    detections_dir = args["detections_dir_path"]
    output_dir = args["output_dir_path"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    conf_threshold_det = args["detection_confidence_threshold"]
    iou_threshold_unk = args["iou_thresh_unk"]
    apply_nms = args["apply_nms"]
    iou_threshold_nms = args["iou_thresh_nms"]
    main(annotations_dir, detections_dir, output_dir, conf_threshold_det, iou_threshold_unk,
         apply_nms, iou_threshold_nms)
