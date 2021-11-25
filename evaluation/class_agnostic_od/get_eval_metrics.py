"""
The script computes the VOC style AP@50 and Recall@50 for class agnostic object detector.
It can read ground truths from either Pascal VOC or COCO format annotations. The detections should be a dictionary with
keys as 'image names' and values as the tuple of predicted boxes & scores (i.e. ([boxes], [scores])).

"""

import argparse
import pickle
import time
import xml.etree.ElementTree as ET
import numpy as np
import json
import csv


PERCENTAGE_SMALL = 0.05
PERCENTAGE_MEDIUM = 0.2


def parse_voc_rec(filename, metric_type="all"):
    """Parse a PASCAL VOC xml file."""
    with open(filename, 'r') as f:
        tree = ET.parse(f)
    # Get the image width and height
    size = tree.find("size")
    img_w, img_h = size.find("width").text, size.find("height").text
    img_area = float(img_w) * float(img_h)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        obj_struct["difficult"] = 0
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(float(bbox.find("xmin").text)),
            int(float(bbox.find("ymin").text)),
            int(float(bbox.find("xmax").text)),
            int(float(bbox.find("ymax").text)),
        ]
        box_area = (obj_struct["bbox"][2] - obj_struct["bbox"][0]) * (obj_struct["bbox"][3] - obj_struct["bbox"][1])
        if metric_type == "small":
            if box_area <= PERCENTAGE_SMALL * img_area:
                objects.append(obj_struct)
        elif metric_type == "medium":
            if PERCENTAGE_SMALL * img_area <= box_area <= PERCENTAGE_MEDIUM * img_area:
                objects.append(obj_struct)
        elif metric_type == "large":
            if box_area >= PERCENTAGE_MEDIUM * img_area:
                objects.append(obj_struct)
        else:
            objects.append(obj_struct)

    return objects


def parse_coco_annotations(filename, metric_type="all"):
    dataset = json.load(open(filename, 'r'))
    assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
    img_id_to_name = {}
    img_id_to_wh = {}
    for img in dataset['images']:
        img_id_to_name[img['id']] = img["file_name"]
        img_id_to_wh[img['id']] = (img["width"], img["height"])
    recs = {}
    for ann in dataset["annotations"]:
        image_id = ann["image_id"]
        img_area = float(img_id_to_wh[image_id][0]) * float(img_id_to_wh[image_id][1])
        image_name = img_id_to_name[image_id].split('.')[0]
        if image_name not in recs.keys():
            recs[image_name] = []
        bb = ann['bbox']
        x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
        box_area = (x2 - x1) * (y2 - y1)
        if metric_type == "small":
            if box_area <= PERCENTAGE_SMALL * img_area:
                recs[image_name].append({"name": 'object', "difficult": 0, "bbox": [x1, y1, x2, y2]})
        elif metric_type == "medium":
            if PERCENTAGE_SMALL * img_area <= box_area <= PERCENTAGE_MEDIUM * img_area:
                recs[image_name].append({"name": 'object', "difficult": 0, "bbox": [x1, y1, x2, y2]})
        elif metric_type == "large":
            if box_area >= PERCENTAGE_MEDIUM * img_area:
                recs[image_name].append({"name": 'object', "difficult": 0, "bbox": [x1, y1, x2, y2]})
        else:
            recs[image_name].append({"name": 'object', "difficult": 0, "bbox": [x1, y1, x2, y2]})

    return recs


def parse_det_pkl(path):
    with open(path, "rb") as f:
        file_to_boxes_dict = pickle.load(f)
    return file_to_boxes_dict


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(dets_dir_path, ann_path, ovthresh=0.5, N=50, ann_type="voc", use_07_metric=False, ap_type="all"):
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name

    # Read dets
    image_ids = []
    confidence = []
    BB = []
    file_to_boxes_dict = parse_det_pkl(dets_dir_path)
    for key in file_to_boxes_dict.keys():
        boxes, scores = file_to_boxes_dict[key]
        # Select top-N boxes
        scores = np.array(scores)
        boxes = np.array(boxes)
        sorted_ind = np.argsort(-scores)
        sorted_ind = sorted_ind[:N]
        boxes = boxes[sorted_ind, :]
        scores = scores[sorted_ind]
        boxes = boxes.tolist()
        scores = scores.tolist()

        for b, s in zip(boxes, scores):
            image_ids.append(key)
            confidence.append(s)
            BB.append(b)

    # Load gt
    # read list of images
    imagenames = file_to_boxes_dict.keys()

    # load annots
    if ann_type == "coco":
        if ap_type == "small":
            recs = parse_coco_annotations(ann_path, metric_type="small")
        elif ap_type == "medium":
            recs = parse_coco_annotations(ann_path, metric_type="medium")
        elif ap_type == "large":
            recs = parse_coco_annotations(ann_path, metric_type="large")
        else:
            recs = parse_coco_annotations(ann_path)
    else:
        recs = {}
        for imagename in imagenames:
            if ap_type == "small":
                recs[imagename] = parse_voc_rec(f"{ann_path}/{imagename.split('.')[0]}.xml", metric_type="small")
            elif ap_type == "medium":
                recs[imagename] = parse_voc_rec(f"{ann_path}/{imagename.split('.')[0]}.xml", metric_type="medium")
            elif ap_type == "large":
                recs[imagename] = parse_voc_rec(f"{ann_path}/{imagename.split('.')[0]}.xml", metric_type="large")
            else:
                recs[imagename] = parse_voc_rec(f"{ann_path}/{imagename.split('.')[0]}.xml")
    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        try:
            R = [obj for obj in recs[imagename]]
        except KeyError:
            continue
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
        # difficult = np.array([False for x in R]).astype(np.bool)  # treat all "difficult" as GT
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    confidence = np.array(confidence)
    BB = np.array(BB).reshape(-1, 4)

    # Average boxes/dets per image
    avg_dets_per_image = len(BB) / len(imagenames)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        try:
            R = class_recs[image_ids[d]]
        except KeyError:
            continue
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                    (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                    + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                    - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap, avg_dets_per_image


def parse_arguments():
    """
    Parse the command line arguments
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-ann", "--annotations_path", required=True,
                    help="Path to the directory containing the annotation files or path to the single annotation file.")
    ap.add_argument("-ann_type", "--annotations_type", required=False, default="voc",
                    help="Annotations type ('voc', 'coco').")
    ap.add_argument("-det", "--detections_dir_path", required=True,
                    help="Path to the '.pkl' file containing detections generated from class agnostic OD method. The "
                         "detections should be a dictionary with keys as 'image names' and values as the tuple of "
                         "predicted boxes & scores (i.e. ([boxes], [scores]))")
    ap.add_argument("-N", "--top_N_dets", required=False, type=int, default=50,
                    help="Maximum number of top N detections sorted by confidence to be used for metrics calculations. "
                         "Note that the script also reports average number of ")
    ap.add_argument("-iou", "--iou_thresh", required=False, type=float, default=0.5,
                    help="IOU threshold to be used for computing AP and Recall. Default is 0.5.")
    ap.add_argument("--extra_metrics", action='store_true',
                    help="Flag to decide if to evaluate AP-small, AP-medium and AP-large.")

    args = vars(ap.parse_args())

    return args


def main():
    args = parse_arguments()
    ann_dir_path = args["annotations_path"]
    ann_type = args["annotations_type"]
    dets_dir_path = args["detections_dir_path"]
    top_N_boxes = args["top_N_dets"]
    iou_threshold = args["iou_thresh"]
    extra_metrics = args["extra_metrics"]

    start = time.time()
    rec, prec, ap, avg_dets_per_image = {}, {}, {}, {}
    rec["all"], prec["all"], ap["all"], avg_dets_per_image["all"] = \
        voc_eval(dets_dir_path, ann_dir_path, iou_threshold,
                 top_N_boxes, ann_type=ann_type)
    if extra_metrics:
        rec["small"], prec["small"], ap["small"], avg_dets_per_image["small"] = \
            voc_eval(dets_dir_path, ann_dir_path, iou_threshold,
                     top_N_boxes, ann_type=ann_type, ap_type="small")
        rec["medium"], prec["medium"], ap["medium"], avg_dets_per_image["medium"] = \
            voc_eval(dets_dir_path, ann_dir_path, iou_threshold,
                     top_N_boxes, ann_type=ann_type, ap_type="medium")
        rec["large"], prec["large"], ap["large"], avg_dets_per_image["large"] = \
            voc_eval(dets_dir_path, ann_dir_path, iou_threshold,
                     top_N_boxes, ann_type=ann_type, ap_type="large")
    running_time = time.time() - start
    # Save the results in a .csv file
    header = ['Type', 'Average Boxes per Image', 'AP@50', 'Recall@50', 'Precission@50']
    with open(f"{dets_dir_path.split('.')[0]}.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for key in rec.keys():
            metrics = [key, avg_dets_per_image[key], ap[key] * 100, rec[key][-1] * 100, prec[key][-1] * 100]
            writer.writerow(metrics)
            # Print the results on command line as well
            print(f"{key}:")
            print(f"Processing Time: {running_time} seconds.")
            print(f"Average Boxes per Image: {avg_dets_per_image[key]}")
            print(f"AP@{iou_threshold * 100}: {ap[key] * 100}\n"
                  f"Recall@{iou_threshold * 100}: {rec[key][-1] * 100}\n"
                  f"Precision@{iou_threshold * 100}: {prec[key][-1] * 100}\n")


if __name__ == "__main__":
    main()
