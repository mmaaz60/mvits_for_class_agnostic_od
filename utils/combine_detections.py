import os
import numpy as np
import argparse
import pickle
from nms import nms


def class_agnostic_nms(boxes, scores, iou=0.7):
    if len(boxes) > 1:
        boxes, scores = nms(np.array(boxes), np.array(scores), iou)
        return list(boxes), list(scores)
    else:
        return boxes, scores


def parse_det_pkl(path):
    with open(path, "rb") as f:
        file_to_boxes_dict = pickle.load(f)
    return file_to_boxes_dict


def parse_arguments():
    """
    Parse the command line arguments
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_dir_path", required=True,
                    help="The path to the directory containing the annotations for separate queries.")
    ap.add_argument("-iou", "--nms_iou_theshold", required=False, type=float, default=0.5,
                    help="The iou threshold used to merge the detections.")
    args = vars(ap.parse_args())

    return args


def main():
    args = parse_arguments()
    input_dir_path = args["input_dir_path"]
    nms_iou_theshold = args["nms_iou_theshold"]
    pkl_files = [name for name in os.listdir(input_dir_path) if os.path.isfile(os.path.join(input_dir_path, name))]
    tq_to_dets = []
    for file in pkl_files:
        tq_to_dets.append(parse_det_pkl(f"{input_dir_path}/{file}"))
    combined_img_to_boxes = {}
    image_names = tq_to_dets[0].keys()
    for img in image_names:
        all_boxes = []
        all_scores = []
        for tq_to_det in tq_to_dets:
            boxes, scores = tq_to_det[img]
            all_boxes += boxes
            all_scores += scores
        combined_img_to_boxes[img] = class_agnostic_nms(all_boxes, all_scores, nms_iou_theshold)
    # Save the combined detections
    output_path = f"{input_dir_path}/combined.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(combined_img_to_boxes, f)


if __name__ == "__main__":
    main()
