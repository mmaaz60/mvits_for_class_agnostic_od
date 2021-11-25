"""
The script converts the predictions saved in .txt format to single .pkl file.
"""

import os
import argparse
import numpy as np
import pickle


def parse_det_txt(path, top_N=50):
    if os.path.exists(path):
        with open(path, "r") as f:
            lines = f.readlines()
        boxes = []
        scores = []
        for line in lines:
            content = line.rstrip().split(' ')
            bbox = content[2:]
            boxes.append([int(b) for b in bbox])
            scores.append(float(content[1]))

        if top_N < len(boxes):
            # Select only the top-N boxes
            scores = np.array(scores)
            boxes = np.array(boxes)
            sorted_ind = np.argsort(-scores)
            sorted_ind = sorted_ind[:top_N]
            boxes = boxes[sorted_ind, :]
            scores = scores[sorted_ind]
            boxes = boxes.tolist()
            scores = scores.tolist()

        return boxes, scores
    else:
        return [], []


def parse_arguments():
    """
    Parse the command line arguments
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_path", required=True,
                    help="Path to the directory containing containing predictions in .txt format. One .txt per image.")
    ap.add_argument("-N", "--number_of_top_boxes", required=False, type=int, default=50,
                    help="Total number of top boxes to be considered for each each image.")
    args = vars(ap.parse_args())

    return args


def main():
    # Parse arguments
    args = parse_arguments()
    input_dir_path = args["input_path"]
    top_boxes = args["number_of_top_boxes"]

    files = os.listdir(input_dir_path)
    file_to_boxes = {}
    for file in files:
        file_to_boxes[file.split('.')[0]] = parse_det_txt(f"{input_dir_path}/{file}", top_N=top_boxes)
    # Save the predictions
    output_file_path = f"{input_dir_path}.pkl"
    with open(output_file_path, "wb") as f:
        pickle.dump(file_to_boxes, f)


if __name__ == "__main__":
    main()
