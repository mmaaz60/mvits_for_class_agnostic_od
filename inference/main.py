import os
import argparse
from tqdm import tqdm
from models.model import Model
from inference.save_predictions import SavePKLFormat


def parse_arguments():
    """
    Parse the command line arguments
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--models", required=False, default="mdef_detr",
                    help="The models to be used for performing inference. Available options are,"
                         "['mdef_detr','mdef_detr_minus_language']")
    ap.add_argument("-i", "--input_images_dir_path", required=True,
                    help="The path to input images directory on which to run inference.")
    ap.add_argument("-c", "--model_checkpoints_path", required=False, default=None,
                    help="The path to models checkpoints. Required for all models except mdetr.")
    ap.add_argument("-tq", "--text_query", required=False, default="all objects",
                    help="The text query to be used in case of MViTs.")
    ap.add_argument("--multi_crop", action='store_true', help="Either to perform multi-crop inference or not. "
                                                              "Multi-crop inference is used only for DOTA dataset.")

    args = vars(ap.parse_args())

    return args


def run_inference(model, images_dir, output_path, caption=None, multi_crop=False):
    images = os.listdir(images_dir)
    dumper = SavePKLFormat()
    detections = {}
    for i, image_name in enumerate(tqdm(images)):
        if i > 0 and i % 500 == 0:  # Checkpoints after every 500 iterations
            dumper.update(detections)
            dumper.save(output_path)
            detections = {}
        image_path = f"{images_dir}/{image_name}"
        # Note: Caption is only rquired for MViTs
        if multi_crop:
            detections[image_name.split('.')[0]] = model.infer_image_multi_crop(image_path, caption=caption)
        else:
            detections[image_name.split('.')[0]] = model.infer_image(image_path, caption=caption)
    dumper.update(detections)
    dumper.save(output_path)


def main():
    # Parse arguments
    args = parse_arguments()
    model_name = args["models"]
    images_dir = args["input_images_dir_path"]
    checkpoints_path = args["model_checkpoints_path"]
    text_query = args["text_query"]
    multi_crop = args["multi_crop"]
    model = Model(model_name, checkpoints_path).get_model()
    output_dir = f"{os.path.dirname(images_dir)}/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    if model_name in ['mdef_detr']:
        output_path = f"{output_dir}/{'_'.join(text_query.split(' '))}.pkl"
    else:
        output_path = f"{output_dir}/{model_name}.pkl"
    run_inference(model, images_dir, output_path, caption=text_query, multi_crop=multi_crop)


if __name__ == "__main__":
    main()
