import os
import argparse
from multiprocessing import Pool
from models.model import Model
from inference.main import run_inference


def parse_arguments():
    """
    Parse the command line arguments
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--models", required=False, default="mdef_detr",
                    help="The models to be used for performing inference. Available options are,"
                         "['mdef_detr']")
    ap.add_argument("-i", "--input_images_dir_path", required=True,
                    help="The path to input images directory on which to run inference.")
    ap.add_argument("-c", "--model_checkpoints_path", required=False, default=None,
                    help="The path to models checkpoints. Required for all models except mdetr.")
    ap.add_argument("-tq_list", "--text_queries_list", required=False, default='[all objects,all entities,'
                                                                               'all visible entities and objects,'
                                                                               'all obscure entities and objects]',
                    help="The list of text queries to be used for generating predictions using MViTs.")
    ap.add_argument("-p", "--no_processes", required=False, type=int, default=2,
                    help="Number of parallel processes.")
    ap.add_argument("--multi_crop", action='store_true', help="Either to perform multi-crop inference or not. "
                                                              "Multi-crop inference is used only for DOTA dataset.")

    args = vars(ap.parse_args())

    return args


args = parse_arguments()
model_name = args["models"]
assert model_name in ['mdef_detr', 'mdetr']  # Currently supported MViTs
images_dir = args["input_images_dir_path"]
checkpoints_path = args["model_checkpoints_path"]
tq_list = args["text_queries_list"]
num_processes = args["no_processes"]
multi_crop = args["multi_crop"]


def run_mvit_inference(model_name, checkpoints_path, images_dir, output_path, caption=None, multi_crop=False):
    model = Model(model_name, checkpoints_path).get_model()
    run_inference(model, images_dir, output_path, caption, multi_crop)


def run(text_query):
    for tq in text_query:
        output_dir = f"{os.path.dirname(images_dir)}/{model_name}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/{'_'.join(tq.split(' '))}.pkl"
        run_mvit_inference(model_name, checkpoints_path, images_dir, output_path, caption=tq, multi_crop=multi_crop)


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def main():
    tq_list_parsed = map(str, tq_list.strip('[]').split(','))
    tq_list_chunks = chunkIt(list(tq_list_parsed), num_processes)

    with Pool(num_processes) as p:
        p.map(run, tq_list_chunks)


if __name__ == "__main__":
    main()
