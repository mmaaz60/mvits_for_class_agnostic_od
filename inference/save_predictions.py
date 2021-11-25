import os
import pickle


class SavePredictions:
    def __init__(self):
        self.predictions = None

    def save(self, save_path):
        raise NotImplementedError


class SaveTxtFormat(SavePredictions):
    def __init__(self):
        SavePredictions.__init__(self)

    def update(self, predictions):
        self.predictions = predictions

    def save(self, save_path):
        for i, image_name in enumerate(self.predictions.keys()):
            with open(f"{save_path}/{image_name.split('.')[0]}.txt", 'w+') as f:
                boxes, scores = self.predictions[image_name]
                for b, c in zip(boxes, scores):
                    # class_id, score, x1, y1, x2, y2
                    f.write(f"{0} {c} {int(b[0])} {int(b[1])} {int(b[2])} {int(b[3])}\n")


class SavePKLFormat(SavePredictions):
    def __init__(self):
        SavePredictions.__init__(self)

    def update(self, predictions):
        self.predictions = predictions

    def save(self, save_path):
        img_to_boxes = {}
        if os.path.exists(save_path):
            with open(save_path, "rb") as f:
                img_to_boxes = pickle.load(f)
        img_to_boxes = {**img_to_boxes, **self.predictions}
        with open(save_path, "wb") as f:
            pickle.dump(img_to_boxes, f)
