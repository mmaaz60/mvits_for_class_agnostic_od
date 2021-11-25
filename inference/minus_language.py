import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from infer import Inference

torch.set_grad_enabled(False)


class MinusLanguage(Inference):
    """
    The class supports the inference using both MDETR_Minus_Language & MDef-DETR_Minus_Language models.
    """
    def __init__(self, model, confidence_thresh=0.0):
        Inference.__init__(self, model)
        self.conf_thresh = confidence_thresh
        self.transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @staticmethod
    def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def infer_image(self, image_path, **kwargs):
        # Read the image
        im = Image.open(image_path)
        imq = np.array(im)
        if len(imq.shape) != 3:
            im = im.convert('RGB')
        img = self.transform(im).unsqueeze(0).cuda()
        # propagate through the models
        memory_cache = self.model(img, encode_and_save=True)
        outputs = self.model(img, encode_and_save=False, memory_cache=memory_cache)
        # keep only predictions with self.conf_thresh+ confidence
        probas = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
        keep = (probas > self.conf_thresh).cpu()
        # convert boxes from [0; 1] to image scales
        bboxes_scaled = self.rescale_bboxes(outputs['pred_boxes'].cpu()[0, keep], im.size)
        kept_probs = probas[keep]
        # Convert outputs to the required format
        bboxes = list(bboxes_scaled.numpy())
        probs = list(kept_probs.numpy())
        boxes, scores = [], []
        for b, conf in zip(bboxes, probs):
            boxes.append([int(b[0]), int(b[1]), int(b[2]), int(b[3])])
            scores.append(conf)
        # Read image, perform inference, parse results, append the predicted boxes to detections
        return boxes, scores
